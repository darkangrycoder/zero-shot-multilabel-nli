"""
dataset.py  --  Data loading, NLI inference, Dataset, evaluation helpers
=========================================================================

Usage::

    from dataset import load_data, run_nli_inference
    from dataset import MultiLabelDataset, multi_collate
    from dataset import find_best_thresholds, print_metrics
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (f1_score, hamming_loss, accuracy_score,
                              precision_score, recall_score)
from tqdm.auto import tqdm

from model import LABEL_NAMES, SPECIALIST_LABELS, LABEL_DESCRIPTIONS

LABEL_DESCS = list(LABEL_DESCRIPTIONS.values())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path, train_split=0.8, seed=42):
    """
    Load synthetic_data.json and split into train / val sets.

    Expected format::

        [{"text": "...", "labels": ["Finance", "Economy"]}, ...]

    Args:
        path        : path to synthetic_data.json
        train_split : fraction for training (default 0.8)
        seed        : random seed for reproducibility

    Returns:
        (train_data, val_data) -- lists of dicts
    """
    with open(path) as f:
        raw_data = json.load(f)

    random.seed(seed)
    indices = list(range(len(raw_data)))
    random.shuffle(indices)
    split      = int(len(indices) * train_split)
    train_data = [raw_data[i] for i in indices[:split]]
    val_data   = [raw_data[i] for i in indices[split:]]

    print(f'Loaded {len(raw_data)} samples  train: {len(train_data)}  val: {len(val_data)}')
    return train_data, val_data


# ---------------------------------------------------------------------------
# NLI inference
# ---------------------------------------------------------------------------

def run_nli_inference(data_items, nli_classifier, batch_size=16):
    """
    Run DeBERTa zero-shot NLI over a list of data items.

    Args:
        data_items     : list of dicts with 'text' key
        nli_classifier : loaded transformers zero-shot-classification pipeline
        batch_size     : inference batch size

    Returns:
        list of {label: score} dicts aligned with data_items
    """
    texts = [d['text'] for d in data_items]
    all_scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc='NLI Inference'):
        batch = texts[i: i + batch_size]
        raw   = nli_classifier(batch, candidate_labels=LABEL_DESCS, multi_label=True)
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            score_map = {}
            for desc, score in zip(item['labels'], item['scores']):
                name = [k for k, v in LABEL_DESCRIPTIONS.items() if v == desc][0]
                score_map[name] = round(score, 6)
            all_scores.append(score_map)
    return all_scores


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def negative_sampling(batch_labels, all_labels, max_num_negatives=10):
    """
    Sample random negative labels not in batch_labels.
    Hardens training by exposing the model to near-miss labels.

    Args:
        batch_labels      : list of lists of positive label strings per sample
        all_labels        : full label vocabulary
        max_num_negatives : max negatives to sample per example

    Returns:
        list of lists of negative label strings per sample

    Example::

        neg = negative_sampling([['Economy', 'Finance']], LABEL_NAMES, 3)
        # -> [['Technology', 'Health', 'Environment']]
    """
    num_negatives = random.randint(1, max_num_negatives)
    negative_samples = []
    for labels in batch_labels:
        candidates = [l for l in all_labels if l not in labels]
        neg = random.sample(candidates, min(num_negatives, len(candidates)))
        negative_samples.append(neg)
    return negative_samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MultiLabelDataset(Dataset):
    """
    PyTorch Dataset for multi-label text classification.

    Stores texts and pre-computed NLI scores (avoids re-running the heavy
    DeBERTa NLI inference every epoch).

    Args:
        data_items        : list of {text, labels}
        nli_scores        : list of {label: float} from run_nli_inference
        specialist_labels : labels that have PolyEncoder heads
    """

    def __init__(self, data_items, nli_scores, specialist_labels):
        self.texts             = [d['text'] for d in data_items]
        self.nli_scores_raw    = nli_scores
        self.specialist_labels = specialist_labels
        self.targets = {
            l: [float(l in d['labels']) for d in data_items]
            for l in specialist_labels
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text'   : self.texts[idx],
            'nli_raw': self.nli_scores_raw[idx],
            'targets': {l: self.targets[l][idx] for l in self.specialist_labels},
        }


def multi_collate(batch):
    """
    DataLoader collate function.
    Keeps texts as strings (tokenised inside the model), stacks label targets.

    Returns:
        texts    : list[str]
        nli_raws : list[dict]
        targets  : dict {label: (B,) float tensor}
    """
    texts    = [b['text'] for b in batch]
    nli_raws = [b['nli_raw'] for b in batch]
    targets  = {
        l: torch.tensor([b['targets'][l] for b in batch], dtype=torch.float32)
        for l in batch[0]['targets']
    }
    return texts, nli_raws, targets


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def build_matrices(data_items, score_dicts, threshold=0.5):
    """Build (y_true, y_pred) arrays with a single fixed threshold."""
    y_true, y_pred = [], []
    for item, scores in zip(data_items, score_dicts):
        gt = set(item['labels'])
        pr = {l for l in LABEL_NAMES if scores.get(l, 0) >= threshold}
        y_true.append([int(l in gt) for l in LABEL_NAMES])
        y_pred.append([int(l in pr) for l in LABEL_NAMES])
    return np.array(y_true), np.array(y_pred)


def build_matrices_per_thresh(data_items, score_dicts, thresholds):
    """Build (y_true, y_pred) arrays with per-label thresholds."""
    y_true, y_pred = [], []
    for item, scores in zip(data_items, score_dicts):
        gt = set(item['labels'])
        pr = {l for l in LABEL_NAMES if scores.get(l, 0) >= thresholds.get(l, 0.5)}
        y_true.append([int(l in gt) for l in LABEL_NAMES])
        y_pred.append([int(l in pr) for l in LABEL_NAMES])
    return np.array(y_true), np.array(y_pred)


def find_best_thresholds(data_items, score_dicts):
    """
    Grid-search the F1-maximising threshold per label on a held-out set.

    Args:
        data_items  : validation items with ground-truth labels
        score_dicts : model score dicts aligned with data_items

    Returns:
        dict {label: best_threshold}
    """
    grid   = np.arange(0.01, 0.99, 0.01)
    yt     = np.array([[int(l in d['labels']) for l in LABEL_NAMES] for d in data_items])
    sc     = np.array([[s.get(l, 0.0)         for l in LABEL_NAMES] for s in score_dicts])
    result = {}

    print(f'  {"Label":<17}  {"Best Thresh":>12}  {"Best F1":>8}')
    print(f'  {"-"*17}  {"-"*12}  {"-"*8}')
    for i, label in enumerate(LABEL_NAMES):
        f1s    = [f1_score(yt[:, i], (sc[:, i] >= t).astype(int), zero_division=0) for t in grid]
        best_t = float(grid[int(np.argmax(f1s))])
        best_f = float(np.max(f1s))
        result[label] = best_t
        mark = '*' if label in SPECIALIST_LABELS else ' '
        print(f'  {mark} {label:<15}  {best_t:>12.2f}  {best_f:>8.4f}')

    return result


def print_metrics(y_true, y_pred, title='Results'):
    """Print a full metric report and return as a dict."""
    macro_f1   = f1_score(y_true, y_pred, average='macro',   zero_division=0)
    ham_acc    = 1 - hamming_loss(y_true, y_pred)
    exact_m    = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec  = recall_score(y_true, y_pred,    average='macro', zero_division=0)
    per_label  = f1_score(y_true, y_pred, average=None, zero_division=0)

    print(f'\n{"="*60}\n  {title}\n{"="*60}')
    print(f'  Macro F1        : {macro_f1:.4f}')
    print(f'  Hamming Acc     : {ham_acc:.4f}')
    print(f'  Exact Match     : {exact_m:.4f}')
    print(f'  Macro Precision : {macro_prec:.4f}')
    print(f'  Macro Recall    : {macro_rec:.4f}')
    print(f'\n  Per-label F1:')
    for label, f1 in zip(LABEL_NAMES, per_label):
        bar  = '|' * int(f1 * 30)
        mark = '*' if label in SPECIALIST_LABELS else ' '
        print(f'  {mark} {label:<15} {f1:.4f}  {bar}')

    return {
        'macro_f1'   : macro_f1,  'hamming_acc': ham_acc,
        'exact_match': exact_m,   'macro_prec' : macro_prec,
        'macro_rec'  : macro_rec,
        'per_label'  : dict(zip(LABEL_NAMES, per_label)),
    }
