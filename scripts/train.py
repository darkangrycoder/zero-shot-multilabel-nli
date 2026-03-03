"""
scripts/train.py  --  Training script for HybridNLI-PolyEncoder v2
===================================================================

Usage (from repo root)::

    python scripts/train.py
    python scripts/train.py --epochs 30 --lr 1e-3
    python scripts/train.py --no_hub          # skip HuggingFace upload
    HF_TOKEN=hf_xxx python scripts/train.py   # auto-upload after training

The script reads config.yaml for defaults; every config value can be
overridden via CLI flags.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import pipeline
from tqdm.auto import tqdm

try:
    import yaml
    def _load_yaml(path):
        with open(path) as f:
            return yaml.safe_load(f)
except ImportError:
    def _load_yaml(path):
        raise ImportError("pyyaml is required: pip install pyyaml")

# Allow imports from repo root when called as scripts/train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (MultiHeadHybridV2, LABEL_NAMES, SPECIALIST_LABELS,
                   save_checkpoint, push_to_hub)
from dataset import (load_data, run_nli_inference,
                     MultiLabelDataset, multi_collate,
                     build_matrices, build_matrices_per_thresh,
                     find_best_thresholds, print_metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train HybridNLI-PolyEncoder v2')
    p.add_argument('--config',        default='config.yaml',
                   help='path to config.yaml (default: config.yaml)')
    p.add_argument('--data',          default=None,
                   help='override synthetic_data_path in config')
    p.add_argument('--epochs',        type=int,   default=None)
    p.add_argument('--batch_size',    type=int,   default=None)
    p.add_argument('--lr',            type=float, default=None)
    p.add_argument('--kl_weight',     type=float, default=None)
    p.add_argument('--blend_weight',  type=float, default=None)
    p.add_argument('--patience',      type=int,   default=None)
    p.add_argument('--ckpt_dir',      default=None,
                   help='override checkpoint output directory')
    p.add_argument('--no_hub',        action='store_true',
                   help='skip HuggingFace Hub upload')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_multihead_v2(
    model,
    train_data, train_nli,
    val_data,   val_nli,
    num_epochs   = 25,
    batch_size   = 16,
    lr           = 3e-4,
    kl_weight    = 0.3,
    nli_conf_thr = 0.75,
    blend_weight = 1.0,
    patience     = 7,
):
    """
    Train MultiHeadHybridV2 with dual-loss per specialist label.

    Per-label loss at each step::

        poly_loss  = BCE(poly_logits,   true_label)   # trains PolyHead weights
        blend_loss = BCE(blended_score, true_label)   # trains alpha directly (v2 fix)
        kl_loss    = KL(poly || nli)  on samples where NLI is confident

    Total = sum over labels (poly_loss + blend_weight*blend_loss + kl_weight*kl_loss)

    Returns:
        history dict with keys: best_macro_f1, train_losses, val_f1_hist, alpha_history
    """
    from sklearn.metrics import f1_score as _f1

    train_ds = MultiLabelDataset(train_data, train_nli, model.specialist_labels)
    val_ds   = MultiLabelDataset(val_data,   val_nli,   model.specialist_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=multi_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=multi_collate)

    optimizer = AdamW([
        {'params': model.poly_heads.parameters(), 'lr': lr},
        {'params': model.alpha_raws.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.02)
    bce_logit = nn.BCEWithLogitsLoss()
    bce_prob  = nn.BCELoss()

    best_f1, best_state, no_improve = 0.0, None, 0
    train_losses, val_f1_hist       = [], []
    alpha_history = {l: [] for l in model.specialist_labels}

    n_train = (sum(p.numel() for p in model.poly_heads.parameters()) +
               sum(p.numel() for p in model.alpha_raws.parameters()))
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {n_train:,} / {n_total:,}  ({100*n_train/n_total:.2f}%)')
    print(f'Epochs={num_epochs}  batch={batch_size}  lr={lr}  '
          f'kl={kl_weight}  blend={blend_weight}  patience={patience}')

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for texts, nli_raws, targets in train_loader:
            optimizer.zero_grad()
            blended, poly_logits, nli_tensors = model(texts, nli_raws)
            loss = torch.tensor(0.0, device=model.device)

            for label in model.specialist_labels:
                tgt     = targets[label].to(model.device)
                nli_s   = nli_tensors[label]
                blend_s = blended[label]

                poly_loss  = bce_logit(poly_logits[label], tgt)
                blend_loss = bce_prob(blend_s.clamp(1e-7, 1 - 1e-7), tgt)

                conf = (nli_s > nli_conf_thr) | (nli_s < (1 - nli_conf_thr))
                if conf.sum() > 0:
                    pp   = torch.sigmoid(poly_logits[label][conf]).clamp(1e-7, 1 - 1e-7)
                    np_  = nli_s[conf].clamp(1e-7, 1 - 1e-7)
                    kl_loss = F.kl_div(
                        torch.log(torch.stack([pp, 1 - pp], dim=1)),
                        torch.stack([np_, 1 - np_], dim=1),
                        reduction='batchmean',
                    )
                else:
                    kl_loss = torch.tensor(0.0, device=model.device)

                loss = loss + poly_loss + blend_weight * blend_loss + kl_weight * kl_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # -- validation --
        model.eval()
        val_scores = []
        with torch.no_grad():
            for texts, nli_raws, _ in val_loader:
                val_scores.extend([p['scores'] for p in model.predict(texts, nli_raws)])

        y_true = np.array([[int(l in d['labels']) for l in LABEL_NAMES] for d in val_data])
        _, y_pred = build_matrices(val_data, val_scores, threshold=0.5)
        macro_f1  = _f1(y_true, y_pred, average='macro', zero_division=0)
        val_f1_hist.append(macro_f1)
        for label in model.specialist_labels:
            alpha_history[label].append(model.alpha(label).item())

        print(f'Epoch {epoch:02d}/{num_epochs}  '
              f'loss={train_losses[-1]:.4f}  val_f1={macro_f1:.4f}  '
              + '  '.join(f'a_{l[:3]}={alpha_history[l][-1]:.3f}' for l in model.specialist_labels))

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}  (patience={patience})')
                break

    if best_state:
        model.load_state_dict(best_state)
        print(f'Restored best model  (val Macro F1 = {best_f1:.4f})')

    return {
        'best_macro_f1' : best_f1,
        'train_losses'  : train_losses,
        'val_f1_hist'   : val_f1_hist,
        'alpha_history' : alpha_history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = _load_yaml(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # merge config + CLI overrides
    data_path    = args.data         or cfg['data']['synthetic_data_path']
    num_epochs   = args.epochs       or cfg['training']['num_epochs']
    batch_size   = args.batch_size   or cfg['training']['batch_size']
    lr           = args.lr           or cfg['training']['learning_rate']
    kl_weight    = args.kl_weight    or cfg['training']['kl_weight']
    blend_weight = args.blend_weight or cfg['training']['blend_weight']
    patience     = args.patience     or cfg['training']['patience']
    ckpt_dir     = args.ckpt_dir     or cfg['checkpoint']['dir']
    ckpt_path    = os.path.join(ckpt_dir, cfg['checkpoint']['filename'])

    os.makedirs('data',     exist_ok=True)
    os.makedirs('outputs',  exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    # -- data ----------------------------------------------------------------
    train_data, val_data = load_data(
        data_path,
        train_split = cfg['data']['train_split'],
        seed        = cfg['data']['random_seed'],
    )

    # -- backbone ------------------------------------------------------------
    print(f"Loading {cfg['model']['name']} ...")
    nli_classifier = pipeline(
        'zero-shot-classification',
        model  = cfg['model']['name'],
        device = 0 if device == 'cuda' else -1,
    )

    # -- NLI inference -------------------------------------------------------
    print('Running NLI inference on train set ...')
    train_nli = run_nli_inference(train_data, nli_classifier)
    print('Running NLI inference on val set ...')
    val_nli   = run_nli_inference(val_data,   nli_classifier)

    # NLI baseline
    y_true_val, y_pred_nli = build_matrices(val_data, val_nli, threshold=0.5)
    print_metrics(y_true_val, y_pred_nli, 'NLI Baseline (no training)')

    # -- model ---------------------------------------------------------------
    print('Building MultiHeadHybridV2 ...')
    model = MultiHeadHybridV2(
        nli_pipeline      = nli_classifier,
        specialist_labels = SPECIALIST_LABELS,
        num_codes         = cfg['model']['num_codes'],
        proj_dim          = cfg['model']['proj_dim'],
    ).to(device)
    model.setup_label_embeddings()

    # -- train ---------------------------------------------------------------
    history = train_multihead_v2(
        model, train_data, train_nli, val_data, val_nli,
        num_epochs   = num_epochs,
        batch_size   = batch_size,
        lr           = lr,
        kl_weight    = kl_weight,
        blend_weight = blend_weight,
        patience     = patience,
    )

    # -- val-tune thresholds -------------------------------------------------
    print('\nOptimising per-label thresholds on val set ...')
    model.eval()
    final_preds = []
    for i in tqdm(range(0, len(val_data), batch_size), desc='Final inference'):
        with torch.no_grad():
            final_preds.extend(model.predict(
                [d['text'] for d in val_data[i:i+batch_size]],
                val_nli[i:i+batch_size],
            ))
    val_score_dicts = [p['scores'] for p in final_preds]
    val_thresholds  = find_best_thresholds(val_data, val_score_dicts)

    _, y_pred_tuned = build_matrices_per_thresh(val_data, val_score_dicts, val_thresholds)
    y_true_val = np.array([[int(l in d['labels']) for l in LABEL_NAMES] for d in val_data])
    final_metrics = print_metrics(y_true_val, y_pred_tuned, 'v2 MultiHead (val-tuned thresholds)')

    # -- save ----------------------------------------------------------------
    save_checkpoint(
        model, val_thresholds, ckpt_path,
        extra_meta={
            'best_macro_f1'     : history['best_macro_f1'],
            'val_tuned_macro_f1': final_metrics['macro_f1'],
        },
    )
    thresh_json = os.path.join(ckpt_dir, 'val_thresholds.json')
    with open(thresh_json, 'w') as f:
        json.dump(val_thresholds, f, indent=2)
    print(f'val_thresholds.json saved -> {thresh_json}')

    # -- HF upload -----------------------------------------------------------
    if not args.no_hub:
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            push_to_hub(ckpt_dir, cfg['huggingface']['repo_id'],
                        hf_token=hf_token, private=True)
        else:
            print('\nSet HF_TOKEN env var to auto-upload, or rerun with --no_hub to skip.')

    print(f'\nDone.  Val-Tuned Macro F1 = {final_metrics["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
