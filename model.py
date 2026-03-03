"""
model.py  --  HybridNLI-PolyEncoder v2
=======================================
Architecture
------------
One frozen DeBERTa-v3-large forward pass feeds two branches:

  NLI branch      : zero-shot scores for all 5 labels
  PolyEncoder heads (4): Economy, Technology, Finance, Environment
  Health          : NLI score used directly (0.87 zero-shot baseline)

Blended score per specialist label:
  score = sigmoid(alpha_raw) * nli_score + (1-sigmoid(alpha_raw)) * poly_prob

alpha_raw is trained via a *direct* blend-loss gradient (v2 fix -- in v1 alpha
received ~0 gradient and stayed at 0.5 forever).

Benchmark (val set, val-tuned thresholds):
  Macro F1 = 0.9054   (+0.2366 over NLI-only baseline of 0.6688)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, HfApi, login

LABEL_NAMES       = ['Technology', 'Health', 'Finance', 'Economy', 'Environment']
SPECIALIST_LABELS = ['Economy', 'Technology', 'Finance', 'Environment']

LABEL_DESCRIPTIONS = {
    'Technology'  : 'This text is about technology, software, hardware, artificial intelligence, or digital innovation',
    'Health'      : 'This text is about medicine, healthcare, wellness, disease, medical research, or public health',
    'Finance'     : 'This text is about banking, investment, stock markets, corporate earnings, or financial instruments',
    'Economy'     : 'This text is about economic policy, GDP, employment, trade, inflation, or macroeconomic trends',
    'Environment' : 'This text is about climate change, pollution, ecology, sustainability, conservation, or nature',
}


# ---------------------------------------------------------------------------
# PolyEncoderHead
# ---------------------------------------------------------------------------

class PolyEncoderHead(nn.Module):
    """
    Lightweight PolyEncoder specialist head (~534K params).

    K learnable code vectors attend over DeBERTa token embeddings.
    Poly-attention aggregates them conditioned on the label embedding.
    Temperature-scaled dot-product gives the final scalar score.

    Args:
        hidden_dim : DeBERTa hidden size (1024 for deberta-v3-large)
        num_codes  : number of learnable code vectors  (default 8)
        proj_dim   : projection / comparison dimension (default 256)
    """

    def __init__(self, hidden_dim: int = 1024, num_codes: int = 8, proj_dim: int = 256):
        super().__init__()
        self.num_codes = num_codes
        self.proj_dim  = proj_dim

        self.codes = nn.Embedding(num_codes, hidden_dim)
        nn.init.normal_(self.codes.weight, std=0.02)

        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.label_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def encode_context(self, token_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate token embeddings into K poly vectors via code attention.

        Args:
            token_embs     : (B, T, hidden_dim)
            attention_mask : (B, T)
        Returns:
            poly_vecs : (B, K, proj_dim)  L2-normalised
        """
        token_embs = token_embs.float()
        B, T, D    = token_embs.shape
        codes      = self.codes.weight.unsqueeze(0).expand(B, -1, -1)      # (B, K, D)
        logits     = torch.bmm(codes, token_embs.transpose(1, 2))          # (B, K, T)
        pad_mask   = (attention_mask == 0).unsqueeze(1).expand(-1, self.num_codes, -1)
        logits     = logits.masked_fill(pad_mask, -1e4)
        attn_w     = torch.softmax(logits, dim=-1)
        poly_vecs  = torch.bmm(attn_w, token_embs)                         # (B, K, D)
        poly_vecs  = self.ctx_proj(poly_vecs)                               # (B, K, proj)
        return F.normalize(poly_vecs, dim=-1)

    def encode_label(self, label_emb: torch.Tensor) -> torch.Tensor:
        """
        Project and L2-normalise a label embedding.

        Args:
            label_emb : (hidden_dim,) or (1, hidden_dim)
        Returns:
            (proj_dim,)
        """
        if label_emb.dim() == 1:
            label_emb = label_emb.unsqueeze(0)
        return F.normalize(self.label_proj(label_emb).squeeze(0), dim=-1)

    def poly_score(self, poly_vecs: torch.Tensor, label_vec: torch.Tensor) -> torch.Tensor:
        """
        Poly-attention aggregation + temperature-scaled dot-product.

        Args:
            poly_vecs : (K, proj_dim)
            label_vec : (proj_dim,)
        Returns:
            scalar logit
        """
        code_attn = torch.softmax(poly_vecs @ label_vec, dim=0)            # (K,)
        ctx       = (code_attn.unsqueeze(1) * poly_vecs).sum(0)           # (proj_dim,)
        return (ctx @ label_vec) * self.log_temp.exp()

    def forward(self,
                token_embs:     torch.Tensor,
                attention_mask: torch.Tensor,
                label_emb:      torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embs     : (B, T, hidden_dim)
            attention_mask : (B, T)
            label_emb      : (hidden_dim,) pre-computed label embedding
        Returns:
            logits : (B,)
        """
        poly_vecs = self.encode_context(token_embs, attention_mask)
        label_vec = self.encode_label(label_emb)
        return torch.stack([
            self.poly_score(poly_vecs[i], label_vec)
            for i in range(poly_vecs.size(0))
        ])


# ---------------------------------------------------------------------------
# MultiHeadHybridV2
# ---------------------------------------------------------------------------

class MultiHeadHybridV2(nn.Module):
    """
    HybridNLI-PolyEncoder v2 -- zero-shot multi-label classifier.

    One frozen DeBERTa-v3-large forward pass feeds both the NLI branch
    and 4 PolyEncoder specialist heads.  Outputs are blended via learned
    per-label alpha weights trained with a dual-loss (v2 fix).

    Trainable parameters: ~2.1M / 435M total (0.49%)

    Args:
        nli_pipeline      : loaded transformers zero-shot-classification pipeline
        specialist_labels : list of label strings that receive a PolyHead
        num_codes         : PolyEncoder code vectors per head (default 8)
        proj_dim          : projection dimension               (default 256)
    """

    def __init__(self, nli_pipeline, specialist_labels, num_codes=8, proj_dim=256):
        super().__init__()
        self.backbone          = nli_pipeline.model.deberta
        self.tokenizer         = nli_pipeline.tokenizer
        self.hidden_dim        = self.backbone.config.hidden_size
        self.specialist_labels = specialist_labels

        # Freeze backbone -- no gradient flows through 435M params
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.poly_heads = nn.ModuleDict({
            l: PolyEncoderHead(self.hidden_dim, num_codes, proj_dim)
            for l in specialist_labels
        })

        # init alpha_raw = 0 --> sigmoid(0) = 0.5 (equal blend)
        # moves during training via blend-loss gradient (v2 fix)
        self.alpha_raws = nn.ParameterDict({
            l: nn.Parameter(torch.tensor(0.0)) for l in specialist_labels
        })

        self._label_embs: dict = {}

    # -- properties ----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.poly_heads.parameters()).device

    def alpha(self, label: str) -> torch.Tensor:
        """Return sigmoid(alpha_raw) in (0,1) for a given label."""
        return torch.sigmoid(self.alpha_raws[label])

    # -- setup ---------------------------------------------------------------

    def setup_label_embeddings(self) -> None:
        """
        Pre-compute mean-pooled label embeddings from the frozen DeBERTa backbone.
        Call once after moving the model to the target device.
        """
        print('Computing label embeddings from frozen DeBERTa:')
        for label in self.specialist_labels:
            enc = self.tokenizer(
                LABEL_DESCRIPTIONS[label], return_tensors='pt',
                padding=True, truncation=True, max_length=64,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out  = self.backbone(**enc)
                mask = enc['attention_mask'].unsqueeze(-1).float()
                emb  = (out.last_hidden_state.float() * mask).sum(1) / mask.sum(1)
            self._label_embs[label] = emb.squeeze(0)
            print(f'  {label:<15} {self._label_embs[label].shape}')

    # -- forward -------------------------------------------------------------

    def _get_token_embeddings(self, texts: list, max_length: int = 128) -> tuple:
        enc = self.tokenizer(
            texts, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.backbone(**enc)
        return out.last_hidden_state.float(), enc['attention_mask']

    def forward(self, texts: list, nli_score_dicts: list) -> tuple:
        """
        Args:
            texts           : list of B strings
            nli_score_dicts : list of B dicts {label: nli_score}

        Returns:
            blended     : dict {label: (B,) probability tensor}
            poly_logits : dict {label: (B,) raw logit tensor}
            nli_tensors : dict {label: (B,) NLI score tensor}
        """
        token_embs, attn_mask = self._get_token_embeddings(texts)
        blended, poly_logits, nli_tensors = {}, {}, {}

        for label in self.specialist_labels:
            nli_s = torch.tensor(
                [s.get(label, 0.0) for s in nli_score_dicts],
                dtype=torch.float32, device=self.device,
            )
            logit     = self.poly_heads[label](token_embs, attn_mask, self._label_embs[label])
            poly_prob = torch.sigmoid(logit)
            a         = self.alpha(label)

            blended[label]     = a * nli_s + (1.0 - a) * poly_prob
            poly_logits[label] = logit
            nli_tensors[label] = nli_s

        return blended, poly_logits, nli_tensors

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict(self, texts: list, nli_score_dicts: list, thresholds: dict = None) -> list:
        """
        End-to-end inference with per-label thresholds.

        Args:
            texts           : list of strings
            nli_score_dicts : list of NLI score dicts from run_nli_inference
            thresholds      : dict {label: float}, defaults to 0.5

        Returns:
            list of dicts: {text, scores, predicted_labels}
        """
        self.eval()
        if thresholds is None:
            thresholds = {l: 0.5 for l in LABEL_NAMES}

        blended, _, _ = self.forward(texts, nli_score_dicts)
        results = []
        for i, (text, nli_s) in enumerate(zip(texts, nli_score_dicts)):
            scores = dict(nli_s)
            for label in self.specialist_labels:
                scores[label] = float(blended[label][i].item())
            preds = [l for l in LABEL_NAMES if scores[l] >= thresholds.get(l, 0.5)]
            results.append({'text': text, 'scores': scores, 'predicted_labels': preds})
        return results


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(model, val_thresholds, path, extra_meta=None):
    """
    Save poly head weights, alpha values, label embeddings, and thresholds.

    Args:
        model          : trained MultiHeadHybridV2
        val_thresholds : dict {label: float}
        path           : destination .pt path
        extra_meta     : optional dict of additional keys to store
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        'poly_heads_state' : {l: model.poly_heads[l].state_dict() for l in model.specialist_labels},
        'alpha_raws'       : {l: model.alpha_raws[l].item()       for l in model.specialist_labels},
        'label_embs'       : {l: e.cpu() for l, e in model._label_embs.items()},
        'specialist_labels': model.specialist_labels,
        'val_thresholds'   : val_thresholds,
        'num_codes'        : model.poly_heads[model.specialist_labels[0]].num_codes,
        'proj_dim'         : model.poly_heads[model.specialist_labels[0]].proj_dim,
        'hidden_dim'       : model.hidden_dim,
        'backbone_name'    : 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0',
        'label_names'      : LABEL_NAMES,
        'label_descriptions': LABEL_DESCRIPTIONS,
    }
    if extra_meta:
        payload.update(extra_meta)
    torch.save(payload, path)
    print(f'Checkpoint saved -> {path}')


def load_model_from_checkpoint(ckpt_path, nli_pipeline):
    """
    Restore MultiHeadHybridV2 from a saved checkpoint.

    Args:
        ckpt_path    : path to .pt file (local or from hf_hub_download)
        nli_pipeline : loaded transformers zero-shot-classification pipeline

    Returns:
        model      : MultiHeadHybridV2 ready for inference
        thresholds : dict {label: float}

    Example::

        from transformers import pipeline
        from model import load_model_from_checkpoint

        nli = pipeline('zero-shot-classification',
                       model='MoritzLaurer/deberta-v3-large-zeroshot-v2.0', device=0)
        model, thresholds = load_model_from_checkpoint('checkpoints/multihead_v2/multihead_v2.pt', nli)
        model = model.to('cuda')
        for l, emb in model._label_embs.items():
            model._label_embs[l] = emb.to(model.device)
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = MultiHeadHybridV2(
        nli_pipeline      = nli_pipeline,
        specialist_labels = ckpt['specialist_labels'],
        num_codes         = ckpt['num_codes'],
        proj_dim          = ckpt['proj_dim'],
    )
    for l, sd in ckpt['poly_heads_state'].items():
        model.poly_heads[l].load_state_dict(sd)
    with torch.no_grad():
        for l, val in ckpt['alpha_raws'].items():
            model.alpha_raws[l].fill_(val)
    model._label_embs = ckpt['label_embs']
    thresholds = ckpt.get('val_thresholds', {l: 0.5 for l in ckpt.get('label_names', LABEL_NAMES)})
    print(f'Model loaded from {ckpt_path}')
    return model, thresholds


def push_to_hub(ckpt_dir, repo_id, hf_token=None, private=True):
    """
    Upload checkpoint files to HuggingFace Hub.

    Args:
        ckpt_dir  : local directory with multihead_v2.pt, val_thresholds.json, README.md
        repo_id   : e.g. 'your-username/hybrid-nli-polyencoder'
        hf_token  : HF write token (or set HF_TOKEN env var before calling)
        private   : create private repo (default True)
    """
    if hf_token:
        login(token=hf_token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True, private=private)
    for fname in ['multihead_v2.pt', 'val_thresholds.json', 'README.md']:
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.exists(fpath):
            api.upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                            repo_id=repo_id, repo_type='model')
            print(f'  Uploaded {fname}')
    print(f'Model live at: https://huggingface.co/{repo_id}')
