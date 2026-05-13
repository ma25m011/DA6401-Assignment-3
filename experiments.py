"""
experiments.py — W&B Ablation Experiments for DA6401 Assignment 3

Five ablations required for the report:
    1. Noam scheduler vs fixed LR
    2. With vs without √(1/d_k) scaling in attention
    3. Attention rollout heatmap (last encoder layer, per head)
    4. Sinusoidal PE vs learned PE
    5. Label smoothing ε=0.1 vs ε=0.0

Run all:
    python experiments.py

Run one:
    python experiments.py --exp 3
"""

import argparse
import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

import model as model_module
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu, load_checkpoint
from lr_scheduler import NoamScheduler

CACHE_PATH = "dataset_cache.pkl"
PLOTS_DIR  = "plots"
PAD_IDX = 1
os.makedirs(PLOTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Dataset cache (skip spaCy tokenization on subsequent runs)
# ══════════════════════════════════════════════════════════════════════

class _CachedDataset(Dataset):
    PAD, SOS, EOS, UNK = 1, 2, 3, 0

    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def get_datasets():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        train_ds = _CachedDataset(cache['train_data'], cache['src_vocab'], cache['tgt_vocab'])
        val_ds = _CachedDataset(cache['val_data'], cache['src_vocab'], cache['tgt_vocab'])
        test_ds = _CachedDataset(cache['test_data'], cache['src_vocab'], cache['tgt_vocab'])
        return train_ds, val_ds, test_ds

    print("Building dataset for the first time (this takes ~3 min)...")
    from dataset import Multi30kDataset
    train_ds = Multi30kDataset(split='train')
    val_ds = Multi30kDataset(split='validation', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds = Multi30kDataset(split='test', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    cache = {
        'src_vocab': train_ds.src_vocab,
        'tgt_vocab': train_ds.tgt_vocab,
        'train_data': train_ds.data,
        'val_data': val_ds.data,
        'test_data': test_ds.data,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved dataset cache to {CACHE_PATH}")
    return train_ds, val_ds, test_ds


def collate_fn(batch, pad_idx=1):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    'd_model': 512, 'N': 6, 'num_heads': 8, 'd_ff': 2048, 'dropout': 0.1,
    'warmup_steps': 4000, 'batch_size': 128, 'num_epochs': 8,
    'smoothing': 0.1, 'lr': 1.0,
}


def build_model(cfg, src_vocab_size, tgt_vocab_size, device):
    return Transformer(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=cfg['d_model'], N=cfg['N'], num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'], dropout=cfg['dropout'],
    ).to(device)


def make_loaders(train_ds, val_ds, test_ds, batch_size):
    coll = partial(collate_fn, pad_idx=PAD_IDX)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll),
        DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=coll),
    )


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Noam scheduler vs fixed LR
# ══════════════════════════════════════════════════════════════════════

def exp1_noam_vs_fixed(train_ds, val_ds, device):
    cfg = DEFAULT_CFG.copy()
    cfg['num_epochs'] = 8
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, val_ds, cfg['batch_size'])
    src_v, tgt_v = len(train_ds.src_vocab), len(train_ds.tgt_vocab)

    for use_noam in [True, False]:
        run_name = "noam" if use_noam else "fixed_lr"
        wandb.init(project="da6401-a3", group="exp1-scheduler", name=run_name,
                   config={**cfg, 'scheduler': run_name}, reinit=True)

        model = build_model(cfg, src_v, tgt_v, device)
        loss_fn = LabelSmoothingLoss(tgt_v, pad_idx=PAD_IDX, smoothing=cfg['smoothing'])

        if use_noam:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                          betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'],
                                       warmup_steps=cfg['warmup_steps'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                          betas=(0.9, 0.98), eps=1e-9)
            scheduler = None

        for epoch in range(cfg['num_epochs']):
            train_loss = run_epoch(train_loader, model, loss_fn, optimizer,
                                    scheduler, epoch, True, device)
            val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                                  epoch, False, device)
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                       'lr': optimizer.param_groups[0]['lr']})
            print(f"  [{run_name}] Epoch {epoch:02d}  train={train_loss:.2f}  val={val_loss:.2f}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: With vs without √(1/d_k) scaling
# ══════════════════════════════════════════════════════════════════════

def attn_no_scale(Q, K, V, mask=None):
    """scaled_dot_product_attention WITHOUT the √(1/d_k) division."""
    scores = torch.matmul(Q, K.transpose(-2, -1))   # NO division
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


def exp2_attn_scaling(train_ds, val_ds, device):
    cfg = DEFAULT_CFG.copy()
    cfg['batch_size'] = 64
    max_steps = 1000
    train_loader, _, _ = make_loaders(train_ds, val_ds, val_ds, cfg['batch_size'])
    src_v, tgt_v = len(train_ds.src_vocab), len(train_ds.tgt_vocab)

    original_attn = model_module.scaled_dot_product_attention
    try:
        for with_scaling in [True, False]:
            run_name = "with_scaling" if with_scaling else "no_scaling"
            wandb.init(project="da6401-a3", group="exp2-scaling", name=run_name,
                       config={**cfg, 'scaling': with_scaling}, reinit=True)

            model_module.scaled_dot_product_attention = (
                original_attn if with_scaling else attn_no_scale
            )

            model = build_model(cfg, src_v, tgt_v, device)
            loss_fn = LabelSmoothingLoss(tgt_v, pad_idx=PAD_IDX, smoothing=cfg['smoothing'])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                          betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'],
                                       warmup_steps=cfg['warmup_steps'])

            model.train()
            step = 0
            done = False
            while not done:
                for src, tgt in train_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_in, tgt_lbl = tgt[:, :-1], tgt[:, 1:]
                    src_mask = make_src_mask(src)
                    tgt_mask = make_tgt_mask(tgt_in)

                    logits = model(src, tgt_in, src_mask, tgt_mask)
                    B, T, V = logits.shape
                    loss = loss_fn(logits.reshape(B * T, V), tgt_lbl.reshape(B * T))

                    optimizer.zero_grad()
                    loss.backward()

                    g_q = model.encoder.layers[0].self_attn.W_Q.weight.grad
                    g_k = model.encoder.layers[0].self_attn.W_K.weight.grad
                    gn_q = g_q.norm().item() if g_q is not None else 0.0
                    gn_k = g_k.norm().item() if g_k is not None else 0.0

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    wandb.log({'step': step, 'loss': loss.item(),
                               'grad_norm_Q': gn_q, 'grad_norm_K': gn_k})
                    step += 1
                    if step >= max_steps:
                        done = True
                        break

            print(f"  [{run_name}] Done after {step} steps")
            wandb.finish()
    finally:
        model_module.scaled_dot_product_attention = original_attn


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Attention rollout heatmap (uses existing checkpoint)
# ══════════════════════════════════════════════════════════════════════

def exp3_attention_heatmap(train_ds, val_ds, test_ds, device):
    cfg = DEFAULT_CFG.copy()
    src_v, tgt_v = len(train_ds.src_vocab), len(train_ds.tgt_vocab)

    ckpt_dir = "checkpoints"
    if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
        print(f"[ERROR] No checkpoint found in {ckpt_dir}/. Run train.py first.")
        return
    ckpts = sorted(os.listdir(ckpt_dir))
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading {ckpt_path}...")

    model = build_model(cfg, src_v, tgt_v, device)
    load_checkpoint(ckpt_path, model)
    model.eval()

    wandb.init(project="da6401-a3", group="exp3-attention",
               name="attention_heatmap", reinit=True)

    src_ids, tgt_ids = test_ds[0]
    src = src_ids.unsqueeze(0).to(device)
    src_mask = make_src_mask(src)

    idx_to_src = {v: k for k, v in train_ds.src_vocab.items()}
    src_tokens = [idx_to_src.get(i.item(), '?') for i in src[0]]

    captured = []
    original_attn = model_module.scaled_dot_product_attention
    def capture(Q, K, V, mask=None):
        out, w = original_attn(Q, K, V, mask)
        captured.append(w.detach().cpu())
        return out, w

    model_module.scaled_dot_product_attention = capture
    try:
        with torch.no_grad():
            _ = model.encode(src, src_mask)
    finally:
        model_module.scaled_dot_product_attention = original_attn

    last = captured[-1][0]   # [num_heads, src_len, src_len]
    num_heads = last.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for h in range(num_heads):
        ax = axes[h // 4, h % 4]
        im = ax.imshow(last[h].numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f"Head {h+1}")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(src_tokens, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Last encoder layer attention — '{' '.join(src_tokens[1:-1])}'")
    plt.tight_layout()

    wandb.log({"attention_heatmap": wandb.Image(fig),
               "src_sentence": ' '.join(src_tokens[1:-1])})
    out_path = os.path.join(PLOTS_DIR, "attention_heatmap.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  [OK] Logged attention heatmap. Saved as {out_path}")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Sinusoidal vs learned PE
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """nn.Embedding-based positional encoding (drop-in for PositionalEncoding)."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embed(positions))


def exp4_pe_comparison(train_ds, val_ds, test_ds, device):
    cfg = DEFAULT_CFG.copy()
    cfg['num_epochs'] = 10
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg['batch_size'])
    src_v, tgt_v = len(train_ds.src_vocab), len(train_ds.tgt_vocab)

    for use_learned in [False, True]:
        run_name = "learned_pe" if use_learned else "sinusoidal_pe"
        wandb.init(project="da6401-a3", group="exp4-pe", name=run_name,
                   config={**cfg, 'pe': run_name}, reinit=True)

        model = build_model(cfg, src_v, tgt_v, device)
        if use_learned:
            model.pos_enc = LearnedPositionalEncoding(cfg['d_model'], cfg['dropout']).to(device)

        loss_fn = LabelSmoothingLoss(tgt_v, pad_idx=PAD_IDX, smoothing=cfg['smoothing'])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                      betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'],
                                   warmup_steps=cfg['warmup_steps'])

        for epoch in range(cfg['num_epochs']):
            train_loss = run_epoch(train_loader, model, loss_fn, optimizer,
                                    scheduler, epoch, True, device)
            val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                                  epoch, False, device)
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            print(f"  [{run_name}] Epoch {epoch:02d}  train={train_loss:.2f}  val={val_loss:.2f}")

        bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
        wandb.log({'test_bleu': bleu})
        print(f"  [{run_name}] Test BLEU: {bleu:.2f}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Label smoothing ε=0.1 vs ε=0.0
# ══════════════════════════════════════════════════════════════════════

def mean_correct_token_prob(model, val_loader, device, pad_idx=1):
    model.eval()
    total_prob = 0.0
    total_count = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_lbl = tgt[:, :-1], tgt[:, 1:]
            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            probs = torch.softmax(logits, dim=-1)
            correct = probs.gather(2, tgt_lbl.unsqueeze(-1)).squeeze(-1)
            mask = tgt_lbl != pad_idx
            total_prob += (correct * mask).sum().item()
            total_count += mask.sum().item()
    return total_prob / total_count if total_count > 0 else 0.0


def exp5_label_smoothing(train_ds, val_ds, device):
    cfg = DEFAULT_CFG.copy()
    cfg['num_epochs'] = 10
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, val_ds, cfg['batch_size'])
    src_v, tgt_v = len(train_ds.src_vocab), len(train_ds.tgt_vocab)

    for smoothing in [0.1, 0.0]:
        run_name = f"smooth_{smoothing}"
        wandb.init(project="da6401-a3", group="exp5-smoothing", name=run_name,
                   config={**cfg, 'smoothing': smoothing}, reinit=True)

        model = build_model(cfg, src_v, tgt_v, device)
        loss_fn = LabelSmoothingLoss(tgt_v, pad_idx=PAD_IDX, smoothing=smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                      betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'],
                                   warmup_steps=cfg['warmup_steps'])

        for epoch in range(cfg['num_epochs']):
            train_loss = run_epoch(train_loader, model, loss_fn, optimizer,
                                    scheduler, epoch, True, device)
            val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                                  epoch, False, device)
            mean_p = mean_correct_token_prob(model, val_loader, device)
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                       'mean_correct_prob': mean_p})
            print(f"  [{run_name}] Epoch {epoch:02d}  train={train_loss:.2f}  "
                  f"val={val_loss:.2f}  prob={mean_p:.4f}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=None,
                        help='Run only experiment N (1-5). Default: run all.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = get_datasets()
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Src vocab: {len(train_ds.src_vocab)}, Tgt vocab: {len(train_ds.tgt_vocab)}")

    experiments = {
        1: lambda: exp1_noam_vs_fixed(train_ds, val_ds, device),
        2: lambda: exp2_attn_scaling(train_ds, val_ds, device),
        3: lambda: exp3_attention_heatmap(train_ds, val_ds, test_ds, device),
        4: lambda: exp4_pe_comparison(train_ds, val_ds, test_ds, device),
        5: lambda: exp5_label_smoothing(train_ds, val_ds, device),
    }

    if args.exp:
        print(f"\n=== Experiment {args.exp} ===")
        experiments[args.exp]()
    else:
        for n, fn in experiments.items():
            print(f"\n=== Experiment {n} ===")
            fn()


if __name__ == "__main__":
    main()
