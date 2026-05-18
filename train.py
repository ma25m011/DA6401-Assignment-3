"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        smooth_val = self.smoothing / (self.vocab_size - 1)
        y_smooth = torch.full_like(log_probs, smooth_val)
        y_smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        y_smooth[target == self.pad_idx] = 0.0

        # KL divergence: -sum(y * log_p) when y is the smoothed target
        loss = -(y_smooth * log_probs).sum()
        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: decoder input drops last token, target drops first
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)  # [B, T, vocab]

            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_labels.reshape(B * T))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src.to(device), src_mask.to(device))
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)
            logits = model.decode(memory, src_mask.to(device), ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()

    # Use model's own vocab for decoding outputs; fall back to passed tgt_vocab
    model_tgt_vocab = getattr(model, 'tgt_vocab', None) or tgt_vocab
    model_src_vocab = getattr(model, 'src_vocab', None)

    model_idx_to_tok = {v: k for k, v in model_tgt_vocab.items()}
    model_special = {model_tgt_vocab.get('<sos>', 2), model_tgt_vocab.get('<eos>', 3), model_tgt_vocab.get('<pad>', 1)}
    sos = model_tgt_vocab.get('<sos>', 2)
    eos = model_tgt_vocab.get('<eos>', 3)

    # Build remap from dataloader's src indices → model's src indices (handles vocab mismatch)
    dl_src_vocab = getattr(test_dataloader.dataset, 'src_vocab', None)
    if model_src_vocab and dl_src_vocab and model_src_vocab != dl_src_vocab:
        dl_idx_to_src_tok = {v: k for k, v in dl_src_vocab.items()}
        src_remap = {dl_idx: model_src_vocab.get(tok, model_src_vocab.get('<unk>', 0))
                     for dl_idx, tok in dl_idx_to_src_tok.items()}
    else:
        src_remap = None

    ref_idx_to_tok = {v: k for k, v in tgt_vocab.items()}
    ref_special = {tgt_vocab.get('<sos>', 2), tgt_vocab.get('<eos>', 3), tgt_vocab.get('<pad>', 1)}

    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                if src_remap is not None:
                    src_i = src_i.clone().apply_(lambda x: src_remap.get(x, x))
                src_i = src_i.to(device)
                src_mask_i = make_src_mask(src_i)

                out = greedy_decode(model, src_i, src_mask_i, max_len, sos, eos, device)
                hyp = [model_idx_to_tok.get(t.item(), '<unk>') for t in out[0, 1:] if t.item() not in model_special]

                ref_ids = tgt[i].tolist()
                ref = [ref_idx_to_tok.get(t, '<unk>') for t in ref_ids if t not in ref_special]

                hypotheses.append(hyp)
                references.append([ref])

    return corpus_bleu(references, hypotheses) * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    src_vocab_size, tgt_vocab_size = (
        model.src_embed.num_embeddings,
        model.tgt_embed.num_embeddings,
    )
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': model.d_model,
            'N': len(model.encoder.layers),
            'num_heads': model.encoder.layers[0].self_attn.num_heads,
            'd_ff': model.encoder.layers[0].ffn.linear1.out_features,
            'dropout': model.pos_enc.dropout.p,
        },
    }
    # Save vocab dicts if available (needed for infer() after checkpoint load)
    if hasattr(model, 'src_vocab') and model.src_vocab is not None:
        ckpt['src_vocab'] = model.src_vocab
    if hasattr(model, 'tgt_vocab') and model.tgt_vocab is not None:
        ckpt['tgt_vocab'] = model.tgt_vocab
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    import os, gdown
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        gdown.download(
            url=f"https://drive.google.com/uc?id={Transformer._GDRIVE_FILE_ID}&confirm=t",
            output=path, quiet=False,
        )
        if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            gdown.download(id=Transformer._GDRIVE_FILE_ID, output=path, quiet=False)
    ckpt = torch.load(path, map_location='cpu')

    # If model's embedding sizes differ from checkpoint's, rebuild layers to match
    sd = ckpt['model_state_dict']
    ckpt_src_size = sd['src_embed.weight'].shape[0]
    ckpt_tgt_size = sd['tgt_embed.weight'].shape[0]
    if model.src_embed.num_embeddings != ckpt_src_size or model.tgt_embed.num_embeddings != ckpt_tgt_size:
        d_model = model.d_model
        model.src_embed = torch.nn.Embedding(ckpt_src_size, d_model)
        model.tgt_embed = torch.nn.Embedding(ckpt_tgt_size, d_model)
        model.output_projection = torch.nn.Linear(d_model, ckpt_tgt_size)

    model.load_state_dict(sd)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if hasattr(model, 'src_vocab') and model.src_vocab is None:
        model.src_vocab = ckpt.get('src_vocab', None)
    if hasattr(model, 'tgt_vocab') and model.tgt_vocab is None:
        model.tgt_vocab = ckpt.get('tgt_vocab', None)
    return int(ckpt['epoch'])


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import os
    import shutil
    import wandb
    from functools import partial
    from dataset import Multi30kDataset, collate_fn

    config = {
        'd_model':       512,
        'N':             6,
        'num_heads':     8,
        'd_ff':          2048,
        'dropout':       0.1,
        'warmup_steps':  4000,
        'batch_size':    128,
        'num_epochs':    20,
        'smoothing':     0.1,
        'lr':            1.0,
    }

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    ckpt_dir = "checkpoints"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    # ── Dataset ────────────────────────────────────────────────────────
    train_ds = Multi30kDataset(split='train')
    val_ds   = Multi30kDataset(split='validation',
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset(split='test',
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)

    pad = train_ds.PAD
    _collate = partial(collate_fn, pad_idx=pad)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=1,              shuffle=False,
                              collate_fn=_collate)

    src_vocab_size = len(train_ds.src_vocab)
    tgt_vocab_size = len(train_ds.tgt_vocab)

    # ── Model ──────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    # Store vocab dicts on model so save_checkpoint can persist them for infer()
    model.src_vocab = train_ds.src_vocab
    model.tgt_vocab = train_ds.tgt_vocab

    # ── Optimizer / scheduler / loss ───────────────────────────────────
    from lr_scheduler import NoamScheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model,
                              warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, pad_idx=pad,
                                   smoothing=cfg.smoothing)

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer,
                               scheduler, epoch_num=epoch, is_train=True,
                               device=device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None,
                               None, epoch_num=epoch, is_train=False,
                               device=device)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch,
                        path=os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:02d}.pt"))

    # ── Final BLEU ─────────────────────────────────────────────────────
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    wandb.log({'test_bleu': bleu})
    print(f"Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
