# DA6401 - Assignment 3: Implementing a Transformer for Machine Translation

## Overview

Implementation of the Transformer architecture from ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., 2017) for German→English Neural Machine Translation on the Multi30k dataset.

- **Dataset:** [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) — 29,000 train / 1,014 val / 1,000 test pairs
- **Task:** German → English translation
- **Evaluation:** Corpus-level BLEU score

## Links

- **W&B Report:** [DA6401 Assignment 3 — Transformer for German→English NMT](https://wandb.ai/ma25m011-ii/da6401-a3/reports/DA6401-Assignment-3-Transformer-for-German-English-NMT-Multi30k---VmlldzoxNjg2ODk5MQ?accessToken=7rf80ox6f2lpmo8sbqvv7a33tcovue0nsyqcwfbamiqwk6ptskf0sser7dg29e68)
- **W&B Project:** [da6401-a3](https://wandb.ai/ma25m011-ii/da6401-a3)
- **Best Checkpoint (Google Drive):** [checkpoint_epoch19.pt](https://drive.google.com/file/d/1nxN-ZgwXUos5aU_v6172baqMHwO39Vvf/view?usp=drive_link)

## Project Structure

```
da6401_assignment_3/
├── model.py           # Full Transformer architecture
│                      #   scaled_dot_product_attention, make_src_mask, make_tgt_mask
│                      #   MultiHeadAttention, PositionalEncoding
│                      #   PositionwiseFeedForward, EncoderLayer, DecoderLayer
│                      #   Encoder, Decoder, Transformer
├── lr_scheduler.py    # NoamScheduler (warmup + inverse sqrt decay)
├── dataset.py         # Multi30kDataset — spaCy tokenization, vocab, collate_fn
├── train.py           # LabelSmoothingLoss, run_epoch, greedy_decode,
│                      #   evaluate_bleu, save_checkpoint, load_checkpoint,
│                      #   run_training_experiment
├── experiments.py     # 5 W&B ablation experiments
├── plots/             # Saved plots (attention heatmaps etc.)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
wandb login
```

## Training

```bash
python train.py
```

Trains for 20 epochs with the base config (d_model=512, N=6, num_heads=8, d_ff=2048). Logs to W&B project `da6401-a3`. Checkpoints saved to `checkpoints/`.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| N (layers) | 6 |
| num_heads | 8 |
| d_ff | 2048 |
| dropout | 0.1 |
| warmup_steps | 4000 |
| batch_size | 128 |
| label smoothing ε | 0.1 |
| Adam β1, β2, ε | 0.9, 0.98, 1e-9 |

## Results

| Metric | Value |
|--------|-------|
| Test BLEU (20 epochs) | 22.16 |

## W&B Ablation Experiments

Run all 5 experiments:
```bash
python experiments.py
```

Run a single experiment:
```bash
python experiments.py --exp 3
```

| # | Experiment | W&B Group | Key finding |
|---|-----------|-----------|-------------|
| 1 | Noam scheduler vs fixed LR | `exp1-scheduler` | Noam trains more stably; fixed LR starts lower due to no warmup but both converge similarly at 8 epochs |
| 2 | With vs without √(1/d_k) scaling | `exp2-scaling` | Logged Q/K gradient norms for first 1000 steps — scaling prevents exploding gradients |
| 3 | Attention rollout heatmap | `exp3-attention` | Per-head attention weights from last encoder layer logged as W&B image |
| 4 | Sinusoidal PE vs learned PE | `exp4-pe` | Sinusoidal: **19.91 BLEU** vs Learned: **17.70 BLEU** |
| 5 | Label smoothing ε=0.1 vs ε=0.0 | `exp5-smoothing` | ε=0.0 reaches higher prediction confidence (~0.44) vs ε=0.1 (~0.40) — smoothing acts as regularizer |

## Implementation

- `MultiHeadAttention` is implemented from scratch — `torch.nn.MultiheadAttention` is **not used**
- Positional encoding uses `register_buffer` (not `nn.Parameter`) so it is not trained
- Mask convention: `True` = masked out (set to `-inf` before softmax)
- Post-LayerNorm used: `norm(x + sublayer(x))` as in the original paper
- Embeddings scaled by √d_model before adding positional encoding (§3.4 of the paper)
- `NoamScheduler` uses `step = last_epoch + 1` to avoid division by zero at step 0
- `run_epoch` uses teacher forcing: decoder input = `tgt[:, :-1]`, labels = `tgt[:, 1:]`
- `greedy_decode` and `Transformer.infer` implement inline greedy decoding (no circular imports)

