"""
dataset.py — Multi30k German→English Dataset
DA6401 Assignment 3
"""

from collections import Counter
from datasets import load_dataset
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Multi30kDataset(Dataset):
    """
    German→English translation dataset built from bentrevett/multi30k.

    Args:
        split      : 'train', 'validation', or 'test'
        src_vocab  : Optional pre-built source vocab dict (token→idx). If None, built from training split.
        tgt_vocab  : Optional pre-built target vocab dict (token→idx). If None, built from training split.
    """

    SPECIAL = ['<unk>', '<pad>', '<sos>', '<eos>']
    UNK, PAD, SOS, EOS = 0, 1, 2, 3

    def __init__(self, split='train', src_vocab=None, tgt_vocab=None):
        self.split = split
        self.nlp_de = spacy.load("de_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm")

        raw = load_dataset("bentrevett/multi30k")
        self.raw_data = raw[split]

        if src_vocab is None or tgt_vocab is None:
            train_data = raw['train']
            self.src_vocab, self.tgt_vocab = self._build_vocab(train_data)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self._process_data(self.raw_data)

    # ── Vocab ──────────────────────────────────────────────────────────

    def _tokenize_de(self, text):
        return [tok.text.lower() for tok in self.nlp_de(text)]

    def _tokenize_en(self, text):
        return [tok.text.lower() for tok in self.nlp_en(text)]

    def _build_vocab(self, train_data):
        src_counter = Counter()
        tgt_counter = Counter()
        de_texts = [ex['de'] for ex in train_data]
        en_texts = [ex['en'] for ex in train_data]
        print("Building vocab...")
        for doc in tqdm(self.nlp_de.pipe(de_texts, batch_size=256), total=len(de_texts), desc="Tokenizing DE"):
            src_counter.update(tok.text.lower() for tok in doc)
        for doc in tqdm(self.nlp_en.pipe(en_texts, batch_size=256), total=len(en_texts), desc="Tokenizing EN"):
            tgt_counter.update(tok.text.lower() for tok in doc)

        src_vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL)}
        for tok, _ in src_counter.most_common():
            if tok not in src_vocab:
                src_vocab[tok] = len(src_vocab)

        tgt_vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL)}
        for tok, _ in tgt_counter.most_common():
            if tok not in tgt_vocab:
                tgt_vocab[tok] = len(tgt_vocab)

        return src_vocab, tgt_vocab

    # ── Data processing ────────────────────────────────────────────────

    def _encode(self, tokens, vocab):
        return [self.SOS] + [vocab.get(t, self.UNK) for t in tokens] + [self.EOS]

    def _process_data(self, raw_data):
        de_texts = [ex['de'] for ex in raw_data]
        en_texts = [ex['en'] for ex in raw_data]
        processed = []
        de_docs = list(tqdm(self.nlp_de.pipe(de_texts, batch_size=256), total=len(de_texts), desc=f"Processing {self.split} DE"))
        en_docs = list(tqdm(self.nlp_en.pipe(en_texts, batch_size=256), total=len(en_texts), desc=f"Processing {self.split} EN"))
        for de_doc, en_doc in zip(de_docs, en_docs):
            src_ids = self._encode([t.text.lower() for t in de_doc], self.src_vocab)
            tgt_ids = self._encode([t.text.lower() for t in en_doc], self.tgt_vocab)
            processed.append((src_ids, tgt_ids))
        return processed

    # ── Dataset interface ──────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=1):
    """Pad a batch of (src, tgt) tensor pairs to equal length within each side."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
