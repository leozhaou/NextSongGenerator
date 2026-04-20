"""
models/lstm.py

LSTM-based language model for song lyric completion.
Defines the model architecture, tokenizer wrapper, and dataset class.

Self-registers with models.registry under the name "lstm" so that
train.py and evaluate.py can load it without any direct imports.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LyricDataset(Dataset):
    """
    Sliding-window character- or word-level dataset built from a list of
    lyric strings.

    Args:
        lyrics      : list of raw lyric strings
        tokenizer   : LyricTokenizer instance (must already be fitted)
        seq_len     : number of tokens per input window
        stride      : step size between consecutive windows (default = 1)
    """

    def __init__(self, lyrics: list[str], tokenizer, seq_len: int = 64, stride: int = 1):
        self.seq_len = seq_len
        window = seq_len + 1  # x + y

        # Encode all lyrics into one flat array, then record (offset, length) per window.
        # This avoids storing millions of duplicate Python lists and cuts RAM usage ~10×.
        all_ids: list[int] = []
        offsets: list[int] = []

        for lyric in lyrics:
            ids = tokenizer.encode(lyric)
            if len(ids) < window:
                continue
            base = len(all_ids)
            all_ids.extend(ids)
            for start in range(0, len(ids) - seq_len, stride):
                offsets.append(base + start)

        # Store as numpy arrays for O(1) indexing and low overhead
        self._data    = np.array(all_ids, dtype=np.int32)
        self._offsets = np.array(offsets,  dtype=np.int64)
        self._window  = window

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int):
        start = self._offsets[idx]
        chunk = self._data[start : start + self._window]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class LyricTokenizer:
    """
    Simple word-level tokenizer with special tokens.

    Special tokens
    --------------
    <PAD>  index 0  – padding
    <UNK>  index 1  – unknown word
    <BOS>  index 2  – beginning of sequence
    <EOS>  index 3  – end of sequence
    <NL>   index 4  – newline / line-break

    Usage
    -----
        tok = LyricTokenizer()
        tok.fit(list_of_lyric_strings)
        ids  = tok.encode("hello world")
        text = tok.decode(ids)
    """

    PAD, UNK, BOS, EOS, NL = "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"
    SPECIALS = [PAD, UNK, BOS, EOS, NL]

    def __init__(self, max_vocab: int = 20_000):
        self.max_vocab  = max_vocab
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.fitted = False

    # ------------------------------------------------------------------
    def fit(self, lyrics: list[str]) -> "LyricTokenizer":
        from collections import Counter
        counter: Counter = Counter()
        for lyric in lyrics:
            for line in lyric.split("\n"):
                counter.update(line.lower().split())

        vocab = self.SPECIALS + [w for w, _ in counter.most_common(self.max_vocab - len(self.SPECIALS))]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert self.fitted, "Call .fit() before .encode()"
        unk = self.word2idx[self.UNK]
        tokens: list[int] = []
        if add_special_tokens:
            tokens.append(self.word2idx[self.BOS])
        for line in text.split("\n"):
            for word in line.lower().split():
                tokens.append(self.word2idx.get(word, unk))
            tokens.append(self.word2idx[self.NL])
        if add_special_tokens:
            tokens.append(self.word2idx[self.EOS])
        return tokens

    # ------------------------------------------------------------------
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        skip = set(self.word2idx[t] for t in self.SPECIALS) if skip_special else set()
        words: list[str] = []
        for idx in ids:
            if idx in skip:
                continue
            words.append(self.idx2word.get(idx, self.UNK))
        return " ".join(words)

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMLyricModel(nn.Module):
    """
    Multi-layer LSTM language model for lyric completion.

    Architecture
    ------------
        Embedding  →  Dropout  →  LSTM (num_layers)  →  Dropout  →  Linear

    Args:
        vocab_size   : size of the token vocabulary
        embed_dim    : word-embedding dimension
        hidden_dim   : LSTM hidden state size
        num_layers   : number of stacked LSTM layers
        dropout      : dropout probability (applied after embedding & between LSTM layers)
        pad_idx      : vocabulary index of <PAD> token (embedding set to zero-gradient)
        tie_weights  : if True, share embedding and output projection weights
                       (requires embed_dim == hidden_dim)
    """

    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int  = 256,
        hidden_dim : int  = 512,
        num_layers : int  = 2,
        dropout    : float = 0.3,
        pad_idx    : int  = 0,
        tie_weights: bool = False,
    ):
        super().__init__()

        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.pad_idx     = pad_idx

        # --- layers ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        self.out_drop  = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, vocab_size)

        # optional weight tying
        if tie_weights:
            if embed_dim != hidden_dim:
                raise ValueError("tie_weights requires embed_dim == hidden_dim")
            self.fc.weight = self.embedding.weight

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.pad_idx is not None:
            self.embedding.weight.data[self.pad_idx].zero_()
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # set forget-gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zero-initialised (h_0, c_0) for the given batch size."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h, c

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x      : (batch, seq_len)  token indices
            hidden : optional (h, c) tuple – pass None to auto-initialise

        Returns:
            logits : (batch, seq_len, vocab_size)
            hidden : updated (h, c) for stateful generation
        """
        batch_size = x.size(0)
        device     = x.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        emb    = self.embed_drop(self.embedding(x))          # (B, T, E)
        out, hidden = self.lstm(emb, hidden)                  # (B, T, H)
        logits = self.fc(self.out_drop(out))                  # (B, T, V)
        return logits, hidden

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids  : list[int],
        tokenizer,
        max_new_tokens: int = 100,
        temperature : float = 1.0,
        top_k       : int   = 50,
        top_p       : float = 0.95,
        device      : torch.device | None = None,
    ) -> str:
        """
        Auto-regressively generate tokens given a prompt.

        Sampling strategy: top-k + top-p (nucleus) with temperature scaling.

        Args:
            prompt_ids    : encoded prompt token ids
            tokenizer     : LyricTokenizer for decoding output
            max_new_tokens: maximum number of new tokens to generate
            temperature   : softmax temperature (lower = more conservative)
            top_k         : keep only top-k logits before sampling
            top_p         : nucleus probability mass cutoff
            device        : torch device (defaults to CPU)

        Returns:
            Generated lyric string (decoded, special tokens stripped)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        ids    = list(prompt_ids)
        hidden = self.init_hidden(1, device)
        eos_id = tokenizer.word2idx[tokenizer.EOS]

        # warm up hidden state on the prompt
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        _, hidden = self.forward(x, hidden)
        next_token = ids[-1]

        generated: list[int] = []
        for _ in range(max_new_tokens):
            x = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, hidden = self.forward(x, hidden)          # (1, 1, V)
            logits = logits[0, 0] / max(temperature, 1e-8)    # (V,)

            # top-k filter
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")

            # top-p (nucleus) filter
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == eos_id:
                break
            generated.append(next_token)

        return tokenizer.decode(generated)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LSTMLyricModel("
            f"vocab={self.vocab_size}, "
            f"embed={self.embed_dim}, "
            f"hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"params={self.count_parameters():,})"
        )