"""
models/gpt2.py

GPT-2 transformer-based language model for song lyric completion.
Wraps Hugging Face transformers GPT-2 with custom tokenizer integration.

This model is fine-tuned on filtered lyrics and designed for coherent
multi-line generation with controlled sampling (temperature, top-k, top-p).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LyricDataset(Dataset):
    """
    Sliding-window token dataset for GPT-2 fine-tuning.
    
    Matches the LSTM dataset interface for compatibility with training scripts.

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
    
    Matches the LSTM tokenizer interface for model interchangeability.

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

class GPT2LyricModel(nn.Module):
    """
    GPT-2 based transformer language model for lyric completion.

    Uses Hugging Face's GPT-2 architecture with custom initialization
    for fine-tuning on song lyrics. Provides interface compatible with LSTM.

    Architecture
    --------
        Token Embedding  →  Positional Embedding  →  Transformer Blocks  →  LM Head

    Args:
        vocab_size      : size of the token vocabulary
        hidden_size     : transformer hidden dimension
        num_hidden_layers : number of transformer blocks
        num_attention_heads : number of attention heads
        intermediate_size : FFN hidden dimension
        dropout         : dropout probability
        pad_idx         : vocabulary index of <PAD> token
        use_cache       : whether to use KV cache for generation
    """

    def __init__(
        self,
        vocab_size      : int,
        hidden_size     : int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout         : float = 0.1,
        pad_idx         : int = 0,
        use_cache       : bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.pad_idx = pad_idx
        self.use_cache = use_cache

        # Configure and initialize GPT-2
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            n_inner=intermediate_size,
            activation_function="gelu",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=use_cache,
            pad_token_id=pad_idx,
        )

        self.gpt2 = GPT2LMHeadModel(config)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Initialize weights with small values for stable fine-tuning."""
        for module in self.gpt2.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if self.pad_idx is not None:
                    module.weight.data[self.pad_idx].zero_()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x      : (batch, seq_len) token indices
            hidden : ignored (kept for LSTM interface compatibility)

        Returns:
            logits : (batch, seq_len, vocab_size)
            hidden : None (transformer doesn't use explicit hidden states)
        """
        # GPT-2 returns CausalLMOutputWithPast; we extract logits
        outputs = self.gpt2(
            input_ids=x,
            attention_mask=(x != self.pad_idx).long(),
            return_dict=True,
        )
        return outputs.logits, None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids  : list[int],
        tokenizer,
        max_new_tokens: int = 100,
        temperature : float = 0.8,
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
            device        : torch device (defaults to model device)

        Returns:
            Generated lyric string (decoded, special tokens stripped)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        eos_id = tokenizer.word2idx[tokenizer.EOS]

        generated: list[int] = []
        past_key_values = None

        for step in range(max_new_tokens):
            # If using cache, only pass the last token after the first step
            if self.use_cache and past_key_values is not None:
                model_input = ids[:, -1:]
            else:
                model_input = ids

            outputs = self.gpt2(
                input_ids=model_input,
                attention_mask=(ids != self.pad_idx).long(),
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=self.use_cache,
            )

            logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)
            past_key_values = outputs.past_key_values if self.use_cache else None

            # top-k filter
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[0, -1]
                logits[logits < kth] = float("-inf")

            # top-p (nucleus) filter
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[:, remove[0]] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs[0], 1).item()

            if next_token == eos_id:
                break

            ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)
            generated.append(next_token)

        return tokenizer.decode(generated)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GPT2LyricModel("
            f"vocab={self.vocab_size}, "
            f"hidden={self.hidden_size}, "
            f"layers={self.num_hidden_layers}, "
            f"params={self.count_parameters():,})"
        )
