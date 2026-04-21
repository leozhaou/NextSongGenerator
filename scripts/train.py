"""
scripts/train.py

Training pipeline for song lyric completion models.

To add a new model, add an elif branch in get_model() and
get_tokenizer_and_dataset() below.

Usage
-----
    LSTM:
        python scripts/train.py \\
            --data       data/song_lyrics.csv \\
            --model      lstm \\
            --language   en \\
            --epochs     20 \\
            --batch_size 64 \\
            --seq_len    64 \\
            --embed_dim  256 \\
            --hidden_dim 512 \\
            --num_layers 2 \\
            --dropout    0.3 \\
            --lr         1e-3 \\
            --save_dir   checkpoints/lstm/

    GPT-2:
        python scripts/train.py \\
            --data       data/song_lyrics.csv \\
            --model      gpt2 \\
            --language   en \\
            --epochs     10 \\
            --batch_size 32 \\
            --seq_len    64 \\
            --hidden_dim 768 \\
            --num_layers 12 \\
            --dropout    0.1 \\
            --lr         5e-5 \\
            --save_dir   checkpoints/gpt2/
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

# Ensure project root is on the path when running as `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------------------------------
# Model loading  –  add a new elif here for each new model
# ---------------------------------------------------------------------------

def get_tokenizer_and_dataset(model_name: str):
    """Return the (TokenizerClass, DatasetClass) for a given model name."""
    if model_name == "lstm":
        from models.lstm import LyricTokenizer, LyricDataset
        return LyricTokenizer, LyricDataset
    elif model_name == "gpt2":
        from models.gpt2 import LyricTokenizer, LyricDataset
        return LyricTokenizer, LyricDataset
    else:
        raise ValueError(f"Unknown model '{model_name}'. Add it to get_tokenizer_and_dataset().")


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate and return the model for a given model name."""
    if model_name == "lstm":
        from models.lstm import LSTMLyricModel
        return LSTMLyricModel(**kwargs)
    elif model_name == "gpt2":
        from models.gpt2 import GPT2LyricModel
        return GPT2LyricModel(**kwargs)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Add it to get_model().")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lyrics(csv_path: str, language: str | None = None) -> list[str]:
    """
    Load and clean lyric strings from the CSV.

    Expected columns: title, tag, artist, year, views, features,
                      lyrics, id, language_cld3, language_ft, language
    """
    df = pd.read_csv(csv_path)

    if language:
        mask = (
            (df["language"].str.lower() == language.lower()) |
            (df["language_ft"].str.lower() == language.lower())
        )
        df = df[mask]
        print(f"[data] Filtered to language='{language}': {len(df):,} songs")
    else:
        print(f"[data] Loaded {len(df):,} songs (no language filter)")

    df = df.dropna(subset=["lyrics"])
    # Fast char-length proxy first (~5 chars/word * 20 words = 100 chars), then exact check
    df = df[df["lyrics"].str.len() >= 100]
    df = df[df["lyrics"].str.split().str.len() >= 20]
    lyrics = df["lyrics"].tolist()
    print(f"[data] {len(lyrics):,} usable lyrics after cleaning")
    return lyrics


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 300))


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"[ckpt] Saved → {path}")


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    grad_clip: float = 1.0,
    max_steps_per_epoch: int | None = None,
) -> float:
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    for step, (x, y) in enumerate(loader):
        if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
            break
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        tokens        = y.numel()
        total_loss   += loss.item() * tokens
        total_tokens += tokens

        if (step + 1) % 100 == 0:
            ppl = compute_perplexity(total_loss / total_tokens)
            print(f"  step {step+1:>5} | loss {total_loss/total_tokens:.4f} | ppl {ppl:.2f}")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tokens        = y.numel()
        total_loss   += loss.item() * tokens
        total_tokens += tokens

        if (step + 1) % 100 == 0:
            print(f"  [val] step {step+1:>5} | loss {total_loss/total_tokens:.4f}")

    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── data ──────────────────────────────────────────────────────────────
    lyrics = load_lyrics(args.data, language=args.language)

    TokenizerClass, DatasetClass = get_tokenizer_and_dataset(args.model)
    tokenizer = TokenizerClass(max_vocab=args.vocab_size)
    tokenizer.fit(lyrics)
    print(f"[vocab] size = {tokenizer.vocab_size:,}")

    dataset = DatasetClass(lyrics, tokenizer, seq_len=args.seq_len, stride=args.stride)
    print(f"[dataset] {len(dataset):,} windows  (seq_len={args.seq_len})")

    val_size   = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"[split] train={len(train_ds):,}  val={len(val_ds):,}")

    # ── model ─────────────────────────────────────────────────────────────
    if args.model == "lstm":
        model_kwargs = dict(
            vocab_size  = tokenizer.vocab_size,
            embed_dim   = args.embed_dim,
            hidden_dim  = args.hidden_dim,
            num_layers  = args.num_layers,
            dropout     = args.dropout,
            pad_idx     = tokenizer.pad_idx,
            tie_weights = args.tie_weights,
        )
    elif args.model == "gpt2":
        model_kwargs = dict(
            vocab_size       = tokenizer.vocab_size,
            hidden_size      = args.hidden_dim,
            num_hidden_layers= args.num_layers,
            num_attention_heads= 8 if args.hidden_dim < 1024 else 12,
            intermediate_size= args.hidden_dim * 4,
            dropout          = args.dropout,
            pad_idx          = tokenizer.pad_idx,
            use_cache        = True,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = get_model(args.model, **model_kwargs).to(device)
    print(f"[model] {model}")

    # ── optimiser & loss ──────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # ── checkpoint setup ──────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[ckpt] Tokenizer saved → {save_dir / 'tokenizer.pkl'}")

    # ── training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_clip,
            args.max_steps_per_epoch,
        )
        val_loss   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_ppl = compute_perplexity(train_loss)
        val_ppl   = compute_perplexity(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train loss {train_loss:.4f}  ppl {train_ppl:.2f} | "
            f"val loss {val_loss:.4f}  ppl {val_ppl:.2f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch"          : epoch,
                    "model_name"     : args.model,
                    "model_kwargs"   : model_kwargs,
                    "model_state"    : model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss"       : val_loss,
                    "val_ppl"        : val_ppl,
                    "train_args"     : vars(args),
                },
                str(save_dir / "best_model.pt"),
            )

        if epoch % args.save_every == 0:
            save_checkpoint(
                {
                    "epoch"       : epoch,
                    "model_name"  : args.model,
                    "model_kwargs": model_kwargs,
                    "model_state" : model.state_dict(),
                    "val_loss"    : val_loss,
                },
                str(save_dir / f"epoch_{epoch:03d}.pt"),
            )

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[done] Best val perplexity : {compute_perplexity(best_val_loss):.2f}")
    print(f"[done] Training history    → {save_dir / 'history.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a lyric-completion model")

    # data
    parser.add_argument("--data",         default="data/song_lyrics.csv")
    parser.add_argument("--language",     default="en",
                        help="Language code to filter ('en', 'es', …). Empty = all.")
    parser.add_argument("--vocab_size",   type=int,   default=20_000)
    parser.add_argument("--seq_len",      type=int,   default=64)
    parser.add_argument("--stride",       type=int,   default=0,
                        help="Sliding-window stride. 0 = auto (seq_len // 2).")
    parser.add_argument("--val_split",    type=float, default=0.1)

    # model
    parser.add_argument("--model",        default="lstm",
                        help="Model to train: lstm | transformer | ...")
    parser.add_argument("--embed_dim",    type=int,   default=256)
    parser.add_argument("--hidden_dim",   type=int,   default=512)
    parser.add_argument("--num_layers",   type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--tie_weights",  action="store_true")

    # training
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0,
                        help="Max training batches per epoch. 0 = full epoch.")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)

    # checkpointing
    parser.add_argument("--save_dir",     default="checkpoints/lstm/")
    parser.add_argument("--save_every",   type=int,   default=5)

    args = parser.parse_args()
    if args.language == "":
        args.language = None
    if args.stride == 0:
        args.stride = args.seq_len // 2
        print(f"[stride] auto → {args.stride}")
    if args.max_steps_per_epoch <= 0:
        args.max_steps_per_epoch = None
    else:
        print(f"[steps] limiting training to {args.max_steps_per_epoch:,} batches/epoch")

    main(args)