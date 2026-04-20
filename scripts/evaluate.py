"""
scripts/evaluate.py

Evaluation suite for song lyric completion models.

Metrics
-------
  1. Perplexity          – standard LM quality (lower = better)
  2. BLEU score          – n-gram overlap vs. reference (higher = better)
  3. Rhyme preservation  – how well end-of-line rhymes are maintained
  4. Theme preservation  – cosine similarity of TF-IDF topic vectors
  5. Human eval template – structured JSON + printable sheet for blind annotation

To add a new model, add an elif branch in load_model_from_checkpoint() below.

Usage
-----
    python scripts/evaluate.py \\
        --checkpoint checkpoints/lstm/best_model.pt \\
        --tokenizer  checkpoints/lstm/tokenizer.pkl \\
        --data       data/song_lyrics.csv \\
        --language   en \\
        --num_samples 200 \\
        --prompt_len  20 \\
        --gen_len     80 \\
        --human_eval_n 10 \\
        --output_dir  results/lstm/
"""

import argparse
import json
import math
import os
import pickle
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from tqdm import tqdm as _tqdm
    def tqdm(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def tqdm(it, desc="", **kw):
        print(f"[{desc}] (install tqdm for a progress bar)")
        return it


# ---------------------------------------------------------------------------
# Model loading  –  add a new elif here for each new model
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, tokenizer, device: torch.device):
    """
    Reconstruct a model from a checkpoint saved by train.py.

    The checkpoint contains:
        model_name   – which architecture was trained
        model_kwargs – the exact constructor arguments used
        model_state  – the trained weights

    Add an elif block below whenever a new model is added to the project.

    Returns (model, model_name) — avoids a second torch.load in the caller.
    """
    ckpt        = torch.load(ckpt_path, map_location=device)
    model_name  = ckpt["model_name"]
    model_kwargs = ckpt.get("model_kwargs", {})

    # disable dropout at inference time
    model_kwargs["dropout"] = 0.0

    if model_name == "lstm":
        from models.lstm import LSTMLyricModel
        model = LSTMLyricModel(**model_kwargs)
    # example for other models
    # elif model_name == "transformer":
    #     from models.transformer import TransformerLyricModel
    #     model = TransformerLyricModel(**model_kwargs)
    # elif model_name == "gpt2":
    #     from models.gpt2 import GPT2LyricModel
    #     model = GPT2LyricModel(**model_kwargs)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Add it to load_model_from_checkpoint().")

    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), model_name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lyrics(csv_path: str, language: str | None = None) -> list[str]:
    df = pd.read_csv(csv_path)
    if language:
        mask = (
            (df["language"].str.lower() == language.lower()) |
            (df["language_ft"].str.lower() == language.lower())
        )
        df = df[mask]
    df = df.dropna(subset=["lyrics"])
    # Fast char-length proxy first (~5 chars/word * 40 words = 200 chars), then exact check
    df = df[df["lyrics"].str.len() >= 200]
    df = df[df["lyrics"].str.split().str.len() >= 40]
    return df["lyrics"].tolist()


def tokenize_words(text: str) -> list[str]:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity_on_samples(
    model,
    tokenizer,
    lyrics_sample: list[str],
    device,
    seq_len: int = 64,
    batch_size: int = 64,
) -> float:
    """Token-averaged cross-entropy loss → perplexity on a held-out set.
    Processes windows in batches for ~batch_size× speedup over per-lyric inference.
    """
    criterion    = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, reduction="sum")
    total_loss   = 0.0
    total_tokens = 0

    # Collect all windows first
    all_x, all_y = [], []
    for lyric in lyrics_sample:
        ids = tokenizer.encode(lyric)
        if len(ids) < seq_len + 1:
            continue
        for start in range(0, len(ids) - seq_len, seq_len):
            chunk = ids[start : start + seq_len + 1]
            all_x.append(chunk[:-1])
            all_y.append(chunk[1:])

    model.eval()
    for i in range(0, len(all_x), batch_size):
        x = torch.tensor(all_x[i : i + batch_size], dtype=torch.long, device=device)
        y = torch.tensor(all_y[i : i + batch_size], dtype=torch.long, device=device)
        logits, _ = model(x)
        loss      = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 300))


# ---------------------------------------------------------------------------
# 2. BLEU Score
# ---------------------------------------------------------------------------

def compute_bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> dict[str, float]:
    """Corpus-level BLEU-1 through BLEU-4 plus an average."""
    refs = [[tokenize_words(r)] for r in references]
    hyps = [tokenize_words(h)   for h in hypotheses]
    sf   = SmoothingFunction().method1

    scores: dict[str, float] = {}
    for n in range(1, max_n + 1):
        weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
        scores[f"bleu-{n}"] = corpus_bleu(refs, hyps, weights=weights, smoothing_function=sf)

    scores["bleu-avg"] = float(np.mean(list(scores.values())))
    return scores


# ---------------------------------------------------------------------------
# 3. Rhyme Preservation
# ---------------------------------------------------------------------------

def get_line_endings(text: str) -> list[str]:
    endings = []
    for line in text.split("\n"):
        words = line.strip().split()
        if words:
            endings.append(words[-1].lower().strip(string.punctuation))
    return endings


def simple_rhyme_suffix(word: str, n: int = 3) -> str:
    return word[-n:] if len(word) >= n else word


def rhyme_density(text: str, suffix_len: int = 3) -> float:
    """Fraction of consecutive line-ending pairs that share a rhyme suffix."""
    endings = get_line_endings(text)
    if len(endings) < 2:
        return 0.0
    pairs   = list(zip(endings, endings[1:]))
    rhyming = sum(
        1 for a, b in pairs
        if simple_rhyme_suffix(a, suffix_len) == simple_rhyme_suffix(b, suffix_len)
    )
    return rhyming / len(pairs)


def compute_rhyme_preservation(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    ref_densities = [rhyme_density(r) for r in references]
    hyp_densities = [rhyme_density(h) for h in hypotheses]

    ratios = [
        min(h / r, 2.0)
        for r, h in zip(ref_densities, hyp_densities)
        if r > 0
    ]

    return {
        "ref_rhyme_density_mean" : float(np.mean(ref_densities)) if ref_densities else 0.0,
        "hyp_rhyme_density_mean" : float(np.mean(hyp_densities)) if hyp_densities else 0.0,
        "rhyme_preservation_mean": float(np.mean(ratios))        if ratios else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Theme Preservation (TF-IDF cosine similarity)
# ---------------------------------------------------------------------------

def compute_theme_preservation(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Cosine similarity between TF-IDF vectors of reference and hypothesis."""
    all_texts  = references + hypotheses
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf    = vectorizer.fit_transform(all_texts)
    n        = len(references)
    ref_vecs = tfidf[:n]
    hyp_vecs = tfidf[n:]

    # Vectorised row-wise dot product then normalise — avoids a Python loop
    # of n individual sklearn calls.
    ref_norm = np.asarray(ref_vecs.multiply(ref_vecs).sum(axis=1)).ravel() ** 0.5
    hyp_norm = np.asarray(hyp_vecs.multiply(hyp_vecs).sum(axis=1)).ravel() ** 0.5
    dot      = np.asarray(ref_vecs.multiply(hyp_vecs).sum(axis=1)).ravel()
    denom    = ref_norm * hyp_norm
    sims     = np.where(denom > 0, dot / denom, 0.0)

    return {
        "theme_sim_mean": float(np.mean(sims)),
        "theme_sim_std" : float(np.std(sims)),
        "theme_sim_min" : float(np.min(sims)),
        "theme_sim_max" : float(np.max(sims)),
    }


# ---------------------------------------------------------------------------
# 5. Human Evaluation Template
# ---------------------------------------------------------------------------

def generate_human_eval_sheet(
    prompts    : list[str],
    references : list[str],
    hypotheses : list[str],
    output_path: str,
    n_samples  : int = 10,
    model_name : str = "Model",
) -> None:
    """
    Write a blind human-evaluation sheet (JSON + printable plain text).

    Criteria rated 1-5: Fluency, Coherence, Creativity, Rhyme, Overall preference.
    A/B assignment is randomised per sample so position doesn't reveal identity.
    """
    rng     = np.random.default_rng(seed=0)
    indices = rng.choice(len(prompts), size=min(n_samples, len(prompts)), replace=False)

    items = []
    for i, idx in enumerate(indices, 1):
        flip  = bool(rng.integers(2))
        sys_a = hypotheses[idx] if not flip else references[idx]
        sys_b = references[idx] if not flip else hypotheses[idx]
        items.append({
            "id"          : i,
            "prompt"      : prompts[idx],
            "system_A"    : sys_a,
            "system_B"    : sys_b,
            "_a_is_model" : not flip,
            "ratings"     : {
                "fluency_A"          : None,
                "fluency_B"          : None,
                "coherence_A"        : None,
                "coherence_B"        : None,
                "creativity_A"       : None,
                "creativity_B"       : None,
                "rhyme_A"            : None,
                "rhyme_B"            : None,
                "overall_preference" : None,
            },
            "comments": "",
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    txt_path = output_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"SONG LYRIC COMPLETION – HUMAN EVALUATION  [{model_name}]\n")
        f.write("=" * 60 + "\n\n")
        f.write("Rate each criterion 1 (poor) → 5 (excellent).\n")
        f.write("Overall preference: A / B / tie\n\n")
        for item in items:
            f.write(f"{'─'*60}\nSample #{item['id']}\n\n")
            f.write(f"PROMPT:\n{item['prompt']}\n\n")
            f.write(f"SYSTEM A:\n{item['system_A']}\n\n")
            f.write(f"SYSTEM B:\n{item['system_B']}\n\n")
            f.write("Ratings:\n")
            for key in item["ratings"]:
                f.write(f"  {key:<25}: ______\n")
            f.write("Comments: _________________________________\n\n")

    print(f"[human-eval] JSON  → {output_path}")
    print(f"[human-eval] Plain → {txt_path}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_samples(
    model,
    tokenizer,
    lyrics     : list[str],
    prompt_len : int,
    gen_len    : int,
    device     : torch.device,
    temperature: float = 0.8,
    top_k      : int   = 50,
    top_p      : float = 0.95,
) -> tuple[list[str], list[str], list[str]]:
    """
    For each lyric use the first `prompt_len` tokens as the prompt,
    generate `gen_len` new tokens, and treat the rest as the reference.

    Returns (prompts_text, references_text, hypotheses_text).
    """
    prompts, references, hypotheses = [], [], []

    for lyric in tqdm(lyrics, desc="generating", unit="lyric"):
        ids = tokenizer.encode(lyric)
        if len(ids) < prompt_len + 10:
            continue

        prompt_ids = ids[:prompt_len]
        ref_ids    = ids[prompt_len:]

        generated = model.generate(
            prompt_ids     = prompt_ids,
            tokenizer      = tokenizer,
            max_new_tokens = gen_len,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
            device         = device,
        )

        prompts.append(tokenizer.decode(prompt_ids))
        references.append(tokenizer.decode(ref_ids))
        hypotheses.append(generated)

    return prompts, references, hypotheses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── load tokenizer & model ────────────────────────────────────────────
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"[tokenizer] vocab size = {tokenizer.vocab_size:,}")

    model, model_name = load_model_from_checkpoint(args.checkpoint, tokenizer, device)
    print(f"[model] {model}")

    # ── load data ─────────────────────────────────────────────────────────
    lyrics = load_lyrics(args.data, language=args.language)
    rng    = np.random.default_rng(args.seed)
    sample = [
        lyrics[i]
        for i in rng.choice(len(lyrics), size=min(args.num_samples, len(lyrics)), replace=False)
    ]
    print(f"[eval] {len(sample)} samples  (model: {model_name})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": model_name}

    # ── 1. Perplexity ─────────────────────────────────────────────────────
    print("\n[1/5] Computing perplexity …")
    ppl = compute_perplexity_on_samples(model, tokenizer, sample, device, seq_len=args.seq_len)
    results["perplexity"] = ppl
    print(f"  Perplexity: {ppl:.2f}")

    # ── generate completions ──────────────────────────────────────────────
    print("\n[gen] Generating completions …")
    prompts, references, hypotheses = generate_samples(
        model, tokenizer, sample,
        prompt_len  = args.prompt_len,
        gen_len     = args.gen_len,
        device      = device,
        temperature = args.temperature,
        top_k       = args.top_k,
        top_p       = args.top_p,
    )
    print(f"  Generated {len(hypotheses)} completions")

    with open(output_dir / "generations.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"prompt": p, "reference": r, "hypothesis": h}
             for p, r, h in zip(prompts, references, hypotheses)],
            f, indent=2, ensure_ascii=False,
        )
    print(f"  Saved → {output_dir / 'generations.json'}")

    # ── 2. BLEU ───────────────────────────────────────────────────────────
    print("\n[2/5] Computing BLEU …")
    bleu = compute_bleu(references, hypotheses)
    results["bleu"] = bleu
    for k, v in bleu.items():
        print(f"  {k}: {v:.4f}")

    # ── 3. Rhyme preservation ─────────────────────────────────────────────
    print("\n[3/5] Computing rhyme preservation …")
    rhyme = compute_rhyme_preservation(references, hypotheses)
    results["rhyme"] = rhyme
    for k, v in rhyme.items():
        print(f"  {k}: {v:.4f}")

    # ── 4. Theme preservation ─────────────────────────────────────────────
    print("\n[4/5] Computing theme preservation (TF-IDF cosine similarity) …")
    theme = compute_theme_preservation(references, hypotheses)
    results["theme"] = theme
    for k, v in theme.items():
        print(f"  {k}: {v:.4f}")

    # ── 5. Human eval sheet ───────────────────────────────────────────────
    print(f"\n[5/5] Writing human evaluation sheet (n={args.human_eval_n}) …")
    generate_human_eval_sheet(
        prompts, references, hypotheses,
        output_path = str(output_dir / "human_eval.json"),
        n_samples   = args.human_eval_n,
        model_name  = model_name,
    )

    # ── summary ───────────────────────────────────────────────────────────
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[done] All metrics → {output_dir / 'metrics.json'}")
    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY  [{model_name}]")
    print("=" * 50)
    print(f"  Perplexity          : {results['perplexity']:.2f}")
    print(f"  BLEU-1              : {results['bleu']['bleu-1']:.4f}")
    print(f"  BLEU-4              : {results['bleu']['bleu-4']:.4f}")
    print(f"  Rhyme preservation  : {results['rhyme']['rhyme_preservation_mean']:.4f}")
    print(f"  Theme similarity    : {results['theme']['theme_sim_mean']:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a lyric-completion model")

    # paths
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to .pt checkpoint produced by train.py")
    parser.add_argument("--tokenizer",    required=True,
                        help="Path to tokenizer .pkl produced by train.py")
    parser.add_argument("--data",         default="data/song_lyrics.csv")
    parser.add_argument("--output_dir",   default="results/eval/")

    # data
    parser.add_argument("--language",     default="en",
                        help="Language filter. Empty = all.")
    parser.add_argument("--num_samples",  type=int,   default=200)
    parser.add_argument("--seq_len",      type=int,   default=64)
    parser.add_argument("--seed",         type=int,   default=42)

    # generation
    parser.add_argument("--prompt_len",   type=int,   default=20)
    parser.add_argument("--gen_len",      type=int,   default=80)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_k",        type=int,   default=50)
    parser.add_argument("--top_p",        type=float, default=0.95)

    # human eval
    parser.add_argument("--human_eval_n", type=int,   default=10)

    args = parser.parse_args()
    if args.language == "":
        args.language = None

    main(args)