# NextSongGenerator

A neural network-based song lyric generation system supporting multiple architectures (LSTM, BERT, GPT-2) for fine-tuned text generation.

## Overview

This project implements transformer and RNN-based language models to generate song lyrics with coherence and musical quality. The pipeline includes data preprocessing, model training, evaluation metrics, and human annotation templates.

### Models Supported

- **LSTM** – Multi-layer LSTM with sliding-window training
- **GPT-2** – Transformer-based autoregressive model (fine-tuned HuggingFace)
- **BERT-base-uncased** – Masked language model fine-tuning for lyric fill-in

## Project Structure

```
├── models/
│   ├── lstm.py          # LSTM architecture + custom tokenizer
│   ├── gpt2.py          # GPT-2 transformer implementation
│   ├── bert.py          # BERT MLM training helpers
│   └── __init__.py
├── scripts/
│   ├── train.py         # Training pipeline (supports all models)
│   ├── evaluate.py      # Comprehensive evaluation metrics
│   └── preprocess.py    # Data filtering & cleaning
├── notebooks/
│   ├── exploration.ipynb # Data analysis
│   └── results.ipynb     # Results visualization
├── results/
│   └── lstm/            # LSTM outputs (metrics, generations)
├── config.py            # Configuration constants
├── requirements.txt     # Python dependencies
└── song_lyrics.csv      # Training dataset
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `torch >= 2.0` – PyTorch
- `transformers >= 4.38` – Hugging Face models
- `pandas`, `numpy` – Data processing
- `scikit-learn` – Evaluation metrics (TF-IDF, BLEU)
- `nltk` – Natural language tools
- `tqdm` – Progress bars

## Quick Start

### 1. Preprocess Data

```bash
python scripts/preprocess.py
```

This filters lyrics to English, samples 12.5% of data, and cleans text.

### 2. Train LSTM

```bash
python scripts/train.py \
    --model lstm \
    --data data/song_lyrics.csv \
    --epochs 20 \
    --batch_size 64 \
    --seq_len 64 \
    --hidden_dim 512 \
    --save_dir checkpoints/lstm/
```

### 3. Train GPT-2

```bash
python scripts/train.py \
    --model gpt2 \
    --data data/song_lyrics.csv \
    --epochs 10 \
    --batch_size 32 \
    --seq_len 64 \
    --hidden_dim 768 \
    --num_layers 12 \
    --lr 5e-5 \
    --save_dir checkpoints/gpt2/
```

### 4. Train BERT (Masked LM)

```bash
python scripts/train.py \
    --model bert \
    --data data/song_lyrics.csv \
    --save_dir checkpoints/bert \
    --language en \
    --min_words 50 \
    --seq_len 128 \
    --epochs 3 \
    --batch_size 16 \
    --lr 5e-5
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/gpt2/best_model.pt \
    --tokenizer checkpoints/gpt2/tokenizer.pkl \
    --data data/song_lyrics.csv \
    --num_samples 200 \
    --output_dir results/gpt2/
```

## Evaluation Metrics

The evaluation suite (`scripts/evaluate.py`) computes:

1. **Perplexity** – Language model quality (lower is better)
2. **BLEU Scores** – N-gram overlap with references (1-4 grams)
3. **Rhyme Preservation** – Ratio of rhyming line-endings
4. **Theme Similarity** – TF-IDF cosine similarity to references
5. **Human Evaluation** – Blind annotation template (JSON + printable sheet)

### Output Files

- `generations.json` – Prompt, reference, and hypothesis completions
- `metrics.json` – All computed metrics
- `human_eval.json` – Structured evaluation template
- `human_eval.txt` – Printable evaluation sheet

## Model Architecture

### LSTM

```
Embedding (256) → Dropout → LSTM (2 layers, 512 hidden)
    → Dropout → Linear (vocab_size)
```

- **Parameters**: ~15M typical
- **Training time**: ~2-3 min/epoch (on GPU)
- **Best for**: Fast training, memory-efficient

### GPT-2

```
Token Embedding (768) → Positional Embedding → 12 Transformer Blocks
    → Layer Norm → LM Head (vocab_size)
```

- **Parameters**: ~125M typical (standard GPT-2)
- **Training time**: ~10-15 min/epoch (on GPU)
- **Best for**: Higher quality generations, transfer learning

## Tokenizer

Both models use the same custom word-level tokenizer with special tokens:

- `<PAD>` – Padding (index 0)
- `<UNK>` – Unknown words (index 1)
- `<BOS>` – Beginning of sequence (index 2)
- `<EOS>` – End of sequence (index 3)
- `<NL>` – Newline/line break (index 4)

## Generation Sampling

Both models support:
- **Temperature scaling** – Controls randomness (0.8 typical)
- **Top-k filtering** – Keep only top-k highest probability tokens (50 typical)
- **Nucleus sampling (top-p)** – Keep tokens until cumulative probability exceeds threshold (0.95 typical)

## Training Pipeline

Both models trained with:
- **Optimizer**: Adam (lr=1e-3 for LSTM, 5e-5 for GPT-2)
- **Loss**: Cross-entropy with padding masking
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Gradient clipping**: 1.0 norm
- **Validation split**: 10%

## Configuration

Edit `config.py` to adjust:
- `MAX_SEQ_LEN` – Max sequence length (default: 128)
- `BATCH_SIZE` – Training batch size (default: 32)
- `EPOCHS` – Training epochs (default: 5)

## Example Generation

**LSTM Prompt**: *"love is in the"*

```
Generated: "love is in the air tonight feeling so alive
and every moment with you feels so right"
```

**GPT-2 Prompt**: *"love is in the"*

```
Generated: "love is in the air all around us now
a beautiful feeling that we cannot deny
every heartbeat reminds me that you're mine"
```

## Directory Layout

```
NextSongGenerator/
├── data/
│   ├── raw/              # Original lyrics CSV
│   └── processed/        # Cleaned & filtered data
├── checkpoints/
│   ├── lstm/
│   │   ├── best_model.pt
│   │   ├── tokenizer.pkl
│   │   └── history.json
│   └── gpt2/
│       ├── best_model.pt
│       ├── tokenizer.pkl
│       └── history.json
├── results/
│   ├── lstm/
│   │   ├── metrics.json
│   │   ├── generations.json
│   │   └── human_eval.json
│   └── gpt2/
│       ├── metrics.json
│       ├── generations.json
│       └── human_eval.json
└── notebooks/
    ├── exploration.ipynb
    └── results.ipynb
```

## Known Issues

- `preprocess.py` contains debug print statement ("I am a big dummy") – should be removed
- BERT is MLM-based (fill-mask), not autoregressive generation
- Requires filtered English lyrics CSV

## Future Improvements

- [ ] Add BERT-specific evaluation pipeline
- [ ] Add rhythm-aware loss function
- [ ] Support multi-language training
- [ ] Implement attention visualization
- [ ] Add inference REST API
- [ ] Fine-tune on artist-specific datasets

