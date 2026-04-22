#!/usr/bin/env python3
"""
test_gpt2.py

Generate song lyrics from a pretrained GPT-2 checkpoint.

Usage
-----
    python test_gpt2.py
"""

import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.gpt2 import GPT2LyricModel

CHECKPOINT = "checkpoints/gpt2_small/best_model.pt"

PROMPTS = [
    "shout out my label thats me"
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}\n")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    with open(Path(CHECKPOINT).parent / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = GPT2LyricModel(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[ckpt] epoch {ckpt['epoch']} | val_loss {ckpt['val_loss']:.4f} | val_ppl {ckpt['val_ppl']:.2f}\n")

    for prompt in PROMPTS:
        output = model.generate(
            prompt_ids=tokenizer.encode(prompt),
            tokenizer=tokenizer,
            max_new_tokens=80,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            device=device,
        )
        print(f"Prompt: {prompt}")
        print(f"{output}")
        print("-" * 60)


if __name__ == "__main__":
    main()
