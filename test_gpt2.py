#!/usr/bin/env python3
"""
Quick test to verify GPT-2 model implementation.
This script tests:
  1. Tokenizer fit/encode/decode
  2. Model instantiation
  3. Forward pass (logits)
  4. Generation (sampling)
"""

import torch
from models.gpt2 import LyricTokenizer, GPT2LyricModel


def main():
    print("=" * 60)
    print("GPT-2 Implementation Test")
    print("=" * 60)

    # Test data
    sample_lyrics = [
        "love is in the air\nfeeling so alive tonight\nevery moment with you feels right",
        "dancing in the moonlight\nwith you by my side\nnothing else matters when we're together",
        "dream of a better world\nwhere peace and love reign\nsinging songs of hope and harmony",
    ]

    # 1. Test Tokenizer
    print("\n[1/4] Testing Tokenizer...")
    tokenizer = LyricTokenizer(max_vocab=5000)
    tokenizer.fit(sample_lyrics)
    print(f"  ✓ Vocab size: {tokenizer.vocab_size}")
    print(f"  ✓ PAD index: {tokenizer.pad_idx}")

    # Encode/Decode
    encoded = tokenizer.encode("love is in the air")
    decoded = tokenizer.decode(encoded)
    print(f"  ✓ Encoded: {encoded[:10]}... (length: {len(encoded)})")
    print(f"  ✓ Decoded: {decoded}")

    # 2. Test Model Instantiation
    print("\n[2/4] Testing Model Instantiation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Device: {device}")

    model = GPT2LyricModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
        dropout=0.1,
        pad_idx=tokenizer.pad_idx,
    ).to(device)
    print(f"  ✓ Model created: {model}")

    # Count parameters
    params = model.count_parameters()
    print(f"  ✓ Total parameters: {params:,}")

    # 3. Test Forward Pass
    print("\n[3/4] Testing Forward Pass...")
    with torch.no_grad():
        x = torch.randint(0, tokenizer.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Logits shape: {logits.shape}")
    print(f"  ✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # 4. Test Generation
    print("\n[4/4] Testing Generation...")
    prompt = "love is in the"
    prompt_ids = tokenizer.encode(prompt)
    generated = model.generate(
        prompt_ids=prompt_ids,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        device=device,
    )
    print(f"  ✓ Prompt: {prompt}")
    print(f"  ✓ Generated: {generated}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
