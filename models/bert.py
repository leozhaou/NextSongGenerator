"""
models/bert.py

Utilities for BERT masked-language-model (MLM) lyric training.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


@dataclass
class BertArtifacts:
    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM
    data_collator: DataCollatorForLanguageModeling


def build_bert_artifacts(
    model_name: str = "bert-base-uncased",
    mlm_probability: float = 0.15,
) -> BertArtifacts:
    """
    Build tokenizer, model, and data collator for MLM fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    return BertArtifacts(
        tokenizer=tokenizer,
        model=model,
        data_collator=data_collator,
    )
