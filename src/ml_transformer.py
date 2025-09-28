"""Transformer-based suspect classifier.

Provides a fine-tuning pipeline using HuggingFace transformers (DistilBERT by default)
for multi-class suspect classification. Falls back gracefully if there are too few
samples.

Usage (programmatic):
    from ml_transformer import ensure_transformer_model, predict_labels
    ensure_transformer_model('inputs/sample_training.json')
    preds = predict_labels(["Some new clue text"], top_k=3)

Environment variables:
  TRANSFORMER_MODEL_NAME  (override default 'distilbert-base-uncased')
  TRANSFORMER_EPOCHS      (default 2 for speed)
  TRANSFORMER_BATCH       (default 8)

Model artifacts stored in models/transformer_model.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder

from transformers import (  # type: ignore
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'
TRANS_MODEL_DIR = MODEL_DIR / 'transformer_model'
META_PATH = TRANS_MODEL_DIR / 'meta.json'


def _default_model_name() -> str:
    return os.environ.get('TRANSFORMER_MODEL_NAME', 'distilbert-base-uncased')


@dataclass
class ClueItem:
    text: str
    label: str


class ClueDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'labels': self.labels[idx]}


def load_training_data(path: str | Path) -> Tuple[List[str], List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        texts.append(item['text'])
        labels.append(item['label'])
    return texts, labels


def ensure_transformer_model(training_json: str | Path) -> None:
    if (TRANS_MODEL_DIR / 'pytorch_model.bin').exists() and META_PATH.exists():
        return
    texts, labels = load_training_data(training_json)
    if len(set(labels)) < 2:
        raise ValueError('Need at least 2 distinct labels for transformer training')
    le = LabelEncoder()
    y = le.fit_transform(labels)
    model_name = _default_model_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize upfront (simple approach)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256)

    class EncodedDataset(Dataset):
        def __len__(self):
            return len(texts)
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in enc.items()} | {'labels': torch.tensor(y[idx])}

    train_dataset = EncodedDataset()

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

    epochs = int(os.environ.get('TRANSFORMER_EPOCHS', '2'))
    batch_size = int(os.environ.get('TRANSFORMER_BATCH', '8'))

    TRANS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(TRANS_MODEL_DIR / 'hf_out'),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy='no',
        report_to=[]
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()

    # Save
    model.save_pretrained(TRANS_MODEL_DIR)
    tokenizer.save_pretrained(TRANS_MODEL_DIR)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump({'labels': le.classes_.tolist(), 'model_name': model_name}, f)


def load_transformer():
    if not (TRANS_MODEL_DIR / 'pytorch_model.bin').exists() or not META_PATH.exists():
        return None, None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # local import
    tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANS_MODEL_DIR)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, tokenizer, meta['labels']


def predict_labels(texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    model, tokenizer, labels = load_transformer()
    if model is None:
        return []
    model.eval()
    with torch.no_grad():
        enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).mean(0)  # aggregate across inputs
        values, idxs = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        out = []
        for v, i in zip(values.tolist(), idxs.tolist()):
            out.append((labels[i], float(v)))
        return out


__all__ = ['ensure_transformer_model', 'predict_labels']
