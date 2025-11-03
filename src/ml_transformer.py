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
from dataclasses import dataclass

# Optional heavy deps: torch/transformers. Import lazily/optionally so API can start without them.
try:  # torch is optional
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _HAS_TORCH = False

try:  # transformers is optional
    from transformers import (  # type: ignore
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments
    )
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    AutoTokenizer = AutoModelForSequenceClassification = Trainer = TrainingArguments = None  # type: ignore
    _HAS_TRANSFORMERS = False

from sklearn.preprocessing import LabelEncoder

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
    """
    Ensure a transformer model is trained and saved to disk.
    If torch/transformers are not installed, this becomes a no-op so the
    application can rely on classic TF-IDF fallback instead.
    """
    # If already present, nothing to do
    if (TRANS_MODEL_DIR / 'pytorch_model.bin').exists() and META_PATH.exists():
        return
    # If dependencies missing, skip silently (backend should still work with classic model)
    if not (_HAS_TORCH and _HAS_TRANSFORMERS):
        # Optionally log a hint
        print('[ml_transformer] torch/transformers not available; skipping transformer training (fallback to classic model).')
        return
    texts, labels = load_training_data(training_json)
    if len(set(labels)) < 2:
        print('[ml_transformer] Not enough label diversity to train transformer; skipping.')
        return
    le = LabelEncoder()
    y = le.fit_transform(labels)
    model_name = _default_model_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore

    # Tokenize upfront (simple approach)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256)  # type: ignore

    if not _HAS_TORCH:
        print('[ml_transformer] torch not available; cannot proceed with training.')
        return

    class EncodedDataset(Dataset):  # type: ignore
        def __len__(self):
            return len(texts)
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in enc.items()} | {'labels': torch.tensor(y[idx])}  # type: ignore

    train_dataset = EncodedDataset()

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))  # type: ignore

    epochs = int(os.environ.get('TRANSFORMER_EPOCHS', '2'))
    batch_size = int(os.environ.get('TRANSFORMER_BATCH', '8'))

    TRANS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(  # type: ignore
        output_dir=str(TRANS_MODEL_DIR / 'hf_out'),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy='no',
        report_to=[]
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)  # type: ignore
    trainer.train()

    # Save
    model.save_pretrained(TRANS_MODEL_DIR)
    tokenizer.save_pretrained(TRANS_MODEL_DIR)  # type: ignore
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump({'labels': le.classes_.tolist(), 'model_name': model_name}, f)


def load_transformer():
    """Load a saved transformer model if available and dependencies installed.
    Returns (model, tokenizer, labels) or (None, None, None).
    """
    if not (TRANS_MODEL_DIR / 'pytorch_model.bin').exists() or not META_PATH.exists():
        return None, None, None
    if not _HAS_TRANSFORMERS:
        print('[ml_transformer] transformers not available; cannot load transformer model.')
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL_DIR)  # type: ignore
    model = AutoModelForSequenceClassification.from_pretrained(TRANS_MODEL_DIR)  # type: ignore
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, tokenizer, meta['labels']


def predict_labels(texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Predict labels using the saved transformer model. If dependencies or
    artifacts are missing, returns an empty list so callers can fallback
    to a classic model.
    """
    model, tokenizer, labels = load_transformer()
    if model is None or tokenizer is None or not _HAS_TORCH:
        return []
    model.eval()
    with torch.no_grad():  # type: ignore
        enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')  # type: ignore
        logits = model(**enc).logits
        # Average across inputs if multiple
        probs = torch.softmax(logits, dim=-1).mean(0)  # type: ignore
        values, idxs = torch.topk(probs, k=min(top_k, probs.shape[-1]))  # type: ignore
        out = []
        for v, i in zip(values.tolist(), idxs.tolist()):
            out.append((labels[i], float(v)))
        return out


__all__ = ['ensure_transformer_model', 'predict_labels']
