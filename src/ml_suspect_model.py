from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "suspect_model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def load_training_data(json_path: str | Path) -> Tuple[List[str], List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        texts.append(item["text"])  # a clue or description
        labels.append(item["label"])  # suspect name or event label
    return texts, labels


def train_and_save(json_path: str | Path, out_path: str | Path = MODEL_PATH) -> str:
    texts, labels = load_training_data(json_path)
    n_classes = len(set(labels))
    n_samples = len(labels)
    # If not enough samples for stratified split, fallback to non-stratified or no split
    if n_samples < n_classes * 2:
        # Train on all, no test set
        X_train, y_train = texts, labels
        X_test, y_test = [], []
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    if X_test and y_test:
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred)
    else:
        report = "Not enough data for test split. Model trained on all data."

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": sorted(set(labels)), "report": report}, out_path)

    return report


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    obj = joblib.load(path)
    return obj["pipeline"]


def rank_labels(texts: List[str], model: Pipeline | None = None, path: str | Path = MODEL_PATH, top_k: int = 3) -> List[Tuple[str, float]]:
    if model is None:
        model = load_model(path)
    # Use decision_function if available, else predict_proba
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(texts)
        classes = model.named_steps["clf"].classes_
        scores = proba.mean(axis=0)
    else:
        # Fallback: map decision scores to ranks (not calibrated)
        import numpy as np
        scores = model.decision_function(texts)
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T  # binary fallback
        classes = model.named_steps["clf"].classes_
        scores = scores.mean(axis=0)

    order = scores.argsort()[::-1]
    return [(str(classes[i]), float(scores[i])) for i in order[:top_k]]
