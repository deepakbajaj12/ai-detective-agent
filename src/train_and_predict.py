from __future__ import annotations
import argparse
from pathlib import Path

from utils import read_clues
from ml_suspect_model import train_and_save, rank_labels, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train suspect model and predict top suspects from clues")
    parser.add_argument("--train", type=str, default=str(Path(__file__).resolve().parents[1] / "inputs" / "sample_training.json"), help="Path to training JSON file")
    parser.add_argument("--clues", type=str, default=str(Path(__file__).resolve().parents[1] / "inputs" / "sample_case.txt"), help="Path to clues text file")
    parser.add_argument("--topk", type=int, default=3, help="Top K suspects to display")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the model")
    args = parser.parse_args()

    model_exists = Path(MODEL_PATH).exists()
    if args.retrain or not model_exists:
        print("Training model...")
        report = train_and_save(args.train)
        print("\nValidation report:\n" + report)
    else:
        print(f"Using existing model at {MODEL_PATH}")

    clues = read_clues(args.clues)
    if not clues:
        print("No clues found to predict from.")
        return

    text = " ".join(clues)
    ranked = rank_labels([text], top_k=args.topk)
    print("\nTop suspects:")
    for label, score in ranked:
        print(f"- {label}: {score:.3f}")


if __name__ == "__main__":
    main()
