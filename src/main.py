import os
from pathlib import Path
try:
    from src.utils import read_clues
except ImportError:  # fallback when run as script
    from utils import read_clues  # type: ignore
from reasoning_agent import analyze_clues
from pdf_generator import save_report

if __name__ == "__main__":
    print("\n🕵️ AI Detective Agent")
    print("=" * 22)

    # Resolve project root (one level up from this file's directory)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Read clues
    input_path = BASE_DIR / "inputs" / "sample_case.txt"
    clues = read_clues(str(input_path))
    print("Clues received:")
    for clue in clues:
        print(f"- {clue}")

    # Analyze clues
    deductions = analyze_clues(clues)

    print("\nDeductions:")
    for deduction in deductions:
        print(f"- {deduction}")

    # Save PDF
    output_path = BASE_DIR / "outputs" / "report.pdf"
    save_report(clues, deductions, str(output_path))
