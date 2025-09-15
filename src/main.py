from utils import read_clues
from reasoning_agent import analyze_clues
from pdf_generator import save_report

if __name__ == "__main__":
    print("\n🕵️ AI Detective Agent")
    print("=" * 22)

    # Read clues
    clues = read_clues("inputs/sample_case.txt")
    print("Clues received:")
    for clue in clues:
        print(f"- {clue}")

    # Analyze clues
    deductions = analyze_clues(clues)

    print("\nDeductions:")
    for deduction in deductions:
        print(f"- {deduction}")

    # Save PDF
    save_report(clues, deductions, "outputs/report.pdf")
