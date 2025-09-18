from src.utils import extract_text_from_pdf, extract_clues_from_text

# Path to your PDF report
pdf_path = "outputs/report.pdf"

# Extract text from the PDF
text = extract_text_from_pdf(pdf_path)

# Extract clues from the text
clues = extract_clues_from_text(text)

print("Extracted clues:")
for clue in clues:
    print("-", clue)
