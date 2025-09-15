from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def save_report(clues, deductions, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "🕵️ AI Detective Report")
    y -= 30

    # Clues
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Clues:")
    y -= 20

    c.setFont("Helvetica", 12)
    for clue in clues:
        # Remove numbering if present
        clean_clue = clue.split(":", 1)[-1].strip()
        c.drawString(70, y, f"- {clean_clue}")
        y -= 20

    y -= 10

    # Deductions
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Deductions:")
    y -= 20

    c.setFont("Helvetica", 12)
    for deduction in deductions:
        clean_deduction = deduction.strip()  # remove unwanted newline/space
        c.drawString(70, y, f"- {clean_deduction}")
        y -= 20

    c.save()
    print(f"✅ PDF report saved to {output_path}")
