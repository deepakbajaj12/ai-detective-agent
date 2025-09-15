# src/reasoning_agent.py

def analyze_clues(clues):
    """
    Simple reasoning function for the AI Detective Agent.
    
    Args:
        clues (list of str): List of clues from the input file.
        
    Returns:
        list of str: List of deductions based on the clues.
    """
    deductions = []

    # Combine all clues into a single string for simple keyword checking
    clues_text = " ".join(clues).lower()

    # Basic deduction logic
    if "unlocked" in clues_text:
        deductions.append("🔎 Deduction: The suspect might have entered without force.")
    if "note" in clues_text:
        deductions.append("📝 Deduction: The note could be an important lead.")
    if "no fingerprints" in clues_text:
        deductions.append("🧤 Deduction: The suspect may have worn gloves.")

    # Return the deductions as a list
    return deductions

