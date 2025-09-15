def read_clues(file_path: str) -> list[str]:
    """
    Reads clues from a text file.
    Each line is treated as one clue.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            clues = [line.strip() for line in f if line.strip()]
        return clues
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return []
