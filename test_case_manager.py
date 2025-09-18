from src.case_manager import add_case, list_cases, add_clue, get_clues, update_case_status

# Add a new case
def main():
    add_case("Sample Case 1")
    add_case("Sample Case 2", status="in-progress")

    # List all cases
    cases = list_cases()
    print("Cases:")
    for case in cases:
        print(case)

    # Add clues to the first case
    case_id = cases[0][0]
    add_clue(case_id, "The door was unlocked.")
    add_clue(case_id, "A note was found on the table.")

    # Get clues for the first case
    clues = get_clues(case_id)
    print(f"Clues for case {case_id}:", clues)

    # Update case status
    update_case_status(case_id, "solved")
    print(f"Updated status for case {case_id} to 'solved'.")

if __name__ == "__main__":
    main()
