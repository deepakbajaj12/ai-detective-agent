from src.utils import extract_entities_and_relations

clues = [
    "Alice entered the room at 9 PM.",
    "A note was found on the table.",
    "Bob was seen near the door."
]
results = extract_entities_and_relations(clues)
for item in results:
    print("Clue:", item['clue'])
    print("Entities:", item['entities'])
    print("Relations:", item['relations'])
    print()