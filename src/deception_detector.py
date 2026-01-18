from typing import Dict, List, Any
import re

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    NLP = None

class DeceptionDetector:
    """
    Advanced feature to analyze witness statements for signs of deception
    using linguistic cues and sentiment patterns.
    """
    
    # Linguistic markers often associated with deceptive statements
    HEDGE_WORDS = {
        "maybe", "perhaps", "possibly", "likely", "unlikely",
        "i think", "i believe", "to the best of my knowledge",
        "as far as i know", "basically", "kinda", "sort of", "guess"
    }
    
    ABSOLUTE_TERMS = {
        "always", "never", "everyone", "nobody", "constantly",
        "absolutely", "totally", "impossible", "certainly"
    }
    
    NEGATIVE_EMOTION_WORDS = {
        "hate", "angry", "furious", "upset", "afraid", "scared", 
        "worried", "nervous", "anxious", "terrible", "bad", "awful"
    }

    DISTANCING_PRONOUNS = {"that", "those"} # e.g., "that woman" vs "she" or "Alice"

    def __init__(self):
        pass

    def analyze_statement(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single statement and returns a deception risk report.
        """
        score = 0.0
        details = []

        # 1. Check for hedge words (Uncertainty)
        hedges_found = [word for word in self.HEDGE_WORDS if word in text.lower()]
        if len(hedges_found) > 1: # One might be normal, multiple is sus
            score += 0.2
            details.append(f"Contains hedge words (uncertainty): {', '.join(hedges_found)}")

        # 2. Check for absolute terms (Over-denial)
        absolutes_found = [word for word in self.ABSOLUTE_TERMS if word in text.lower()]
        if len(absolutes_found) > 1:
            score += 0.15
            details.append(f"Frequent use of absolute terms (potential over-compensation): {', '.join(absolutes_found)}")

        # 3. Text Length Analysis (Too short or too detailed?)
        # A very short denial can be suspicious, or a very long winding story.
        # This is a weak signal, so small weight.
        word_count = len(text.split())
        if word_count < 5:
            score += 0.1
            details.append("Statement is unusually brief")
        
        # 4. Emotional Intensity
        negatives_found = [word for word in self.NEGATIVE_EMOTION_WORDS if word in text.lower()]
        if len(negatives_found) > 2:
            score += 0.1
            details.append("High negative emotional content")

        # 5. Distancing Language (using NLP if available for better accuracy)
        if NLP:
            doc = NLP(text)
            # Look for demonstrative determiners + person (e.g., "that man")
            for token in doc:
                if token.lower_ in self.DISTANCING_PRONOUNS and token.dep_ == "det":
                    head = token.head
                    if head.pos_ == "NOUN" and (head.ent_type_ == "PERSON" or head.lemma_ in ["man", "woman", "guy", "girl", "person"]):
                        score += 0.25
                        details.append(f"Distancing language detected: '{token.text} {head.text}'")
        
        # Normalize score (0 to 1)
        score = min(score, 1.0)
        
        risk_level = "Low"
        if score > 0.4:
            risk_level = "Medium"
        if score > 0.7:
            risk_level = "High"

        return {
            "statement": text,
            "deception_score": round(score, 2),
            "risk_level": risk_level,
            "analysis_details": details
        }

    def analyze_batch(self, statements: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze_statement(stmt) for stmt in statements]
