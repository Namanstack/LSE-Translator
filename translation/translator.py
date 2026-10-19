"""
LSE Translation Layer

Decouples neural gesture classification output (LSE Sign Labels) from
target language text generation, dictionary lookups, and downstream context expansions.
"""

from typing import Dict, Optional


# Canonical 10-Sign LSE Reference Vocabulary
CANONICAL_VOCABULARY = [
    "HOLA",
    "GRACIAS",
    "ADIOS",
    "POR_FAVOR",
    "SI",
    "NO",
    "AGUA",
    "AYUDA",
    "PERDON",
    "BIEN",
]

# Canonical LSE Vocabulary to Spanish & English Text Dictionary
DEFAULT_TRANSLATION_DICT: Dict[str, Dict[str, str]] = {
    "HOLA": {
        "es": "Hola",
        "en": "Hello",
        "description": "Greeting gesture"
    },
    "GRACIAS": {
        "es": "Gracias",
        "en": "Thank you",
        "description": "Expression of gratitude"
    },
    "ADIOS": {
        "es": "Adiós",
        "en": "Goodbye",
        "description": "Farewell gesture"
    },
    "POR_FAVOR": {
        "es": "Por favor",
        "en": "Please",
        "description": "Polite request"
    },
    "SI": {
        "es": "Sí",
        "en": "Yes",
        "description": "Affirmation gesture"
    },
    "NO": {
        "es": "No",
        "en": "No",
        "description": "Negation gesture"
    },
    "AGUA": {
        "es": "Agua",
        "en": "Water",
        "description": "Request or sign for water"
    },
    "AYUDA": {
        "es": "Ayuda",
        "en": "Help",
        "description": "Request for assistance"
    },
    "PERDON": {
        "es": "Perdón",
        "en": "Sorry",
        "description": "Apology gesture"
    },
    "BIEN": {
        "es": "Bien",
        "en": "Good / Well",
        "description": "Positive affirmation gesture"
    },
    "BUENOS_DIAS": {
        "es": "Buenos días",
        "en": "Good morning",
        "description": "Morning greeting"
    }
}


class LSETranslator:
    """
    Translates predicted LSE sign labels into human-readable target sentences/words.
    """

    def __init__(self, dictionary: Optional[Dict[str, Dict[str, str]]] = None):
        self.dictionary = dictionary or DEFAULT_TRANSLATION_DICT

    def translate_label(self, label: str, target_lang: str = "es") -> str:
        """
        Translates a predicted sign label to target language text.

        Args:
            label (str): Predicted sign label (e.g., 'HOLA').
            target_lang (str): 'es' (Spanish) or 'en' (English).

        Returns:
            str: Translated text or fallback string.
        """
        clean_label = label.strip().upper()
        if clean_label in self.dictionary:
            return self.dictionary[clean_label].get(target_lang, clean_label)
        return clean_label

    def get_supported_labels(self) -> list:
        """Returns list of all recognized dictionary vocabulary items."""
        return list(self.dictionary.keys())
