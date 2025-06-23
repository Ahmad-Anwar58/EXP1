import json
import pandas as pd
from difflib import get_close_matches

class Chatbot:
    def __init__(self, faq_path):
        # Load FAQs from JSON
        with open(faq_path, 'r', encoding='utf-8') as f:
            self.faq_data = json.load(f)

    def get_response(self, question, context_data=None):
        question_lower = question.lower()
        questions = [q["question"].lower() for q in self.faq_data]

        # Try exact match
        if question_lower in questions:
            idx = questions.index(question_lower)
            return self.faq_data[idx]["answer"]

        # Try close match using fuzzy matching
        matches = get_close_matches(question_lower, questions, n=1, cutoff=0.6)
        if matches:
            idx = questions.index(matches[0])
            return self.faq_data[idx]["answer"]

        # Contextual logic from sensor data if no FAQ match found
        if context_data is not None:
            if "yield" in question_lower:
                return f"Predicted crop yield is {context_data.get('Crop Yield Prediction (kg)', 'not available')} kg."
            if "irrigation" in question_lower:
                return f"Irrigation efficiency is {context_data.get('Irrigation Efficiency (%)', 'not available')}%."
            if "fertilizer" in question_lower:
                return f"Fertilizer usage is {context_data.get('Fertilizer Usage (kg)', 'not available')} kg."
            if "pesticide" in question_lower:
                return f"Pesticide usage is {context_data.get('Pesticide Usage (kg)', 'not available')} kg."
            if "soil health" in question_lower:
                return f"Soil Health Index is {context_data.get('Soil Health Index (0-100)', 'not available')}."

        return "Sorry, I don't have an answer for that yet. Please try asking in a different way."
