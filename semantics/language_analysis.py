import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('hsbc_final_qa_analysis.csv')

# Correct column filtering based on actual column names
df = df[(df['flag_question'] == False)]

df["text_clean"] = df["Speaker text"].astype(str).str.lower()

# Define semantic/linguistic patterns
patterns = {
    "modal_verbs": r"\b(can|could|may|might|shall|should|will|would|must|)\b",
    "conditional_phrases": r"\b(if|assuming that|provided that|in case|unless|considering)\b",
    "confidence_terms": r"\b(confident|strong|will|definitely|certainly|positive)\b",
    "uncertainty_terms": r"\b(maybe|possibly|potentially|unclear|likely|unlikely|probably|chance|possibly|possible)\b",
    "hedging_terms": r"\b(somewhat|sort of|kind of|relatively|broadly|generally|approximately|around)\b",
    "risk_terms": r"\b(risk|volatility|uncertainty|exposure|default|write[- ]?off)\b"
}

# Function to count matches
def count_matches(text, pattern):
    return len(re.findall(pattern, text))

# Apply each pattern
for label, pattern in patterns.items():
    df[label + "_count"] = df["text_clean"].apply(lambda x: count_matches(x, pattern))

# Add answer length
df["answer_length"] = df["text_clean"].apply(lambda x: len(x.split()))

# Group 1: by risk_or_distress
grouped_risk = df.groupby("risk_or_distress").agg({
    "modal_verbs_count": "mean",
    "conditional_phrases_count": "mean",
    "confidence_terms_count": "mean",
    "uncertainty_terms_count": "mean",
    "hedging_terms_count": "mean",
    "risk_terms_count": "mean",
    "answer_length": "mean"
}).reset_index()

# Group 2: by answer_coverage
grouped_coverage = df.groupby("answer_coverage").agg({
    "modal_verbs_count": "mean",
    "conditional_phrases_count": "mean",
    "confidence_terms_count": "mean",
    "uncertainty_terms_count": "mean",
    "hedging_terms_count": "mean",
    "risk_terms_count": "mean",
    "answer_length": "mean"
}).reset_index()

# Display both result tables
grouped_risk.to_csv("Semantic Analysis by Risk-Distress.csv")
grouped_coverage.to_csv("Semantic Analysis by Answer Coverage.csv")


a=1