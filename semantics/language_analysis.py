import pandas as pd
import re

# Load data
df = pd.read_csv('hsbc_final_qa_analysis.csv')

# Filter out rows flagged as questions
df = df[df['flag_question'] == False]

# Clean and normalize text
df["text_clean"] = df["Speaker text"].astype(str).str.lower()

# Define expanded semantic/linguistic patterns
patterns = {
    "modal_verbs": r"\b(can|could|may|might|shall|should|would|ought to)\b",
    "conditional_phrases": r"\b(if|assuming that|provided that|in case|unless|considering|as long as|even if|in the event|suppose|supposing)\b",
    "confidence_terms": r"\b(confident|strong|will|definitely|certainly|positive|clear|firm|secure|assured|undoubtedly|without a doubt)\b",
    "uncertainty_terms": r"\b(maybe|possibly|potentially|unclear|likely|unlikely|probably|chance|possible|presumably|apparently|suggests|seems)\b",
    "hedging_terms": r"\b(somewhat|sort of|kind of|relatively|broadly|generally|approximately|around|roughly|typically|tend to|appears to|expect|expecting)\b",
    "risk_terms": r"\b(risk|volatility|uncertainty|exposure|default|write[- ]?off|impairment|fragile|shock|headwind|downside)\b"
}

# Count matches normalized per 100 words
def count_per_100_words(text, pattern):
    words = len(text.split())
    matches = len(re.findall(pattern, text))
    return (matches / words * 100) if words > 0 else 0

# Apply each pattern
for label, pattern in patterns.items():
    df[label + "_per_100w"] = df["text_clean"].apply(lambda x: count_per_100_words(x, pattern))

# Add raw length (optional, for reference)
df["answer_length"] = df["text_clean"].apply(lambda x: len(x.split()))

# Group 1: by risk_or_distress
grouped_risk = df.groupby("risk_or_distress").agg({
    "modal_verbs_per_100w": "mean",
    "conditional_phrases_per_100w": "mean",
    "confidence_terms_per_100w": "mean",
    "uncertainty_terms_per_100w": "mean",
    "hedging_terms_per_100w": "mean",
    "risk_terms_per_100w": "mean",
    "answer_length": "mean"
}).reset_index()

# Group 2: by answer_coverage
grouped_coverage = df.groupby("answer_coverage").agg({
    "modal_verbs_per_100w": "mean",
    "conditional_phrases_per_100w": "mean",
    "confidence_terms_per_100w": "mean",
    "uncertainty_terms_per_100w": "mean",
    "hedging_terms_per_100w": "mean",
    "risk_terms_per_100w": "mean",
    "answer_length": "mean"
}).reset_index()

# Save both tables
grouped_risk.to_csv("Semantic_Analysis_by_Risk_Distress.csv", index=False)
grouped_coverage.to_csv("Semantic_Analysis_by_Answer_Coverage.csv", index=False)
