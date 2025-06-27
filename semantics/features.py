# features.py ── minimal counters, no extra libs
import re, spacy
nlp = spacy.load("en_core_web_sm")

COND   = {"if", "when", "unless", "provided", "assuming"}
HEDGE  = {"approximately", "likely", "potentially", "expect", "intend"}
MODAL  = {"may", "might", "could", "should", "would", "can", "will"}
CERT   = {"definitely", "certainly", "confident", "absolutely"}
UNCERT = {"uncertain", "believe", "think", "hope", "estimate"}

def analyse_answer(txt: str) -> dict:
    doc   = nlp(txt)
    lower = [t.lower_ for t in doc if not t.is_punct]

    cert_cnt   = sum(tok in CERT   for tok in lower)
    uncert_cnt = sum(tok in UNCERT for tok in lower)

    return {
        "word_cnt"  : len(lower),
        "cond_cnt"  : sum(tok in COND  for tok in lower),
        "hedge_cnt" : sum(tok in HEDGE for tok in lower),
        "modal_cnt" : sum(tok in MODAL for tok in lower),
        "conf_score": (1 + cert_cnt) / (1 + uncert_cnt),
        "num_cnt"   : len(re.findall(r"\b\d[\d,.%]*\b", txt)),
    }
