
from transformers import AutoTokenizer, pipeline
import textstat
import re, numpy as np, spacy, pandas as pd
from huggingface_hub import login
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from features import analyse_answer
# import dotenv
tqdm.pandas()

# dotenv.load_dotenv(r'C:\Users\dea33\.env')

data = pd.read_csv('hsbc_final_qa_analysis.csv')
data.head()


answers = data[data["flag_question"] == False].reset_index(drop=True)
print(f"{len(answers):,} answer rows retained")


# setup for analysis starts here

nlp = spacy.load("en_core_web_sm")

COND = {"if","when","unless","provided","assuming"}
MODAL = {"may","might","could","should","would","can","will"}
HEDGE = {"approximately","around","potentially","likely","expect","intend"}
CERTAIN = {"definitely","certainly","confident","absolutely"}
UNCERT = {"uncertain","believe","think","hope","estimate"}

def analyse_answer(txt:str)->dict:
    doc = nlp(txt)
    toks = [t for t in doc if not t.is_punct]
    sents = list(doc.sents)

    lower = [t.lower_ for t in toks]
    nums = re.findall(r"\b\d[\d,\.%]*\b", txt)

    return {
        "word_cnt": len(toks),
        "sent_cnt": len(sents),
        "cond_cnt": sum(w in COND for w in lower),
        "modal_cnt": sum(w in MODAL for w in lower),
        "hedge_cnt": sum(w in HEDGE for w in lower),
        "certain_cnt": sum(w in CERTAIN for w in lower),
        "uncert_cnt": sum(w in UNCERT for w in lower),
        "num_cnt": len(nums),
        "num_ratio": len(nums)/max(len(toks),1),
        "fk_grade": textstat.flesch_kincaid_grade(txt),
        "conf_score": (1+sum(w in CERTAIN for w in lower)) / \
                      (1+sum(w in UNCERT for w in lower)),  # >1 = confident
    }


stats = answers["Speaker text"].progress_apply(analyse_answer).apply(pd.Series)

answers = pd.concat([answers, stats], axis=1)

# keep only the columns you still need
keep_cols = [
    "Year", "Quarter", "Speaker name", "Institution",
    "answer_coverage", "Speaker text",
    # + every column produced by analyse_answer
] + stats.columns.tolist()
answers = answers[keep_cols]


# model starts here
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

emb = embedder.encode(
        answers["Speaker text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
)
d = emb.shape[1]
index = faiss.IndexFlatIP(d)          # inner-product == cosine after normalisation
index.add(emb.astype("float32"))

answers.to_excel("answers_only.xlsx")
faiss.write_index(index, "answers_only.faiss")


# partial_df = data2[data2['answer_coverage'].str.lower() == 'partially answered'].copy()
# print(f"Found {len(partial_df)} partially answered questions")
a=1