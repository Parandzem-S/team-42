import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from features import analyse_answer

# 0Load & keep answers only
df = pd.read_csv("hsbc_final_qa_analysis.csv")
answers = df[df["flag_question"] == False].reset_index(drop=True)

#  Linguistic features
stats = answers["Speaker text"].apply(analyse_answer).apply(pd.Series)
answers = pd.concat([answers, stats], axis=1)

# 2Embed on CPU (MiniLM = small & fast)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb_array = embedder.encode(
               answers["Speaker text"].tolist(),
               normalize_embeddings=True,     # so dot-product == cosine
               batch_size=64,
            )
answers["embedding"] = list(emb_array)        # store each vector in the row

# 3️⃣  Persist one file
answers.to_pickle("answers.pkl")
print(f"✓ Saved {len(answers):,} answers with embeddings")
