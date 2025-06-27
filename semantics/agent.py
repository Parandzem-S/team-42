# agent.py
import re, json
import numpy as np
import pandas as pd
from pathlib import Path

# ── local LLM via transformers ─────────────────────────────────────────
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# ── retrieval + features ──────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent

# ── 0. load your pre-built answers & embeddings ───────────────────────
root    = Path(__file__).parent
ANS     = pd.read_pickle(root / "answers.pkl")              # built earlier
emb_mat = np.vstack(ANS["embedding"].to_numpy())            # shape (N, 384)

# reuse a small embedder for queries
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── 1. define semantic‐search tool ────────────────────────────────────
def search_answers(query: str, k: int = 5) -> str:
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    sims  = emb_mat @ q_vec
    top_k = sims.argsort()[-k:][::-1]
    rows  = [ANS.iloc[i].to_dict() for i in top_k]
    return json.dumps(rows, indent=2)[:4000]

# ── 2. define stats tool ──────────────────────────────────────────────
def answer_stats(ids_csv: str) -> str:
    ids  = [int(i) for i in re.findall(r"\d+", ids_csv)]
    cols = ["answer_coverage","word_cnt","cond_cnt","hedge_cnt",
            "modal_cnt","conf_score","num_cnt"]
    return ANS.loc[ids, cols].to_json()

# ── 3. define coverage‐summary tool ───────────────────────────────────
def coverage_summary(_="") -> str:
    feats   = ["word_cnt","cond_cnt","hedge_cnt",
               "modal_cnt","conf_score","num_cnt"]
    summary = ANS.groupby("answer_coverage")[feats].agg(["mean","std"]).round(2)
    return summary.to_json()

TOOLS = [
    Tool("search_answers",   search_answers,   "Find similar answers by semantic query."),
    Tool("answer_stats",     answer_stats,     "Get linguistic counters for row IDs."),
    Tool("coverage_summary", coverage_summary, "Get means/stds by answer_coverage."),
]

# ── 4. set up the local text‐generation pipeline ─────────────────────────
# - google/flan-t5-small is ~80M params, runs on CPU or GPU
# - if you have CUDA, add device=0 to pipeline(...)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    device=-1             # set to 0 if you installed GPU PyTorch
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"max_length": 512}
)

# ── 5. assemble the agent ───────────────────────────────────────────────
SYSTEM = SystemMessage(
    content=(
        "You are a forensic linguist analysing bank earnings-call answers. "
        "Use the tools for evidence and always cite row IDs."
    )
)

agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent="chat-zero-shot-react-description",
    system_message=SYSTEM,
    verbose=False
)

# ── 6. REPL ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("💬 Agent ready — type 'quit' to exit")
    while True:
        q = input("👤 ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        out = agent.invoke({"input": q})
        print("🤖", out["output"])
