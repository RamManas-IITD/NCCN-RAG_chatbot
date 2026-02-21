import streamlit as st
import boto3
import json
import faiss
import numpy as np
import pickle
import os

# ---------------- CONFIG ----------------
AWS_REGION = "us-east-1"

TEXT_FILE = "nccn_output.txt"
FAISS_INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
GENERATION_MODEL_ID = "global.anthropic.claude-opus-4-5-20251101-v1:0"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 10
# --------------------------------------


bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# -------- Text utilities --------

def load_text():
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text):
    pages = text.split("=== PAGE")
    chunks = []

    for page in pages:
        if not page.strip():
            continue

        words = page.split()
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunks.append(" ".join(words[start:end]))
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# -------- Embedding / Index --------

def embed_texts(texts):
    vectors = []

    for t in texts:
        payload = {"inputText": t}
        resp = bedrock.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            contentType="application/json",
            body=json.dumps(payload)
        )
        data = json.loads(resp["body"].read())
        vectors.append(data["embedding"])

    return np.array(vectors).astype("float32")


def build_and_save_index():
    st.info("Building embeddings and FAISS index (one-time)...")

    text = load_text()
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    st.success("Index built and saved!")
    return index, chunks


def load_index():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# -------- Retrieval --------

def retrieve(query, index, chunks):
    q_emb = embed_texts([query])
    _, idxs = index.search(q_emb, TOP_K)
    return [chunks[i] for i in idxs[0]]


# -------- Claude Opus --------

def call_opus(context, question):
    prompt = f"""
You are an expert oncology guideline assistant.

Answer the question strictly using the CONTEXT below.
If the answer is not present, say:
"Not specified in the NCCN guideline text provided."
For questions where the below is relavant
First determine AJCC stage based strictly on tumor size, N, and M.
Once the stage is determined:
- Use ONLY treatment recommendations explicitly stated for that exact stage.
- Do NOT include recommendations for higher or lower stages.
- If a treatment applies only above a size threshold, verify the threshold explicitly.
If the guideline text is ambiguous, state "Not specified".

CONTEXT:
{context}

QUESTION:
{question}
"""

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    resp = bedrock.invoke_model(
        modelId=GENERATION_MODEL_ID,
        contentType="application/json",
        body=json.dumps(payload)
    )

    data = json.loads(resp["body"].read())
    return data["content"][0]["text"]


# ================= APP =================

st.set_page_config(page_title="NCCN RAG (Claude Opus)", layout="wide")
st.title("🧬 NCCN Variant-Aware RAG (Claude Opus)")

# Load or build index
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    index, chunks = load_index()
    st.success("Loaded existing FAISS index")
else:
    index, chunks = build_and_save_index()

# Question input
question = st.text_input(
    "Ask an NCCN guideline question (variant-level supported):",
    placeholder="e.g., Are EGFR exon 20 insertions treated uniformly in NCCN?"
)

if question:
    with st.spinner("Retrieving & reasoning..."):
        retrieved = retrieve(question, index, chunks)
        context = "\n\n".join(retrieved)
        answer = call_opus(context, question)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Guideline Context"):
        for i, c in enumerate(retrieved, 1):
            st.markdown(f"**Chunk {i}:**\n\n{c}")

