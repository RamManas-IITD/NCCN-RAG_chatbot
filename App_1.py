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

# ----------------------------------------

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

# -------- TEXT UTILITIES --------

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

            chunk = " ".join(words[start:end])

            chunks.append(chunk)

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# -------- EMBEDDINGS --------

def embed_texts(texts):

    vectors = []

    for t in texts:

        payload = {
            "inputText": t
        }

        resp = bedrock.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            contentType="application/json",
            body=json.dumps(payload)
        )

        data = json.loads(resp["body"].read())

        vectors.append(data["embedding"])

    return np.array(vectors).astype("float32")


# -------- BUILD / LOAD INDEX --------

def build_and_save_index():

    st.info("Building embeddings and FAISS index...")

    text = load_text()

    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    st.success("Index built successfully!")

    return index, chunks


def load_index():

    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# -------- RETRIEVAL --------

def retrieve(query, index, chunks):

    q_emb = embed_texts([query])

    distances, idxs = index.search(q_emb, TOP_K)

    retrieved_chunks = [chunks[i] for i in idxs[0]]

    retrieved_distances = distances[0]

    return retrieved_chunks, retrieved_distances


# -------- CONFIDENCE SCORING --------

def compute_confidence(distances):

    # Use only top 3 retrieved chunks
    distances = np.array(distances[:3])

    # Normalize distances
    max_dist = np.max(distances)
    min_dist = np.min(distances)

    normalized = 1 - (
        (distances - min_dist)
        / (max_dist - min_dist + 1e-8)
    )

    confidence = float(np.mean(normalized))

    return confidence


def confidence_label(score):

    if score >= 0.75:
        return "High"

    elif score >= 0.50:
        return "Medium"

    else:
        return "Low"


# -------- CLAUDE OPUS --------

def call_opus(context, question):

    prompt = f"""
You are an expert oncology guideline assistant.

Answer the question STRICTLY using the CONTEXT below.

If the answer is not present, say:
"Not specified in the NCCN guideline text provided."

For staging/treatment questions:

1. First determine AJCC stage strictly from T, N, and M.
2. Use ONLY recommendations for that exact stage.
3. Do NOT mix stages.
4. Verify all tumor-size thresholds explicitly.
5. If ambiguous, say "Not specified".

CONTEXT:
{context}

QUESTION:
{question}
"""

    payload = {

        "anthropic_version": "bedrock-2023-05-31",

        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
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


# ================= STREAMLIT APP =================

st.set_page_config(
    page_title="NCCN RAG (Claude Opus)",
    layout="wide"
)

st.title("🧬 NCCN Variant-Aware RAG (Claude Opus)")


# -------- LOAD INDEX --------

if (
    os.path.exists(FAISS_INDEX_FILE)
    and os.path.exists(CHUNKS_FILE)
):

    index, chunks = load_index()

    st.success("Loaded existing FAISS index")

else:

    index, chunks = build_and_save_index()


# -------- USER INPUT --------

question = st.text_input(
    "Ask an NCCN guideline question:",
    placeholder="e.g., Are EGFR exon 20 insertions treated uniformly in NCCN?"
)


# -------- MAIN RAG PIPELINE --------

if question:

    with st.spinner("Retrieving & reasoning..."):

        # Retrieve chunks
        retrieved_chunks, distances = retrieve(
            question,
            index,
            chunks
        )

        # Compute confidence
        confidence = compute_confidence(distances)

        label = confidence_label(confidence)

        # Build context
        context = "\n\n".join(retrieved_chunks)

        # Generate answer
        answer = call_opus(context, question)

    # -------- DISPLAY ANSWER --------

    st.subheader("Answer")

    st.write(answer)

    # -------- CONFIDENCE DISPLAY --------

    st.markdown(
        f"## Confidence: {label} ({confidence:.2f})"
    )

    # -------- OPTIONAL WARNING --------

    if confidence < 0.40:

        st.warning(
            "Low retrieval confidence. "
            "Relevant NCCN context may be limited."
        )

    # -------- SHOW RETRIEVED CONTEXT --------

    with st.expander("Retrieved Guideline Context"):

        for i, chunk in enumerate(retrieved_chunks, 1):

            st.markdown(f"### Chunk {i}")

            st.write(chunk)

            st.markdown("---")