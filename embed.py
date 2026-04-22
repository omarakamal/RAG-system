from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import requests

# -----------------------------
# MODEL (OPTIMIZED: normalize embeddings)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents(folder="data"):
    docs = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append({
                    "source": file,
                    "text": f.read()
                })

    return docs


# -----------------------------
# BETTER CHUNKING (sentence-aware-ish)
# -----------------------------
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()

    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks


# -----------------------------
# VECTOR DB
# -----------------------------
VECTOR_DB = []


# -----------------------------
# SAVE / LOAD
# -----------------------------
def save_vectors(filename="vectors.json"):
    with open(filename, "w") as f:
        json.dump(VECTOR_DB, f)


def load_vectors(filename="vectors.json"):
    global VECTOR_DB

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                VECTOR_DB = json.load(f)
            print(f"Loaded {len(VECTOR_DB)} vectors from disk")
        except json.JSONDecodeError:
            VECTOR_DB = []


# -----------------------------
# INGESTION (OPTIMIZED)
# -----------------------------
def ingest():
    docs = load_documents()

    global VECTOR_DB
    VECTOR_DB = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            VECTOR_DB.append({
                "text": chunk,
                "source": doc["source"],
                "embedding": embed_text(chunk)
            })

    print(f"Loaded {len(VECTOR_DB)} chunks into vector DB")

    save_vectors()


# -----------------------------
# FAST COSINE (NO NUMPY LOOP COST)
# -----------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b)  # embeddings already normalized → dot product is enough


# -----------------------------
# SEARCH (OPTIMIZED LOOP)
# -----------------------------
def search(query, top_k=3):
    query_embedding = embed_text(query)

    results = []

    for item in VECTOR_DB:
        score = cosine_similarity(query_embedding, item["embedding"])

        results.append({
            "score": float(score),
            "text": item["text"],
            "source": item["source"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# -----------------------------
# CONTEXT
# -----------------------------
def build_context(results):
    return "\n\n".join([r["text"] for r in results])


# -----------------------------
# PROMPT (FIXED + CLEAN)
# -----------------------------
def build_prompt(context, query):
    return f"""
You are a helpful assistant for a meal plan service.

Rules:
- Only use the context below
- If not found, say "I don't know"
- Be concise

Context:
{context}

Question:
{query}

Answer:
""".strip()


# -----------------------------
# REAL LLM CALL (Ollama)
# -----------------------------
def generate_answer(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# -----------------------------
# MAIN PIPELINE (FIXED)
# -----------------------------
def ask_question(query):
    results = search(query)

    context = build_context(results)

    prompt = build_prompt(context, query)

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": results
    }


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    load_vectors()

    if not VECTOR_DB:
        ingest()

    query = "my order did not get here?"

    response = ask_question(query)

    print("\nANSWER:\n", response["answer"])

    print("\nSOURCES:")
    for r in response["sources"]:
        print("-", round(r["score"], 3), r["source"])



def init(question):
    if not VECTOR_DB:
        ingest()


    response = ask_question(question)

    print("\nANSWER:\n", response["answer"])

    print("\nSOURCES:")
    for r in response["sources"]:
        print("-", round(r["score"], 3), r["source"])