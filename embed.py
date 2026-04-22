from sentence_transformers import SentenceTransformer
import os
import numpy as np
import requests
from db import get_conn

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text, normalize_embeddings=True).tolist()


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
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -----------------------------
# STORE EMBEDDING (DB)
# -----------------------------
def store_embedding(client_id, text, embedding):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO documents (client_id, content, embedding)
        VALUES (%s, %s, %s)
        """,
        (client_id, text, embedding)
    )

    conn.commit()
    cur.close()
    conn.close()


# -----------------------------
# CLEAR CLIENT DATA (IMPORTANT)
# -----------------------------
def clear_client_data(client_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM documents WHERE client_id = %s",
        (client_id,)
    )

    conn.commit()
    cur.close()
    conn.close()


# -----------------------------
# INGESTION (DB ONLY)
# -----------------------------
def ingest(client_id="default"):
    docs = load_documents()

    # optional: clear old data before re-ingesting
    clear_client_data(client_id)

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            embedding = embed_text(chunk)
            store_embedding(client_id, chunk, embedding)

    print(f"Ingestion complete for client: {client_id}")


def to_vector_str(vec):
    return "[" + ",".join(map(str, vec)) + "]"

# -----------------------------
# SEARCH (pgvector)
# -----------------------------
def search(query, client_id="default", top_k=3):
    query_embedding = embed_text(query)
    query_vec = to_vector_str(query_embedding)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT content, embedding <=> %s::vector AS distance
        FROM documents
        WHERE client_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec, client_id, query_vec, top_k)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "text": r[0],
            "score": float(1 - r[1])
        }
        for r in results
    ]
# -----------------------------
# CONTEXT
# -----------------------------
def build_context(results):
    return "\n\n".join([r["text"] for r in results])


# -----------------------------
# PROMPT
# -----------------------------
def build_prompt(context, query):
    return f"""
You are a helpful customer support assistant.

Rules:
- Only use the context below
- If not found, say "I don't know"
- Be concise and helpful

Context:
{context}

Question:
{query}

Answer:
""".strip()


# -----------------------------
# LLM CALL (Ollama)
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
# MAIN PIPELINE
# -----------------------------
def ask_question(query, client_id="default"):
    results = search(query, client_id)

    context = build_context(results)

    prompt = build_prompt(context, query)

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": results
    }


# -----------------------------
# LOCAL TEST
# -----------------------------
if __name__ == "__main__":
    client_id = "test_client"

    # 1. ingest data
    ingest(client_id)

    # 2. query
    query = "my order did not get here?"

    response = ask_question(query, client_id)

    print("\nANSWER:\n", response["answer"])

    print("\nSOURCES:")
    for r in response["sources"]:
        print("-", round(r["score"], 3))