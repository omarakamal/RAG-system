from fastapi import FastAPI
from pydantic import BaseModel
from embed import load_vectors, ask_question, ingest, VECTOR_DB

app = FastAPI()

# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    query: str

# -----------------------------
# STARTUP (LOAD ONLY)
# -----------------------------
load_vectors()

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query:
        return {"error": "No query provided"}

    response = ask_question(req.query)

    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }

# -----------------------------
# INGEST ENDPOINT
# -----------------------------
@app.post("/ingest")
def ingest_data():
    ingest()

    return {
        "message": "Ingestion complete",
        "num_chunks": len(VECTOR_DB)
    }