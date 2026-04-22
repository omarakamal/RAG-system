from fastapi import FastAPI
from pydantic import BaseModel
from embed import (
    load_vectors,
    ask_question,
    ingest,
    VECTOR_DB,
    load_documents
)

app = FastAPI()


# -----------------------------
# MODELS
# -----------------------------
class ChatRequest(BaseModel):
    query: str


class IngestRequest(BaseModel):
    # optional later: client_id
    pass


# -----------------------------
# STARTUP (ONLY LOAD)
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
    """
    Rebuilds vector DB from current documents
    """
    ingest()

    return {
        "message": "Ingestion complete",
        "num_chunks": len(VECTOR_DB)
    }