from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from embed import ingest, ask_question

app = FastAPI(title="RAG AI Backend")


# -----------------------------
# REQUEST MODELS
# -----------------------------
class ChatRequest(BaseModel):
    client_id: str
    query: str


class IngestRequest(BaseModel):
    client_id: str


# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query:
        return {"error": "No query provided"}

    response = ask_question(req.query, req.client_id)

    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }


# -----------------------------
# INGEST ENDPOINT
# -----------------------------
@app.post("/ingest")
def ingest_data(req: IngestRequest):
    ingest(req.client_id)

    return {
        "message": f"Ingestion complete for client: {req.client_id}"
    }


# -----------------------------
# HEALTH CHECK (optional but useful)
# -----------------------------
@app.get("/")
def root():
    return {"status": "API is running"}