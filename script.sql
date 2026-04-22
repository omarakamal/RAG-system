CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    client_id TEXT,
    content TEXT,
    embedding VECTOR(384)  -- matches all-MiniLM-L6-v2
);


SELECT * FROM documents;


CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);