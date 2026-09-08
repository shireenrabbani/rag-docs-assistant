"""pgvector schema + read/write helpers.

Requires the pgvector extension enabled on the target Postgres database:
    CREATE EXTENSION IF NOT EXISTS vector;
"""

import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import DATABASE_URL, EMBEDDING_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIM})
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_schema():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()


def upsert_chunks(records: list[dict]):
    """records: [{chunk_id, source_id, source_path, text, embedding}, ...]"""
    with get_connection() as conn, conn.cursor() as cur:
        for r in records:
            cur.execute(
                """
                INSERT INTO chunks (chunk_id, source_id, source_path, text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                (r["chunk_id"], r["source_id"], r["source_path"], r["text"], r["embedding"]),
            )
        conn.commit()


def vector_search(query_embedding, top_k: int) -> list[dict]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, source_id, source_path, text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def keyword_search(query: str, top_k: int) -> list[dict]:
    """Postgres full-text search (not vector-based) — catches exact-term
    matches (error codes, config keys) that embedding similarity can miss.

    to_tsvector() runs at query time here with no index, which is fine at
    ~3K rows for a portfolio project. At real scale this needs a
    precomputed GIN index: `CREATE INDEX ... USING GIN (to_tsvector('english', text))`.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, source_id, source_path, text,
                   ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', %s)) AS rank
            FROM chunks
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
