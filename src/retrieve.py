"""Query -> retrieved chunks.

Baseline (vector-only) is implemented below and works end to end on Day 2.
Day 3 additions, in order of interview-relevance:
  1. Hybrid search: TODO add_keyword_search() combining BM25/ILIKE with
     vector similarity (e.g. reciprocal rank fusion) — pure vector search
     misses exact-match terms like error codes or config keys.
  2. Reranking: TODO rerank() using a cross-encoder (e.g.
     sentence-transformers 'cross-encoder/ms-marco-MiniLM-L-6-v2') to
     reorder the top TOP_K_RETRIEVE down to TOP_K_RERANK before generation.
Measure precision@5 with and without each addition in src/eval.py and
report the delta — that comparison is the actual engineering content here.
"""

from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, TOP_K_RETRIEVE
from src.store import vector_search

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    query_embedding = embed_query(query)
    return vector_search(query_embedding, top_k)


def add_keyword_search(query: str, candidates: list[dict]) -> list[dict]:
    """TODO (Day 3): fold in a keyword/BM25 signal, e.g. via Postgres
    ts_rank or a simple ILIKE term-overlap score, and fuse the two
    rankings (reciprocal rank fusion is the simplest to defend in an
    interview)."""
    raise NotImplementedError


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """TODO (Day 3): score each candidate with a cross-encoder against
    the query and return the top_k re-sorted by that score."""
    raise NotImplementedError
