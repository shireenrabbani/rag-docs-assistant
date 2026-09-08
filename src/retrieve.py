"""Query -> retrieved chunks.

`retrieve()` is the vector-only baseline (Day 2). `add_keyword_search()`
and `rerank()` are the Day 3 additions. `hybrid_retrieve()` chains all
three into the full pipeline. Keep calling the plain `retrieve()` from
src/eval.py too — the point is to measure precision@5 with vs. without
these additions and report the delta in the README, not just to have
the fancier version.
"""

from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, TOP_K_RERANK, TOP_K_RETRIEVE
from src.store import keyword_search, vector_search

client = OpenAI(api_key=OPENAI_API_KEY)

# RRF constant. 60 is the value from the original Cormack et al. paper and
# the de-facto default (Elasticsearch, Weaviate use it too) — it just
# dampens the influence of rank 1 vs. rank 2 so one list can't dominate
# purely by having its top hit be rank 1. Not tuned for this corpus.
RRF_K = 60

_cross_encoder = None


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    query_embedding = embed_query(query)
    return vector_search(query_embedding, top_k)


def add_keyword_search(query: str, candidates: list[dict]) -> list[dict]:
    """Fuse the vector `candidates` with an independent Postgres full-text
    search using Reciprocal Rank Fusion. Independent (not just re-scoring
    the vector list) because a chunk can be a strong keyword match — an
    exact config key or error code — without ranking anywhere in the
    vector top-K, and RRF still surfaces it.

    RRF score per chunk = sum over each ranking it appears in of
    1 / (RRF_K + rank). Chunks in both lists accumulate both terms;
    chunks in only one list still score, just lower. This avoids having
    to normalize cosine similarity and ts_rank onto a shared scale,
    which is the usual trap in naive hybrid-search implementations.
    """
    keyword_hits = keyword_search(query, top_k=len(candidates))

    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for rank, chunk in enumerate(candidates, start=1):
        scores[chunk["chunk_id"]] = scores.get(chunk["chunk_id"], 0.0) + 1 / (RRF_K + rank)
        chunks_by_id[chunk["chunk_id"]] = chunk

    for rank, chunk in enumerate(keyword_hits, start=1):
        scores[chunk["chunk_id"]] = scores.get(chunk["chunk_id"], 0.0) + 1 / (RRF_K + rank)
        chunks_by_id.setdefault(chunk["chunk_id"], chunk)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [{**chunks_by_id[cid], "rrf_score": scores[cid]} for cid in ranked_ids]


def _get_cross_encoder() -> CrossEncoder:
    # Loaded lazily (not at module import) since ingest/embed don't need
    # it, and the first call downloads ~80MB of model weights.
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """Score each candidate against the query with a cross-encoder and
    return the top_k. A cross-encoder reads (query, chunk) together and
    is much more accurate than comparing separately-computed embeddings
    (a "bi-encoder", which is what vector_search uses) — but it's too
    slow to run over the whole corpus, which is why it only runs here,
    over the small candidate set the first-stage retrieval already
    narrowed down.
    """
    if not candidates:
        return []
    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [{**chunk, "rerank_score": float(score)} for chunk, score in ranked[:top_k]]


def hybrid_retrieve(query: str, top_k: int = TOP_K_RERANK) -> list[dict]:
    """Full pipeline: vector retrieve -> RRF fuse with keyword search -> cross-encoder rerank."""
    vector_candidates = retrieve(query, top_k=TOP_K_RETRIEVE)
    fused = add_keyword_search(query, vector_candidates)
    return rerank(query, fused, top_k=top_k)
