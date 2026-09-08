"""Embed chunks and write them to pgvector.

TODO (Day 2, your judgment call): batch requests instead of one-per-chunk,
add retry/backoff on rate limits, and log $ cost (tokens * price/1K) to
stdout so you have a real cost-per-ingest number for the README.
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY
from src.store import init_schema, upsert_chunks

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_batch(texts: list[str]) -> list[list[float]]:
    # TODO: chunk `texts` into batches of ~100 and call this per-batch instead
    # of assuming the caller already sized it correctly.
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    init_schema()

    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    print(f"Embedding {len(records)} chunks...")

    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        embeddings = embed_batch([r["text"] for r in batch])
        for r, emb in zip(batch, embeddings):
            r["embedding"] = emb
        upsert_chunks(batch)
        print(f"  wrote {i + len(batch)}/{len(records)}")

    print("Done.")


if __name__ == "__main__":
    main()
