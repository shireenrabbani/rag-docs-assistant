"""Load raw docs and split into overlapping token-bounded chunks.

Day 1-2 task. Fixed-size token chunking is the starting point below —
once retrieval eval numbers are in (src/eval.py), come back and compare
against a semantic/heading-aware chunker. Write down the precision delta
in the README "What I'd do differently" section either way: that
comparison is itself a talking point in interviews.
"""

import argparse
import json
from pathlib import Path

import tiktoken

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

ENCODING = tiktoken.get_encoding("cl100k_base")


def load_documents(source_dir: Path) -> list[dict]:
    """Read every .md/.txt file under source_dir into {id, path, text} records."""
    docs = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        docs.append({
            "id": str(path.relative_to(source_dir)),
            "path": str(path),
            "text": path.read_text(encoding="utf-8", errors="ignore"),
        })
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Fixed-size token chunking with overlap. Returns decoded text chunks."""
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunks.append(ENCODING.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def build_chunk_records(docs: list[dict]) -> list[dict]:
    records = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            records.append({
                "chunk_id": f"{doc['id']}::{i}",
                "source_id": doc["id"],
                "source_path": doc["path"],
                "text": chunk,
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Directory of raw .md/.txt docs")
    parser.add_argument("--out", required=True, type=Path, help="Output .jsonl of chunk records")
    args = parser.parse_args()

    docs = load_documents(args.source)
    print(f"Loaded {len(docs)} documents from {args.source}")

    records = build_chunk_records(docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} chunks to {args.out}")


if __name__ == "__main__":
    main()
