"""Load raw docs and split into overlapping token-bounded chunks.

Fixed-size token chunking is the starting point below — once retrieval
eval numbers are in (src/eval.py), consider comparing against a
semantic/heading-aware chunker. Write down the precision delta in the
README "What I'd do differently" section either way: that comparison
is itself a talking point in interviews.

FRONTMATTER HANDLING (see README "Known limitations" / "Results" miss
analysis for the full before/after numbers):
  1. Left in raw            -> baseline 0.44, hybrid 0.60
  2. Stripped entirely      -> baseline 0.36, hybrid 0.64
     (helped short docs with weak body content, hurt docs whose title
     restated the topic and was doing real semantic work)
  3. title+description prepended as one clean line, rest of frontmatter
     discarded (current) -> intent is to keep the useful signal from (1)
     without the raw YAML syntax noise. Re-run src/eval.py after
     re-ingesting + re-embedding to see whether this actually beats both.
"""

import argparse
import json
import re
from pathlib import Path

import tiktoken
import yaml

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

ENCODING = tiktoken.get_encoding("cl100k_base")

# Docusaurus/Jekyll-style frontmatter: a '---' delimited block at the very
# start of the file. re.DOTALL so '.' spans the newlines inside the block,
# and the capture group isolates just the YAML body (without the '---'
# fences) for parsing.
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def clean_frontmatter(text: str) -> str:
    """Replace raw YAML frontmatter with a single plain-text summary line
    built from title + description, so the semantic signal survives
    without the '---\\ntitle: "..."\\n...' syntax diluting the embedding."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return text

    body = text[match.end():]
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        # Malformed frontmatter in a handful of docs shouldn't crash the
        # whole ingest run — fall back to just dropping it.
        return body

    title = str(meta.get("title") or "").strip()
    description = str(meta.get("description") or "").strip()
    summary = ". ".join(p for p in (title, description) if p)
    return f"{summary}\n\n{body}" if summary else body


def load_documents(source_dir: Path) -> list[dict]:
    """Read every .md/.mdx/.txt file under source_dir into {id, path, text} records."""
    docs = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".mdx", ".txt"}:
            continue
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "id": str(path.relative_to(source_dir)),
            "path": str(path),
            "text": clean_frontmatter(raw_text),
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
