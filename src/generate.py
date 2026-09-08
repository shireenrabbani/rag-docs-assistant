"""Retrieved chunks -> grounded answer with citations.

Day 3-4 task. The prompt template below is a deliberately plain starting
point — TODO tighten the grounding instructions once you see hallucinated
or unsupported answers in src/eval.py output. Keep a before/after example
of a prompt fix in the README; "I found a hallucination, traced it to a
weak grounding instruction, and fixed it" is a stronger interview story
than "it worked."
"""

import time

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, GENERATION_MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You answer questions using ONLY the provided context chunks.
Cite the source_id of every chunk you rely on, inline, like [source_id].
If the context does not contain enough information to answer, say so explicitly —
do not guess or use outside knowledge."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source_id']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(query: str, chunks: list[dict]) -> dict:
    start = time.perf_counter()
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, chunks)}],
    )
    latency_s = time.perf_counter() - start

    # Claude's response can include a "thinking" block before the text block,
    # so don't assume content[0] is the answer — find the text block explicitly.
    answer = next(b.text for b in response.content if b.type == "text")

    return {
        "answer": answer,
        "latency_s": latency_s,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "sources": [c["source_id"] for c in chunks],
    }
