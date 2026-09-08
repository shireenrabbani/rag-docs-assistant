"""Retrieval + generation eval harness.

This is the differentiator for this project, not an afterthought.
Populate evals/test_questions.jsonl with ~20-30 real question/expected-source
pairs from your own corpus (hand-label them yourself; that's normal for a v1
eval set), then run:

    python -m src.eval --questions evals/test_questions.jsonl

This does two passes:
  1. Retrieval-only comparison: baseline (vector-only) vs. hybrid
     (keyword-fused + reranked), no generation calls — cheap, and this is
     the number that answers "did the Day 3 work actually help." Report
     both numbers and the delta in the README, not just the winner.
  2. Full eval (retrieval + generation) using whichever retrieval mode you
     pick with --mode, producing the latency/cost/answer numbers for
     evals/results.json.

TODO: wire in ragas (faithfulness, answer_relevancy, context_precision)
once you have >=20 labeled examples — below is a simpler hand-rolled
version so you can start with zero setup and swap to ragas once you
understand what it's measuring.
"""

import argparse
import json
import statistics
from pathlib import Path

from src.generate import generate_answer
from src.retrieve import hybrid_retrieve, retrieve

RETRIEVERS = {"baseline": retrieve, "hybrid": hybrid_retrieve}


def retrieval_hit(expected_source_id: str, retrieved: list[dict]) -> bool:
    return any(c["source_id"] == expected_source_id for c in retrieved)


def run_retrieval_comparison(cases: list[dict], top_k: int = 5) -> dict:
    """Cheap pass: precision@top_k for every registered retriever, no
    generation calls. This is what proves (or disproves) that hybrid +
    rerank was worth building."""
    comparison = {}
    for name, retrieve_fn in RETRIEVERS.items():
        hits = [
            retrieval_hit(case["expected_source_id"], retrieve_fn(case["question"], top_k=top_k))
            for case in cases
        ]
        comparison[name] = sum(hits) / len(hits) if hits else 0
    return comparison


def run_full_eval(cases: list[dict], retrieve_fn, top_k: int = 5) -> dict:
    hits, latencies, costs = [], [], []
    results = []

    for case in cases:
        retrieved = retrieve_fn(case["question"], top_k=top_k)
        hit = retrieval_hit(case["expected_source_id"], retrieved)
        result = generate_answer(case["question"], retrieved)

        hits.append(hit)
        latencies.append(result["latency_s"])
        # TODO: replace with real per-token pricing for GENERATION_MODEL + EMBEDDING_MODEL
        costs.append(result["input_tokens"] * 0.000003 + result["output_tokens"] * 0.000015)

        results.append({
            "question": case["question"],
            "expected_source_id": case["expected_source_id"],
            "retrieval_hit": hit,
            "answer": result["answer"],
            "latency_s": result["latency_s"],
        })

    summary = {
        "n": len(cases),
        "retrieval_precision_at_5": sum(hits) / len(hits) if hits else 0,
        "median_latency_s": statistics.median(latencies) if latencies else 0,
        "avg_cost_per_query_usd": statistics.mean(costs) if costs else 0,
    }
    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out", default=Path("evals/results.json"), type=Path)
    parser.add_argument("--mode", choices=RETRIEVERS.keys(), default="hybrid",
                         help="Retriever to use for the full (retrieval + generation) eval pass.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.questions.open(encoding="utf-8")]

    print(f"Comparing retrievers on {len(cases)} questions (retrieval only, no generation calls)...")
    comparison = run_retrieval_comparison(cases, top_k=args.top_k)
    for name, precision in comparison.items():
        print(f"  {name}: precision@{args.top_k} = {precision:.2f}")
    delta = comparison.get("hybrid", 0) - comparison.get("baseline", 0)
    print(f"  delta (hybrid - baseline) = {delta:+.2f}")

    print(f"\nRunning full eval (retrieval + generation) with mode='{args.mode}'...")
    output = run_full_eval(cases, RETRIEVERS[args.mode], top_k=args.top_k)
    output["retrieval_comparison"] = comparison

    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary"], indent=2))
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
