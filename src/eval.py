"""Retrieval + generation eval harness.

Day 4-5 task — this is the differentiator for this project, not an
afterthought. Populate evals/test_questions.jsonl with ~20-30 real
question/expected-source pairs from your own corpus (hand-label them
yourself; that's normal for a v1 eval set), then run:

    python -m src.eval --questions evals/test_questions.jsonl

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
from src.retrieve import retrieve


def retrieval_hit(expected_source_id: str, retrieved: list[dict]) -> bool:
    return any(c["source_id"] == expected_source_id for c in retrieved)


def run_eval(questions_path: Path) -> dict:
    cases = [json.loads(line) for line in questions_path.open(encoding="utf-8")]

    hits, latencies, costs = [], [], []
    results = []

    for case in cases:
        retrieved = retrieve(case["question"])
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
    args = parser.parse_args()

    output = run_eval(args.questions)
    args.out.write_text(json.dumps(output, indent=2))

    print(json.dumps(output["summary"], indent=2))
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
