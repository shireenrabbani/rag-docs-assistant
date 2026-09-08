# rag-docs-assistant

A retrieval-augmented Q&A assistant over data-platform documentation (Databricks / dbt / Azure docs), built with the same reliability mindset used for production data pipelines: validated ingestion, measured retrieval quality, and observability on every request.

> Status: core pipeline, hybrid retrieval, eval harness (retrieval precision + ragas faithfulness), and demo UI are all built and measured — see Results below. Remaining open items are tracked in Known limitations, not hidden.

## Why this project

Most RAG demos stop at "it answers questions." This one treats retrieval and generation as production data products — the same way I'd treat an ETL pipeline — with explicit accuracy, freshness, and reliability targets instead of vibes.

## Architecture

![Architecture diagram: dbt docs are chunked and embedded into pgvector offline; at query time, vector search and Postgres keyword search are fused with reciprocal rank fusion, reranked with a cross-encoder, and passed to Claude for grounded generation with citations; an eval harness measures retrieval precision and faithfulness throughout.](docs/architecture.svg)

## Stack

- **Ingestion/chunking:** Python (PySpark for larger corpora)
- **Embeddings:** OpenAI `text-embedding-3-small` (swap-in Voyage/Cohere noted in `src/embed.py`)
- **Vector store:** pgvector (Postgres)
- **Retrieval:** hybrid (keyword + vector) with cross-encoder reranking
- **Generation:** Claude API (Anthropic), grounded prompt w/ source citations
- **Eval:** Ragas (retrieval precision/recall, faithfulness/hallucination rate) + custom latency & cost-per-query tracking
- **Demo:** Streamlit

## Results

25 hand-labeled question/expected-source pairs, spanning `build/`, `deploy/`, `local/`, and `running-a-dbt-project/` docs.

Three frontmatter-handling strategies were tested against the same 25-question eval set, because the first fix wasn't a clean win and the second attempt was:

| Strategy | Baseline precision@5 | Hybrid precision@5 |
|---|---|---|
| 1. Raw frontmatter left in chunk text | 0.44 | 0.60 |
| 2. Frontmatter stripped entirely | 0.36 (-0.08) | 0.64 (+0.04) |
| 3. **`title` + `description` extracted, prepended as one clean line** | 0.40 | **0.76 (+0.16)** |

| Metric | Value (hybrid mode, strategy 3) |
|---|---|
| Median latency (end-to-end, retrieval + generation) | 7.29s |
| Cost per query | $0.019 |
| Eval set size | 25 |
| Mean faithfulness (ragas, gpt-4o-mini judge) | 0.982 |
| Hallucination rate | **1.8%** |

Faithfulness measures whether the *generated answer* is actually grounded in whatever context was retrieved — it's a different question from whether retrieval found the *right* doc. A wrong-but-retrieved doc can still produce a faithful (grounded-in-the-wrong-context, so a bad but not hallucinated) answer — which is exactly why 1.8% hallucination coexists with 0.76 (not 1.0) retrieval precision: the system prompt's "cite sources, say so if you don't know" instructions are doing their job even when retrieval misses. Judged by gpt-4o-mini rather than Claude (which generated the answers) to avoid the model grading its own homework.

**How this played out — frontmatter is a mixed signal, not pure noise.** The starting hypothesis was that YAML frontmatter just dilutes the embedding signal. Stripping it entirely (strategy 2) fixed 3 of the original 10 misses (`seeds.md`, `snapshots.md`, `data-tests.md`) but broke 2 that were previously hits — *"How do I define a source in dbt?"* and *"What is continuous integration in dbt and how does it work?"* — because those docs' `title:` fields almost exactly restated the question topic ("Add sources to your DAG", "Continuous integration"), and that title was actively helping vector similarity, not hurting it. So the real fix wasn't "remove the frontmatter," it was "keep the semantic content of the frontmatter, drop the YAML syntax around it": extracting `title` + `description` as one plain-text line (strategy 3) kept every fix from strategy 2 **and** avoided both regressions — 0 questions that used to hit now miss. That's what took hybrid retrieval from 0.60 to 0.76 precision@5, a real 27% relative improvement, not a rounding artifact.

Six questions still miss under strategy 3 (`exposures.md`, `environment-variables.md`, `groups.md`, `custom-databases.md`, `profiles.yml.md`, `using-threads.md`) — a good next investigation, not blocking further work on this project.

## Known limitations

- **Frontmatter handling: resolved.** See the Results section for the full 3-strategy comparison — `title`+`description` extraction (strategy 3) is what's live in `src/ingest.py` now, at 0.76 hybrid precision@5.
- **6 questions still miss retrieval** even under the current setup (`exposures.md`, `environment-variables.md`, `groups.md`, `custom-databases.md`, `profiles.yml.md`, `using-threads.md`) — not yet root-caused, a reasonable next investigation.
- **Faithfulness/hallucination rate: resolved.** ragas `Faithfulness` (gpt-4o-mini judge) is wired into `src/eval.py` — 98.2% mean faithfulness / 1.8% hallucination rate on the current eval set. Not yet tested: whether hallucination rate is meaningfully different on the retrieval *misses* specifically (where the model has the wrong context and a real test of whether it stays honest) vs. the hits — the current number is averaged across both.

## What I'd do differently at scale

- **pgvector's `ivfflat` index (`lists = 100`) was picked arbitrarily for ~3K rows**, not tuned. `lists` should scale with row count (roughly `sqrt(n)` is the common starting heuristic), and past a few million rows I'd look at a managed vector index (e.g. Pinecone, or Postgres `pgvector`'s `hnsw` index type) instead of re-tuning `ivfflat` by hand.
- **`keyword_search()` computes `to_tsvector(text)` at query time with no index** — fine at 3K rows, but the first thing to fix before this corpus grows: `CREATE INDEX ... USING GIN (to_tsvector('english', text))`, noted in `store.py` but not built.
- **No caching.** Repeated or near-duplicate questions re-embed, re-retrieve, and re-generate from scratch every time. A simple exact-match cache on `(query, retrieval_mode)` would cut cost and latency on real traffic with any query repetition.
- **The eval set is static and self-authored.** 25 hand-labeled questions is enough to prove the pipeline and catch regressions, but it doesn't reflect what real users actually ask. At scale I'd add a feedback loop — log real queries, sample and label a slice weekly, and grow the eval set from production traffic instead of my own guesses.
- **RRF's `k=60` and `TOP_K_RETRIEVE=20` / `TOP_K_RERANK=5` are defaults, not tuned values.** Worth a small grid search against the eval set once it's bigger than 25 questions — right now that would just be overfitting to noise.
- **Cost tracking in `eval.py` uses a hardcoded per-token price** (see the `TODO` there) instead of a maintained pricing table — fine for a portfolio project, not fine if this fed a real budget dashboard.
- **The 6 remaining retrieval misses** (`exposures.md`, `environment-variables.md`, `groups.md`, `custom-databases.md`, `profiles.yml.md`, `using-threads.md`) haven't been root-caused individually — the frontmatter investigation was systematic, this hasn't been yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, DATABASE_URL
```

## Usage

```bash
python -m src.ingest --source data/raw/docs --out data/chunks.jsonl
python -m src.embed --in data/chunks.jsonl
python -m src.eval --questions evals/test_questions.jsonl
streamlit run app.py
```

The demo (`app.py`) lets you toggle between hybrid and baseline retrieval live, so the Results table above isn't just a claim — you can watch retrieval quality change in real time on the same question.
