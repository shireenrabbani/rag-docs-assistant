# rag-docs-assistant

A retrieval-augmented Q&A assistant over data-platform documentation (Databricks / dbt / Azure docs), built with the same reliability mindset used for production data pipelines: validated ingestion, measured retrieval quality, and observability on every request.

> Status: 🚧 in progress — this README is filled in as each stage ships. Don't leave placeholder metrics in the final version; replace them with real numbers from `evals/results.json`.

## Why this project

Most RAG demos stop at "it answers questions." This one treats retrieval and generation as production data products — the same way I'd treat an ETL pipeline — with explicit accuracy, freshness, and reliability targets instead of vibes.

## Architecture

```
Docs (Databricks/dbt/Azure) 
  -> chunk (src/ingest.py) 
  -> embed (src/embed.py) 
  -> pgvector (src/store.py)
  -> query -> hybrid retrieve + rerank (src/retrieve.py) 
  -> grounded generation w/ citations (src/generate.py) 
  -> eval harness: retrieval precision + hallucination rate (src/eval.py)
```

[ Add a real diagram here once the pipeline is built — a simple boxes-and-arrows PNG or draw.io export is fine. ]

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
| Median latency (end-to-end, retrieval + generation) | 7.73s |
| Cost per query | $0.019 |
| Eval set size | 25 |
| Faithfulness / hallucination rate | TBD — needs ragas wired in |

**How this played out — frontmatter is a mixed signal, not pure noise.** The starting hypothesis was that YAML frontmatter just dilutes the embedding signal. Stripping it entirely (strategy 2) fixed 3 of the original 10 misses (`seeds.md`, `snapshots.md`, `data-tests.md`) but broke 2 that were previously hits — *"How do I define a source in dbt?"* and *"What is continuous integration in dbt and how does it work?"* — because those docs' `title:` fields almost exactly restated the question topic ("Add sources to your DAG", "Continuous integration"), and that title was actively helping vector similarity, not hurting it. So the real fix wasn't "remove the frontmatter," it was "keep the semantic content of the frontmatter, drop the YAML syntax around it": extracting `title` + `description` as one plain-text line (strategy 3) kept every fix from strategy 2 **and** avoided both regressions — 0 questions that used to hit now miss. That's what took hybrid retrieval from 0.60 to 0.76 precision@5, a real 27% relative improvement, not a rounding artifact.

Six questions still miss under strategy 3 (`exposures.md`, `environment-variables.md`, `groups.md`, `custom-databases.md`, `profiles.yml.md`, `using-threads.md`) — a good next investigation, not blocking further work on this project.

## Known limitations

- **Frontmatter handling: resolved.** See the Results section for the full 3-strategy comparison — `title`+`description` extraction (strategy 3) is what's live in `src/ingest.py` now, at 0.76 hybrid precision@5.
- **6 questions still miss retrieval** even under the current setup (`exposures.md`, `environment-variables.md`, `groups.md`, `custom-databases.md`, `profiles.yml.md`, `using-threads.md`) — not yet root-caused, a reasonable next investigation.
- **Faithfulness/hallucination rate not measured yet** — `src/eval.py` only checks retrieval hits and whether generation ran, not whether the generated answer is actually grounded in the retrieved context. Wiring in ragas is the next real step, not just a TODO comment.

## What I'd do differently at scale

*(Write this honestly after building — e.g., "swap pgvector for a managed index past N docs," "add caching for repeat queries," "add a feedback loop from user corrections back into the eval set." This section signals production judgment, not just that it runs.)*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, DATABASE_URL
```

## Usage

```bash
python -m src.ingest --source data/raw --out data/chunks.jsonl
python -m src.embed --in data/chunks.jsonl
python -m src.eval --questions evals/test_questions.jsonl
streamlit run app.py
```
