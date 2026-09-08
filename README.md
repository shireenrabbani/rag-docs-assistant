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

*(Fill in after Day 4-5 — this section is what a reviewer reads first.)*

| Metric | Value |
|---|---|
| Retrieval precision@5 | TBD |
| Faithfulness / hallucination rate | TBD |
| Median latency (end-to-end) | TBD |
| Cost per query | TBD |
| Eval set size | TBD |

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
