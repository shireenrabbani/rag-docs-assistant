"""Demo UI. Minimal on purpose — the eval numbers, not the UI, are what a
reviewer cares about. Run with: streamlit run app.py
"""

import streamlit as st

from src.generate import generate_answer
from src.retrieve import hybrid_retrieve, retrieve

RETRIEVERS = {
    "Hybrid (RRF + rerank) — recommended": hybrid_retrieve,
    "Baseline (vector-only)": retrieve,
}

st.set_page_config(page_title="Data Platform Docs Assistant")
st.title("Data Platform Docs Assistant")
st.caption("RAG over dbt docs — grounded answers with citations, built like a data product: measured, not vibes-based.")

with st.expander("Measured on a 25-question hand-labeled eval set (see README for full analysis)"):
    st.markdown(
        "- Retrieval precision@5: **0.76** (hybrid) vs. 0.40 (baseline vector-only)\n"
        "- Hallucination rate: **1.8%** (ragas faithfulness, gpt-4o-mini judge)\n"
        "- Median latency: 7.3s · Cost per query: $0.019"
    )

mode = st.radio("Retrieval mode", list(RETRIEVERS.keys()), horizontal=True)
query = st.text_input("Ask a question about dbt (e.g. \"How do I define a source in dbt?\")")

if query:
    retrieve_fn = RETRIEVERS[mode]
    with st.spinner("Retrieving..."):
        chunks = retrieve_fn(query, top_k=5)
    with st.spinner("Generating..."):
        result = generate_answer(query, chunks)

    st.markdown(result["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{result['latency_s']:.2f}s")
    col2.metric("Input tokens", result["input_tokens"])
    col3.metric("Output tokens", result["output_tokens"])

    with st.expander(f"Retrieved chunks ({mode})"):
        for c in chunks:
            score_label = "rerank score" if "rerank_score" in c else "similarity"
            score_value = c.get("rerank_score", c.get("similarity", 0))
            st.markdown(f"**{c['source_id']}** ({score_label}: {score_value:.3f})")
            st.text(c["text"][:500])
