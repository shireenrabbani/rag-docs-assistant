"""Day 5 demo UI. Minimal on purpose — the eval numbers, not the UI, are
what a reviewer cares about. Run with: streamlit run app.py
"""

import streamlit as st

from src.generate import generate_answer
from src.retrieve import retrieve

st.set_page_config(page_title="Data Platform Docs Assistant")
st.title("Data Platform Docs Assistant")
st.caption("RAG over Databricks/dbt/Azure docs — grounded answers with citations")

query = st.text_input("Ask a question")

if query:
    with st.spinner("Retrieving..."):
        chunks = retrieve(query)
    with st.spinner("Generating..."):
        result = generate_answer(query, chunks)

    st.markdown(result["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{result['latency_s']:.2f}s")
    col2.metric("Input tokens", result["input_tokens"])
    col3.metric("Output tokens", result["output_tokens"])

    with st.expander("Retrieved chunks"):
        for c in chunks:
            st.markdown(f"**{c['source_id']}** (similarity: {c['similarity']:.3f})")
            st.text(c["text"][:500])
