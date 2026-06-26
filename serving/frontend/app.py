"""Streamlit chat UI for Know My Professor. Talks to the /chat API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get(
    "KMP_API_URL", "https://kmp-api-309233821309.us-central1.run.app"
).rstrip("/")
REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="Know My Professor")
st.title("Know My Professor")
st.caption("Ask about Northeastern Khoury faculty.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            st.markdown(
                f"**[{c['number']}] {c['professor_name']}** — {c['professor_title']}  \n"
                f"{c['section_type']} · [profile]({c['url']}) · score {c['score']:.2f}"
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_citations(msg.get("citations", []))

if question := st.chat_input("Ask a question about Khoury faculty..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
            except requests.RequestException as e:
                answer = f"Error contacting API: {e}"
                citations = []

        st.markdown(answer)
        render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
