"""Streamlit chat UI for Know My Professor.

Wiring only: the transport lives in api_client.py and the rendering in views.py,
so both are testable without running Streamlit.
"""

from __future__ import annotations

import os

import streamlit as st

from serving.frontend.api_client import ChatClient, ChatError
from serving.frontend.views import render_citations, render_error

API_URL = os.environ.get(
    "KMP_API_URL", "https://kmp-api-309233821309.us-central1.run.app"
)


@st.cache_resource
def get_client() -> ChatClient:
    """One client (and one connection pool) for the whole session."""
    return ChatClient(base_url=API_URL)


st.set_page_config(page_title="Know My Professor")
st.title("Know My Professor")
st.caption("Ask about Northeastern Khoury faculty.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            render_error(msg["content"], msg.get("request_id", ""))
        else:
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
                response = get_client().ask(question)
            except ChatError as e:
                render_error(e.message, e.request_id)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": e.message,
                    "error": True,
                    "request_id": e.request_id,
                    "citations": [],
                })
            else:
                st.markdown(response.answer)
                render_citations(response.citations)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "citations": response.citations,
                })
