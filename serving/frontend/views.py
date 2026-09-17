"""Streamlit rendering.

Thin by design: the string formatting lives in formatting.py (no Streamlit
dependency), leaving only the actual widget calls here.
"""

from __future__ import annotations

import streamlit as st

from serving.frontend.formatting import format_citation, sources_label
from shared.schemas import Citation


def render_citations(citations: list[Citation]) -> None:
    """Show the sources the answer actually cited, collapsed by default."""
    if not citations:
        return
    with st.expander(sources_label(len(citations))):
        for citation in citations:
            st.markdown(format_citation(citation))


def render_error(message: str, request_id: str = "") -> None:
    """Show a failure as an error box, with the id needed to look it up."""
    st.error(message)
    if request_id:
        st.caption(f"Request ID: `{request_id}`")
