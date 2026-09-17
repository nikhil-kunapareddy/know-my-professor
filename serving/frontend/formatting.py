"""Pure presentation logic for the chat UI.

Deliberately free of any Streamlit import: how a citation reads is ordinary
string formatting, and keeping it separate means it can be unit-tested without a
UI framework installed.
"""

from __future__ import annotations

from shared.schemas import Citation


def format_citation(citation: Citation) -> str:
    """One citation as a markdown line.

    A missing URL renders as plain text rather than an empty link — the metadata
    is genuinely absent for some chunks, and `[source]()` is a dead link that
    looks like a bug.
    """
    header = f"**[{citation.number}] {citation.professor_name or 'Unknown'}**"
    if citation.professor_title:
        header += f" — {citation.professor_title}"

    section = citation.section_type.replace("_", " ") or "profile"
    link = f"[source]({citation.url})" if citation.url else "no link"
    return f"{header}  \n{section} · {link} · score {citation.score:.2f}"


def sources_label(count: int) -> str:
    """Expander title for the citation list."""
    return f"{'Source' if count == 1 else 'Sources'} ({count})"
