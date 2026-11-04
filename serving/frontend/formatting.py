"""Pure presentation logic for the chat UI.

Deliberately free of any Streamlit import: how a citation reads is ordinary
string formatting, and keeping it separate means it can be unit-tested without a
UI framework installed.
"""

from __future__ import annotations

import json

from shared.schemas import Citation

# The public address. A one-page Firebase Hosting site frames this app there,
# because Firebase cannot proxy the websocket Streamlit runs on
# (deploy/hosting/README.md).
CANONICAL_URL = "https://knowmyprof.web.app"


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


def redirect_script(canonical_url: str = CANONICAL_URL) -> str:
    """JavaScript that sends a direct visit to the Cloud Run URL to the public one.

    The Cloud Run URL cannot be made private: the frame on the public site loads
    it from the visitor's own browser. So it stays reachable and redirects
    instead. Two guards, both checked in the browser:

    - ``window.top === window.self`` is false inside the frame, so the framed
      copy never redirects (comparing the two is allowed across origins).
    - ``.run.app`` keeps ``streamlit run`` on localhost from bouncing a
      developer to production.
    """
    target = json.dumps(canonical_url)
    return (
        "<script>"
        "if (window.top === window.self"
        " && window.location.hostname.endsWith('.run.app')) {"
        f" window.location.replace({target});"
        " }"
        "</script>"
    )
