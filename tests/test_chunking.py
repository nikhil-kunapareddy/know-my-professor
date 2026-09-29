"""Structure-aware chunking (preprocessing/documents/chunking.py) and section_chunks.

The first block of tests is the migration guarantee: a section that fits in one
chunk renders exactly as ``render_section`` does today, so a source can move
onto ``section_chunks`` without re-embedding anything. It is checked against
every chunk text in the golden fixtures, not against hand-picked strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.documents.chunking import Piece, chunk_blocks, render_block
from preprocessing.documents.config import DEFAULT_MAX_CHARS
from preprocessing.documents.ir import Block, BlockKind, Document
from preprocessing.sources.base import (
    Chunk,
    SectionSpec,
    Source,
    content_hash,
    part_vector_id,
    render_section,
    section_chunks,
)

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "chunk_golden.json").read_text())


def H(text, level=1):
    return Block(BlockKind.HEADING, text=text, level=level)


def P(text, page=None):
    return Block(BlockKind.PARAGRAPH, text=text, page=page)


def L(*items):
    return Block(BlockKind.LIST, items=items)


def T(*rows, header_rows=1):
    return Block(BlockKind.TABLE, rows=tuple(tuple(r) for r in rows), header_rows=header_rows)


# --- the migration guarantee ---------------------------------------------------


def _golden_chunks():
    for group in GOLDEN.values():
        for entry in group:
            yield from entry["chunks"]


def _split(text: str) -> tuple[str, str, str]:
    """``header\\nlabel:\\nbody`` -> (header, label, body)."""
    header, label_line, body = text.split("\n", 2)
    assert label_line.endswith(":")
    return header, label_line[:-1], body


@pytest.mark.parametrize("max_chars", [None, DEFAULT_MAX_CHARS * 100])
def test_a_section_that_fits_renders_exactly_as_render_section_does(max_chars):
    """Every golden chunk, rebuilt through section_chunks: same id, text and hash."""
    checked = 0
    for chunk in _golden_chunks():
        header, label, body = _split(chunk["text"])
        lines = body.split("\n")
        content = [line[2:] for line in lines] if all(line.startswith("- ") for line in lines) else body
        meta = {k: v for k, v in chunk["metadata"].items() if k != "content_hash"}

        [rebuilt] = section_chunks(chunk["vector_id"], header, label, content, meta, max_chars=max_chars)

        assert rebuilt.vector_id == chunk["vector_id"]
        assert rebuilt.text == chunk["text"]
        assert rebuilt.metadata == chunk["metadata"]
        checked += 1
    assert checked == sum(1 for _ in _golden_chunks()) >= 20  # every fixture chunk was walked


def test_section_chunks_matches_render_section_for_plain_values():
    for value in ["One line.", "Two\n\nparagraphs, kept as written.", ["a", "b"], [1, 2]]:
        [chunk] = section_chunks("x#bio", "Jane (Prof)", "Bio", value, {"section_type": "bio"})
        expected = render_section("Jane (Prof)", "Bio", value)
        assert chunk == Chunk("x#bio", expected, {"section_type": "bio", "content_hash": content_hash(expected)})


def test_empty_content_yields_no_chunks():
    assert section_chunks("x#bio", "h", "Bio", "", {}) == []
    assert section_chunks("x#bio", "h", "Bio", [], {}) == []
    assert section_chunks("x#bio", "h", "Bio", [H("Only a heading")], {}) == []


# --- ids and metadata of multi-part sections ------------------------------------


def test_part_one_keeps_the_section_id():
    assert part_vector_id("khoury-x#research_funding", 1) == "khoury-x#research_funding"
    assert part_vector_id("khoury-x#research_funding", 3) == "khoury-x#research_funding@3"


def test_an_oversized_list_splits_into_parts_with_continued_labels():
    items = [f"NSF award {i}: a long and detailed description of the funded work." for i in range(12)]
    chunks = section_chunks("x#research_funding", "Jane", "Funding", items, {"section_type": "f"}, max_chars=300)

    assert len(chunks) > 1
    assert [c.vector_id for c in chunks] == ["x#research_funding"] + [
        f"x#research_funding@{n}" for n in range(2, len(chunks) + 1)
    ]
    assert chunks[0].text.startswith("Jane\nFunding:\n- NSF award 0")
    assert all(c.text.startswith("Jane\nFunding (continued):\n- ") for c in chunks[1:])
    assert [c.metadata["part"] for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.metadata["content_hash"] == content_hash(c.text) for c in chunks)
    # every item survives, whole and in order
    bodies = "\n".join(c.text.split(":\n", 1)[1] for c in chunks)
    assert bodies == "\n".join(f"- {i}" for i in items)


def test_document_sections_carry_heading_path_and_pages():
    doc = Document(
        "x.pdf",
        "application/pdf",
        [H("Runbook"), P("Intro.", page=1), H("Failover", 2), P("Drain first.", page=2), P("Then promote.", page=3)],
    )
    first, second = section_chunks("doc#body", "Runbook", "Document", doc, {})

    assert first.text == "Runbook\nDocument > Runbook:\nIntro."
    assert first.metadata["heading_path"] == "Runbook"
    assert (first.metadata["page_start"], first.metadata["page_end"]) == (1, 1)
    assert second.text == "Runbook\nDocument > Runbook > Failover:\nDrain first.\n\nThen promote."
    assert (second.metadata["page_start"], second.metadata["page_end"]) == (2, 3)
    # Pinecone metadata lists may only hold strings: pages are two numbers
    assert "pages" not in second.metadata


# --- chunk_blocks: structure ------------------------------------------------------


def test_each_heading_starts_a_piece_and_paths_nest_and_pop():
    pieces = chunk_blocks([
        H("Guide"), P("a"),
        H("Setup", 2), P("b"),
        H("Linux", 3), P("c"),
        H("Usage", 2), P("d"),
        H("Appendix"), P("e"),
    ])
    assert [(p.path, p.body) for p in pieces] == [
        (("Guide",), "a"),
        (("Guide", "Setup"), "b"),
        (("Guide", "Setup", "Linux"), "c"),
        (("Guide", "Usage"), "d"),
        (("Appendix",), "e"),
    ]


def test_blocks_under_one_heading_share_a_piece_when_they_fit():
    [piece] = chunk_blocks([P("Intro."), L("x", "y"), T(("A", "B"), ("1", "2"))], max_chars=500)
    assert piece.body == "Intro.\n\n- x\n- y\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_a_block_that_does_not_fit_the_rest_moves_whole_to_the_next_piece():
    long_para = "Word " * 60  # 300 chars, fits a fresh piece but not what is left
    pieces = chunk_blocks([P("x" * 150), P(long_para.strip())], max_chars=320)
    assert [p.body for p in pieces] == ["x" * 150, long_para.strip()]
    assert pieces[1].continued and not pieces[0].continued


def test_an_oversized_table_splits_between_rows_repeating_its_header():
    rows = [("Course", "Title", "Hours")] + [(f"CS {1000 + i}", f"Course number {i}", "4") for i in range(30)]
    pieces = chunk_blocks([T(*rows)], max_chars=300)

    assert len(pieces) > 1
    head = "| Course | Title | Hours |\n| --- | --- | --- |"
    for p in pieces:
        assert p.body.startswith(head + "\n| CS ")
        assert len(p.body) <= 300
    body_rows = [line for p in pieces for line in p.body.split("\n")[2:]]
    assert body_rows == [f"| CS {1000 + i} | Course number {i} | 4 |" for i in range(30)]


def test_an_oversized_paragraph_splits_between_sentences():
    sentences = [f"Sentence number {i} says something useful." for i in range(20)]
    pieces = chunk_blocks([P(" ".join(sentences))], max_chars=200)
    assert len(pieces) > 1
    assert all(len(p.body) <= 200 for p in pieces)
    assert " ".join(p.body for p in pieces) == " ".join(sentences)
    assert all(p.body.endswith(".") for p in pieces)  # never cut mid-sentence here


def test_a_sentence_longer_than_the_budget_is_wrapped_at_words():
    word_soup = " ".join(f"token{i}" for i in range(100))  # no sentence breaks
    pieces = chunk_blocks([P(word_soup)], max_chars=120)
    assert all(len(p.body) <= 120 for p in pieces)
    assert " ".join(p.body for p in pieces).split() == word_soup.split()


def test_no_piece_exceeds_the_budget_on_a_mixed_document():
    blocks = [H("Doc")]
    for i in range(15):
        blocks += [
            H(f"Section {i}", 2),
            P(" ".join(f"Sentence {i}.{j} is here." for j in range(i * 3 + 1))),
            L(*(f"item {i}.{j} " * (j + 1) for j in range(i + 1))),
            T(("k", "v"), *((f"key{j}", f"value {j}") for j in range(i * 2 + 1))),
        ]
    pieces = chunk_blocks(blocks, max_chars=400)
    assert pieces and all(len(p.body) <= 400 for p in pieces)


def test_empty_blocks_are_skipped():
    pieces = chunk_blocks([P("   "), L(" "), T(("", ""), header_rows=0), P("kept")])
    assert pieces == [Piece(path=(), body="kept")]


def test_render_block_layouts():
    assert render_block(L("a", "b")) == "- a\n- b"
    assert render_block(T(("A", "B|C"), ("1", "2"))) == "| A | B\\|C |\n| --- | --- |\n| 1 | 2 |"
    assert render_block(T(("1", "2"), header_rows=0)) == "| 1 | 2 |"
    assert render_block(Block(BlockKind.FIGURE, text="A bar chart.")) == "Figure: A bar chart."


# --- registry guard ------------------------------------------------------------------


def test_registry_rejects_a_section_key_holding_the_part_separator():
    from preprocessing.sources.registry import SOURCES, _validate

    class Bad(Source):
        name, prefix = "bad", "bad/"
        sections = (SectionSpec("notes@v2", "Notes"),)
        namespace = SOURCES[0].namespace
        depends_on_entities = True

        def to_chunks(self, record):
            return []

    with pytest.raises(ValueError, match="notes@v2"):
        _validate((*SOURCES, Bad()))
