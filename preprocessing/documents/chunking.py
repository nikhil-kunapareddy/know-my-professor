"""Structure-aware chunking: cut along the document's own seams.

A chunk never spans two sections, and a block is only ever split when it cannot
fit in a chunk on its own. In order of preference:

1. **A heading starts a new chunk.** Every chunk records the heading path it
   sits under, so "Program Requirements > Electives" travels with the rows it
   titles instead of being left in an earlier chunk.
2. **A block moves whole.** A paragraph, list or table that does not fit in
   what is left of the current chunk starts the next one, if it fits there.
3. **Only an oversized block is split**, and then only at its own boundaries:
   a list between items, a table between rows -- with the header row repeated
   in every piece, so no piece is a bare run of cell values -- and a paragraph
   between sentences. Cutting inside a word is the last resort, for a single
   sentence longer than the whole budget.

With ``max_chars=None`` nothing is ever split, and a section renders exactly as
``preprocessing.sources.base.render_section`` renders it today: a paragraph as
its own text, a list as ``- item`` lines. That identity is what lets an existing
source move onto this chunker without re-embedding anything that fits, and
``tests/test_chunking.py`` pins it against the golden fixtures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .ir import Block, BlockKind

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


@dataclass(frozen=True)
class Piece:
    """One chunk's body, before an entity header and label are put on it."""

    #: Headings this piece sits under, outermost first.
    path: tuple[str, ...]
    body: str
    #: True for every piece of a section after its first.
    continued: bool = False
    #: Pages the piece's blocks came from, for paginated input.
    pages: tuple[int, ...] = ()


# --- rendering: the one place a block's text layout is decided -----------------


def _cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def table_head(block: Block) -> str:
    """A table's header as markdown, separator line included. Empty without one."""
    if not block.header_rows:
        return ""
    width = max(len(r) for r in block.rows)
    lines = [_row(r) for r in block.rows[: block.header_rows]]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(lines)


def table_body_rows(block: Block) -> list[str]:
    return [_row(r) for r in block.rows[block.header_rows :]]


def render_block(block: Block) -> str:
    """A whole block as chunk text.

    Tables render as markdown: it is the layout a language model reads most
    reliably as rows and columns, where tab- or space-separated cells blur.
    """
    if block.kind is BlockKind.LIST:
        return "\n".join(f"- {item}" for item in block.items)
    if block.kind is BlockKind.TABLE:
        return "\n".join(part for part in (table_head(block), *table_body_rows(block)) if part)
    if block.kind is BlockKind.FIGURE:
        return f"Figure: {block.text}"
    return block.text


# --- splitting an oversized block ------------------------------------------------


def _wrap(text: str, max_chars: int) -> list[str]:
    """Cut at whitespace into pieces of at most ``max_chars``; a longer word stands alone."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _sentences(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        parts.extend([sentence] if len(sentence) <= max_chars else _wrap(sentence, max_chars))
    return [p for p in parts if p.strip()]


@dataclass(frozen=True)
class _Atom:
    """The smallest unit packing moves: a whole block, or one part of a split one."""

    block: int
    text: str
    #: Joins this atom to the previous atom of the SAME block.
    sep: str = ""
    #: For a table row: the header to repeat above the rows in each piece.
    head: str | None = None
    page: int | None = None


def _atoms(index: int, block: Block, max_chars: int) -> list[_Atom]:
    """Split an oversized block at its own boundaries."""
    page = block.page
    if block.kind is BlockKind.LIST:
        atoms = []
        for item in block.items:
            for j, part in enumerate(_sentences(item, max_chars - 2)):
                atoms.append(_Atom(index, ("- " if j == 0 else "  ") + part, "\n", page=page))
        return atoms
    if block.kind is BlockKind.TABLE:
        head = table_head(block)
        rows = table_body_rows(block)
        if not rows:  # a header-only table: nothing to repeat it above
            return [_Atom(index, line, "\n", page=page) for line in head.split("\n")]
        budget = max(max_chars - len(head) - 1, 1)
        atoms = []
        for row in rows:
            for part in [row] if len(row) <= budget else _wrap(row, budget):
                atoms.append(_Atom(index, part, "\n", head=head, page=page))
        return atoms
    text = render_block(block)
    return [_Atom(index, part, " ", page=page) for part in _sentences(text, max_chars)]


# --- packing ---------------------------------------------------------------------


def _render(atoms: list[_Atom]) -> str:
    groups: list[list[_Atom]] = []
    for atom in atoms:
        if groups and groups[-1][-1].block == atom.block:
            groups[-1].append(atom)
        else:
            groups.append([atom])
    rendered = []
    for group in groups:
        text = group[0].text + "".join(a.sep + a.text for a in group[1:])
        head = group[0].head
        rendered.append(f"{head}\n{text}" if head else text)
    return "\n\n".join(rendered)


def _added_length(current: list[_Atom], atom: _Atom) -> int:
    """Characters ``atom`` adds when appended to ``current``."""
    if not current:
        sep = 0
    elif current[-1].block == atom.block:
        sep = len(atom.sep)
    else:
        sep = 2
    starts_table = atom.head and (not current or current[-1].block != atom.block)
    return sep + len(atom.text) + (len(atom.head) + 1 if starts_table else 0)


def _pack(blocks: list[Block], max_chars: int | None) -> list[list[_Atom]]:
    """Greedy packing of one section's blocks into pieces of at most ``max_chars``."""
    whole = [_Atom(i, render_block(b), page=b.page) for i, b in enumerate(blocks)]
    if max_chars is None:
        return [whole] if whole else []

    pieces: list[list[_Atom]] = []
    current: list[_Atom] = []
    length = 0

    def emit() -> None:
        nonlocal current, length
        if current:
            pieces.append(current)
        current, length = [], 0

    for i, block in enumerate(blocks):
        atom = whole[i]
        if length + _added_length(current, atom) <= max_chars:
            parts = [atom]
        elif len(atom.text) <= max_chars:
            emit()
            parts = [atom]
        else:
            parts = _atoms(i, block, max_chars)
        for part in parts:
            added = _added_length(current, part)
            if current and length + added > max_chars:
                emit()
                added = _added_length(current, part)
            current.append(part)
            length += added
    emit()
    return pieces


def chunk_blocks(blocks: Iterable[Block], max_chars: int | None = None) -> list[Piece]:
    """Pieces of ``blocks``: one section per heading path, each packed to the budget."""
    pieces: list[Piece] = []
    path: list[tuple[int, str]] = []
    section: list[Block] = []

    def flush() -> None:
        if not section:
            return
        names = tuple(text for _, text in path)
        for n, atoms in enumerate(_pack(section, max_chars)):
            pages = tuple(sorted({a.page for a in atoms if a.page is not None}))
            pieces.append(Piece(names, _render(atoms), continued=n > 0, pages=pages))
        section.clear()

    for block in blocks:
        if block.kind is BlockKind.HEADING:
            flush()
            level = block.level or 1
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, " ".join(block.text.split())))
        elif not block.is_empty():
            section.append(block)
    flush()
    return pieces
