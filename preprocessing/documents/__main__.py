"""Parse one local file and show what each tier made of it.

    python -m preprocessing.documents FILE [--ocr tesseract] [--captioner gemini]
                                           [--max-chars 1600] [--json]

Tiers 1 and 2 are off unless asked for, so the default run is free and offline.
``--captioner gemini`` reads GEMINI_API_KEY; ``--ocr tesseract`` needs the
tesseract binary. ``--json`` prints the stored form of the Document instead.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

from .chunking import chunk_blocks, render_block
from .config import DEFAULT_MAX_CHARS
from .ir import BlockKind
from .parser import DocumentParser
from .tiers.ocr import OcrUnavailable, build_ocr
from .tiers.vision import CaptionUnavailable, build_captioner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--ocr", default="none", help="OCR engine: tesseract | none (default)")
    parser.add_argument("--captioner", default="none", help="figure captioner: gemini | none (default)")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="chunk budget")
    parser.add_argument("--json", action="store_true", help="print the stored Document JSON")
    args = parser.parse_args()

    try:
        doc_parser = DocumentParser(ocr=build_ocr(args.ocr), captioner=build_captioner(args.captioner))
    except (OcrUnavailable, CaptionUnavailable, ValueError) as e:
        sys.exit(f"error: {e}")

    declared, _ = mimetypes.guess_type(args.file.name)
    doc = doc_parser.parse(args.file.read_bytes(), media_type=declared, source=str(args.file))

    if args.json:
        print(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False))
        return

    print(f"{doc.source}  [{doc.media_type}]  blocks by tier: {doc.tier_counts() or 'none'}")
    for r in doc.routes:
        print(f"  page {r.page}: tier {int(r.tier)} ({r.tier.name.lower()}), {r.figures} figure(s) captioned")

    print("\n--- blocks ---")
    for b in doc.blocks:
        where = f"p.{b.page}" if b.page is not None else ""
        conf = f" conf {b.confidence:.2f}" if b.confidence is not None else ""
        tag = f"H{b.level}" if b.kind is BlockKind.HEADING else b.kind.value.upper()
        first = render_block(b).split("\n", 1)[0] if b.kind is not BlockKind.HEADING else b.text
        print(f"  {tag:<9} t{int(b.tier)}{conf} {where:>5}  {first[:90]}")

    if doc.review:
        print("\n--- held for review ---")
        for item in doc.review:
            where = f"p.{item.page} " if item.page is not None else ""
            print(f"  {where}{item.reason}" + (f": {item.text[:70]}" if item.text else ""))

    pieces = chunk_blocks(doc.blocks, args.max_chars)
    print(f"\n--- {len(pieces)} chunk(s) at max_chars={args.max_chars} ---")
    for i, piece in enumerate(pieces, 1):
        path = " > ".join(piece.path) or "(no heading)"
        print(f"\n[{i}] {path}{' (continued)' if piece.continued else ''}  {len(piece.body)} chars")
        print(piece.body)


if __name__ == "__main__":
    main()
