"""Tiered document parsing and structure-aware chunking.

The pipeline for any document a source brings -- HTML, PDF, or an image::

    bytes --probe & route--> tier 0  layout   (DOM / PDF text layer; free, exact)
                             tier 1  OCR      (scanned pages; confidence per paragraph)
                             tier 2  vision   (figures; description + text + numbers)
          --> Document (typed blocks: heading, paragraph, list, table, figure)
          --> chunk_blocks (cut at headings, then block boundaries, then rows/items/sentences)
          --> preprocessing.sources.base.section_chunks (vector ids, labels, metadata)

Modules:

- ``ir``        the canonical Document / Block model, JSON round-trippable
- ``probe``     media-type sniffing and the per-page routing rule
- ``tiers/``    one module per tier, each importing its own dependency lazily
- ``parser``    ``DocumentParser``: runs the route, fills ``Document.review``
- ``chunking``  the structure-aware chunker
- ``config``    every threshold, with what it guards

Where it runs: a source's scraper parses at scrape time and stores
``Document.to_dict()`` in its GCS record; the source's ``to_chunks`` rebuilds it
with ``Document.from_dict()`` at ingest time and hands it to ``section_chunks``.
``ir`` and ``chunking`` are pure standard library, so ingest needs none of the
parsing dependencies.

Try it on a file::

    python -m preprocessing.documents path/to/file.pdf [--ocr tesseract] [--captioner gemini]

This package imports nothing from ``preprocessing.sources``; the dependency runs
one way, sources -> documents.
"""
