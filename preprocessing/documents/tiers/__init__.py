"""The parse tiers, one module each, cheapest first.

Imported lazily by ``DocumentParser``, never from ``preprocessing.documents``
itself: each tier pulls its own dependency (trafilatura, pdfplumber,
pytesseract, the Gemini SDK), and the ingest image -- which only rebuilds
stored documents and chunks them -- installs none of them.

- ``html``   tier 0  HTML DOM via trafilatura's XML output
- ``pdf``    tier 0  PDF text layer via pdfplumber
- ``ocr``    tier 1  ``OcrEngine`` registry (Tesseract)
- ``vision`` tier 2  ``Captioner`` registry (Gemini)
"""
