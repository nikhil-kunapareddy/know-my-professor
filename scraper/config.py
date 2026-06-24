"""Shared constants for the Khoury faculty scraper."""

from __future__ import annotations

import re
from pathlib import Path

BASE = "https://www.khoury.northeastern.edu"
LISTING = f"{BASE}/people/"
USER_AGENT = (
    "KhouryFacultyScraper/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30
OUTPUT_DIR = Path(__file__).parent / "data"

PROFILE_URL_RE = re.compile(rf"^{re.escape(BASE)}/people/[a-z0-9-]+/$")
