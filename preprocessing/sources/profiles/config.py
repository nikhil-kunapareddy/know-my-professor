"""Configuration owned by the profiles source.

Lives here rather than in ``shared/config.py`` because nothing outside this
package needs it -- ``shared`` is reserved for constants that BOTH the serving
and data sides must agree on.
"""

from __future__ import annotations

import re
from pathlib import Path

KHOURY_BASE = "https://www.khoury.northeastern.edu"
KHOURY_LISTING = f"{KHOURY_BASE}/people/"
PROFILE_URL_RE = re.compile(rf"^{re.escape(KHOURY_BASE)}/people/[a-z0-9-]+/$")

SCRAPER_USER_AGENT = (
    "KhouryFacultyScraper/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
SCRAPER_REQUEST_DELAY_SECONDS = 1.0
SCRAPER_REQUEST_TIMEOUT_SECONDS = 30

# Local-only output dir (gitignored); GCS is the source of truth in production.
LOCAL_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data"
