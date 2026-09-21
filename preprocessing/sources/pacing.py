"""Rate limiting shared by the scrapers.

Lives here rather than inside one source because every scraper that fans out
across workers needs the same guarantee, and two copies would drift.
"""

from __future__ import annotations

import threading
import time


class Pacer:
    """Serializes requests to one host, holding ``delay`` between them.

    A plain ``sleep`` inside each worker would let N workers hit the host at
    once and merely stagger the next round. Holding the lock ACROSS the sleep is
    what actually caps the rate at one request per ``delay``, no matter how many
    workers are running.
    """

    def __init__(self, delay: float):
        self.delay = delay
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.delay
