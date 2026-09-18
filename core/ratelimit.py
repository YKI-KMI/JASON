"""A minimal thread-safe token-bucket rate limiter.

Used to be polite to free data sources (Yahoo Finance, news feeds) so dev/demo
runs don't get blocked.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, calls_per_minute: int = 30) -> None:
        self.min_interval = 60.0 / max(1, calls_per_minute)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                delta = now - self._last
                if delta >= self.min_interval:
                    self._last = now
                    return
                sleep_for = self.min_interval - delta
            time.sleep(sleep_for)
