from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class _Entry[V]:
    expires_at: float
    value: V


class TTLCache[K, V]:
    def __init__(self, ttl_seconds: float, max_entries: int = 128) -> None:
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._entries[key] = _Entry(time.monotonic() + self._ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
