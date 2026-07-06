"""
Simple in-memory cache with time-based expiration.

This module provides a lightweight cache so that repeated scan requests
with the same parameters (pattern + interval + lookback) don't re-run
the entire scanner if the previous result is still fresh.

How it works
------------
- Stores results in a plain Python dictionary.
- Each entry has a timestamp recording when it was saved.
- When you call get(), it checks if the entry is older than `ttl_seconds`.
  If yes, it returns None (cache miss). If no, it returns the cached data.
- Default TTL is 15 minutes (900 seconds).

No Redis, no external services — just a dict that lives in memory for as
long as the FastAPI server is running.
"""

import time


class ScanCache:
    """
    A simple time-based cache using a Python dictionary.

    Usage
    -----
    cache = ScanCache(ttl_seconds=900)   # 15-minute cache
    cache.set("cup_handle:1d:3mo", result_data)
    cached = cache.get("cup_handle:1d:3mo")  # returns data or None
    """

    def __init__(self, ttl_seconds=900):
        """
        Parameters
        ----------
        ttl_seconds : int
            How many seconds a cached result stays valid.
            Default is 900 (15 minutes).
        """
        self._store = {}          # {key: {"data": ..., "timestamp": ...}}
        self._ttl = ttl_seconds   # Time-to-live in seconds

    def get(self, key):
        """
        Retrieve a cached result if it exists and hasn't expired.

        Parameters
        ----------
        key : str
            Cache key (e.g. "cup_handle:1d:3mo").

        Returns
        -------
        dict or None
            The cached data, or None if not found / expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        age = time.time() - entry["timestamp"]
        if age > self._ttl:
            # Expired — remove it and return None
            del self._store[key]
            return None

        return entry["data"]

    def set(self, key, data):
        """
        Store a result in the cache with the current timestamp.

        Parameters
        ----------
        key : str
            Cache key.
        data : any
            The data to cache (typically a dict).
        """
        self._store[key] = {
            "data": data,
            "timestamp": time.time(),
        }

    def clear(self):
        """Remove all cached entries."""
        self._store.clear()

    def info(self):
        """
        Returns cache statistics for debugging.

        Returns
        -------
        dict
            Number of entries and TTL setting.
        """
        return {
            "entries": len(self._store),
            "ttl_seconds": self._ttl,
        }
