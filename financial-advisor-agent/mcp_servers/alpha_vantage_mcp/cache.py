from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

try:
    import redis  # type: ignore
except Exception:
    redis = None


class Cache:
    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url
        self._client = None
        if redis_url and redis is not None:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)

        # fallback in-memory cache (single-process)
        self._mem: Dict[str, Dict[str, Any]] = {}

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if self._client:
            val = self._client.get(key)
            if not val:
                return None
            return json.loads(val)
        obj = self._mem.get(key)
        if not obj:
            return None
        if obj["exp"] < time.time():
            self._mem.pop(key, None)
            return None
        return obj["val"]

    def set_json(self, key: str, value: Dict[str, Any], ttl_sec: int) -> None:
        if self._client:
            self._client.setex(key, ttl_sec, json.dumps(value))
            return
        self._mem[key] = {"val": value, "exp": time.time() + ttl_sec}
