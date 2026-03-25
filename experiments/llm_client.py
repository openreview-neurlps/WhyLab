"""
LLM Client with 3-mode caching for reproducible experiments.
=============================================================
Modes:
  - online:  API call + cache write (initial data collection)
  - replay:  cache-only, errors on miss (deterministic reproduction)
  - hybrid:  cache hit → reuse, miss → API call (cost-controlled ablations)

Cache key = sha256(model + temperature + system_prompt + user_prompt + seed).
Each entry stores: prompt, response, token counts, latency, timestamp.
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from experiments.llm_providers import LLMResponse, create_provider




class CacheMissError(Exception):
    """Raised in replay mode when a cache entry is missing."""
    pass


def _safe_model_name(model: str) -> str:
    """Convert model name to filesystem-safe string."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model)


class CachedLLMClient:
    """LLM client with deterministic caching for reproducibility.

    Supports Gemini, OpenAI (GPT-5), and Anthropic (Claude) via
    the Provider pattern in llm_providers.py.

    Args:
        model: Model identifier (e.g. "gemini-2.0-flash", "gpt-5-mini", "claude-sonnet-4.6").
        cache_dir: Directory for cache JSONL files.
        mode: "online" | "replay" | "hybrid".
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum output tokens.
        prompt_version: Version tag to invalidate cache on template changes.
    """

    VALID_MODES = {"online", "replay", "hybrid"}

    def __init__(
        self,
        model: str,
        cache_dir: str | Path,
        mode: str = "replay",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        prompt_version: str = "v1",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version

        # Cache: key → response dict (model-specific file)
        self._cache: dict[str, dict] = {}
        safe_name = _safe_model_name(model)
        self._cache_file = self.cache_dir / f"llm_cache_{safe_name}.jsonl"
        # Legacy fallback: also load old e4_llm_cache.jsonl if it exists
        legacy_file = self.cache_dir / "e4_llm_cache.jsonl"
        if legacy_file.exists():
            self._load_cache_from(legacy_file)
        self._load_cache_from(self._cache_file)

        # Stats
        self.stats = {"calls": 0, "cache_hits": 0, "api_calls": 0, "total_tokens": 0}

        # Init provider (lazy — only if online/hybrid)
        self._provider = None
        if mode in ("online", "hybrid"):
            self._provider = create_provider(model, temperature, max_tokens)

    def _cache_key(self, system_prompt: str, user_prompt: str, seed: int = 0) -> str:
        """Deterministic cache key from all parameters that affect output."""
        payload = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt_version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "seed": seed,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _load_cache_from(self, path: Path):
        """Load cache entries from a JSONL file."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._cache[entry["key"]] = entry

    def _save_entry(self, entry: dict):
        """Append a single cache entry to JSONL file."""
        with open(self._cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _call_api(self, system_prompt: str, user_prompt: str, seed: int = 0) -> LLMResponse:
        """Make API call via provider with automatic retry on rate limit."""
        if self._provider is None:
            raise RuntimeError(
                f"Provider not initialized for {self.model}. Check API keys."
            )

        max_retries = 5
        base_delay = 10

        for attempt in range(max_retries + 1):
            try:
                return self._provider.call(system_prompt, user_prompt, seed=seed)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str or "resource" in err_str
                if is_rate_limit and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 320)
                    print(f"  [Rate limit] Retry {attempt + 1}/{max_retries} after {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        seed: int = 0,
    ) -> LLMResponse:
        """Generate LLM response with caching.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The actual task prompt.
            seed: Seed for cache key differentiation.

        Returns:
            LLMResponse with text and metadata.

        Raises:
            CacheMissError: In replay mode when cache entry is missing.
        """
        self.stats["calls"] += 1
        key = self._cache_key(system_prompt, user_prompt, seed)

        # Check cache
        if key in self._cache:
            entry = self._cache[key]
            self.stats["cache_hits"] += 1
            return LLMResponse(
                text=entry["response"],
                prompt_tokens=entry.get("prompt_tokens", 0),
                completion_tokens=entry.get("completion_tokens", 0),
                latency_ms=0.0,
                cache_hit=True,
                model=entry.get("model", self.model),
            )

        # Cache miss
        if self.mode == "replay":
            raise CacheMissError(
                f"Cache miss in replay mode. key={key[:16]}... "
                f"prompt_version={self.prompt_version}"
            )

        # Online or hybrid → make API call
        self.stats["api_calls"] += 1
        resp = self._call_api(system_prompt, user_prompt, seed=seed)
        self.stats["total_tokens"] += resp.prompt_tokens + resp.completion_tokens

        # Save to cache
        entry = {
            "key": key,
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "seed": seed,
            "response": resp.text,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_ms": resp.latency_ms,
            "timestamp": time.time(),
        }
        self._cache[key] = entry
        self._save_entry(entry)

        return resp

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "mode": self.mode,
        }
