"""Prompt versioning helpers."""

from __future__ import annotations

import hashlib


def compute_prompt_hash(text: str) -> str:
    """First 16 hex chars of SHA256(text).

    Pin a prompt's hash alongside the prompt itself so replay determinism
    is self-verifying. Stable across Python versions.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
