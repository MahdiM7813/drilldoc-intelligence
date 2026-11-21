from __future__ import annotations

import re


def normalize_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().upper()).strip("-._")
    if not normalized:
        raise ValueError("standard name and version must contain identifier characters")
    return normalized


def make_standard_id(name: str, version: str, revision: str | None = None) -> str:
    parts = [normalize_component(name), normalize_component(version)]
    if revision:
        parts.append(normalize_component(revision))
    return "-".join(parts)
