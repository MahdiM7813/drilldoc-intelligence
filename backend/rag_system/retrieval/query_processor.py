from __future__ import annotations

import re


def process_query(query: str) -> str:
    """Validate a query and normalize whitespace without rewriting its meaning."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    processed = re.sub(r"\s+", " ", query).strip()
    if not processed:
        raise ValueError("query must not be empty")
    return processed
