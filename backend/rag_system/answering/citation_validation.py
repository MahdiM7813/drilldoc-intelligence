from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any

SOURCE_ID = re.compile(r"\[S(\d+)\]")

@dataclass(frozen=True)
class CitationValidation:
    valid_ids: list[str]
    invalid_ids: list[str]
    sources: list[dict[str, Any]]

def extract_citations(text: str) -> list[str]:
    result = []
    for match in SOURCE_ID.finditer(text):
        value = f"S{match.group(1)}"
        if value not in result:
            result.append(value)
    return result

def validate_citations(text: str, source_map: dict[str, dict[str, Any]]) -> CitationValidation:
    ids = extract_citations(text)
    valid = [value for value in ids if value in source_map]
    invalid = [value for value in ids if value not in source_map]
    return CitationValidation(valid, invalid, [source_map[value] for value in valid])
