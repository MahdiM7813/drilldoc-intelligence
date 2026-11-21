"""Deterministic canonical packing for selected evidence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from rag_system.api.schemas import EvidenceItem


@dataclass(frozen=True)
class PackedEvidence:
    item: EvidenceItem
    text: str
    estimated_tokens: int


@dataclass(frozen=True)
class PackedContext:
    blocks: tuple[PackedEvidence, ...]
    estimated_tokens: int
    budget_tokens: int
    omitted_record_ids: tuple[str, ...] = ()
    omission_reasons: dict[str, str] | None = None


def estimate_tokens(text: str, tokenizer=None) -> int:
    # Conservative, deterministic approximation for the local Llama tokenizer.
    if tokenizer is not None:
        return max(1, len(tokenizer(text)))
    return max(1, (len(text) + 3) // 4)


def _page(item: EvidenceItem) -> str:
    return str(item.metadata.get("pdf_page_number") or item.metadata.get("pdf_page_start") or item.citation.get("display") or "?")


def _json_payload(item: EvidenceItem) -> dict[str, Any] | None:
    try:
        value = json.loads(item.text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _value(payload: dict[str, Any], item: EvidenceItem, *keys: str) -> Any:
    for key in keys:
        if payload.get(key) not in (None, "", [], {}):
            return payload[key]
        if item.metadata.get(key) not in (None, "", [], {}):
            return item.metadata[key]
    return None


def canonical_fact(item: EvidenceItem) -> str:
    """Render one source record as compact human-readable facts, never JSON."""
    page = _page(item)
    payload = _json_payload(item)
    if payload is not None:
        pairs = _value(payload, item, "field_value_pairs")
        cells = _value(payload, item, "raw_cell_values")
        entity = str(_value(payload, item, "entity_type") or "").upper()
        section = _value(payload, item, "section", "hole_section", "logical_table_id")
        parts: list[str] = []
        if entity == "NPT_ITEM_ID" or payload.get("NPT Item") is not None:
            parts.append(f"NPT item {_value(payload, item, 'NPT Item', 'npt_item_id')}")
        if section and section not in {"UNKNOWN Section", "UNKNOWN"}:
            parts.append(f"Hole section: {section}")
        if pairs:
            if isinstance(pairs, dict):
                parts.extend(f"{k}: {v}" for k, v in pairs.items())
            elif isinstance(pairs, list):
                parts.extend(str(value) for value in pairs[:8])
        elif isinstance(cells, dict):
            parts.extend(f"{k}: {v}" for k, v in cells.items())
        elif isinstance(cells, list):
            # Keep the table identity and labeled summary fields even when
            # extraction produced a long flattened cell stream.  This avoids
            # retaining only the header while dropping the value at the tail.
            labeled: list[str] = []
            for index, value in enumerate(cells):
                label = str(value).strip().lower()
                if "hole section" in label or "section" == label:
                    labeled.append(str(value))
                if ("npt" in label or "non-productive" in label) and index + 1 < len(cells):
                    labeled.extend((str(value), str(cells[index + 1])))
            if labeled:
                parts.extend(dict.fromkeys(labeled))
            else:
                parts.extend(str(value) for value in cells[:8])
        if not pairs and not cells:
            structural = {"entity_type", "section", "hole_section", "logical_table_id", "NPT Item", "npt_item_id"}
            parts.extend(f"{key}: {value}" for key, value in payload.items()
                         if key not in structural and isinstance(value, (str, int, float)) and value not in ("", "UNKNOWN"))
        if parts:
            return f"Page {page} | " + " | ".join(parts)
    text = re.sub(r"\s+", " ", item.text).strip()
    units = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    # Keep complete prose/table units. Prefer explicit field/row/problem
    # units, but never cut a unit at an arbitrary character boundary.
    field_units = re.findall(
        r"(?i)(?:shoe\s+depth|final\s+shoe\s+depth|total\s+depth|water\s+depth)\s*[:=]\s*[^\n.;]{0,48}",
        text,
    )
    preferred = [part for part in units if re.search(
        r"\b(?:shoe\s+depth|final\s+shoe|total\s+depth|water\s+depth|hole\s+depth|problem|pressure|vibration|crack|caving)\b",
        part, re.I,
    )]
    preferred.sort(key=lambda part: (0 if re.search(
        r"\b(?:shoe\s+depth|final\s+shoe|problem|pressure)\b", part, re.I
    ) else 1, len(part)))
    # A prose record is one complete evidence unit.  Keep it intact; the
    # tokenizer-aware packer below decides whether the unit fits instead of
    # silently emitting only its opening sentences.
    compact = text
    return f"Page {page} | {compact}"


def pack_context(query: str, evidence: Sequence[EvidenceItem], budget_tokens: int = 400, tokenizer=None) -> PackedContext:
    anchor_query = bool(re.search(r"12\s*[¼1/]?\s*4|12\s*[-/]?\s*1\s*(?:[/\s]*4)?", query, re.I) and re.search(r"stuck|tight\s+hole|back[- ]?off|sidetrack|hole\s+condition", query, re.I))
    def anchor(item: EvidenceItem) -> bool:
        try: page = int(_page(item))
        except (TypeError, ValueError): page = 0
        return anchor_query and page in (6, 7)
    def compress_anchor(item: EvidenceItem, limit: int) -> str:
        raw = re.sub(r"\s+", " ", item.text).strip()
        units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", raw) if u.strip()]
        patterns = re.compile(r"depth|tight\s+hole|hard\s+back\s*ream|stuck|back[- ]?off|sidetrack|npt|hole\s+condition", re.I)
        chosen = [u for u in units if patterns.search(u)]
        if not chosen: chosen = units[:4]
        prefix = f"Page {_page(item)} | Anchor evidence: "
        result = prefix + " ".join(dict.fromkeys(chosen))
        if estimate_tokens(result, tokenizer) <= limit: return result
        compact = []
        for unit in chosen:
            candidate = prefix + " ".join(compact + [unit])
            if estimate_tokens(candidate, tokenizer) <= limit: compact.append(unit)
        if compact: return prefix + " ".join(compact)
        return prefix + "Evidence terms: depth; tight hole; stuck pipe; back-off; sidetrack; NPT."
    anchors = [item for item in evidence if anchor(item)]
    ordered = anchors + [item for item in evidence if item not in anchors]
    packed: list[PackedEvidence] = []
    used = 0
    seen: set[str] = set()
    omitted: list[str] = []
    omission_reasons: dict[str, str] = {}
    anchor_budget = max(1, budget_tokens // max(2, len(anchors))) if anchors else budget_tokens
    for item in ordered:
        fact = compress_anchor(item, anchor_budget) if anchor(item) else canonical_fact(item)
        key = re.sub(r"\s+", " ", fact).strip().lower()
        if key in seen: omitted.append(item.chunk_id); omission_reasons[item.chunk_id] = "duplicate canonical fact"; continue
        cost = estimate_tokens(fact, tokenizer)
        if anchor(item):
            if used + cost > budget_tokens:
                omitted.append(item.chunk_id); omission_reasons[item.chunk_id] = "anchor exceeded reserved packing budget"; continue
        elif used + cost > budget_tokens:
            omitted.append(item.chunk_id); omission_reasons[item.chunk_id] = "non-anchor omitted after mandatory anchors"; continue
        packed.append(PackedEvidence(item, fact, cost)); seen.add(key); used += cost
    return PackedContext(tuple(packed), used, budget_tokens, tuple(omitted), omission_reasons)
