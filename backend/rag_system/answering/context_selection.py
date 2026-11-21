"""Deterministic, source-aware selection of already-retrieved evidence."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence

from rag_system.api.schemas import EvidenceItem

_PROBLEM = re.compile(r"problem|stuck|loss|failure|downtime|tight\s+hole|washout|caving|npt", re.I)
_COMPARE = re.compile(r"compare|highest|lowest|most|least|by\s+section|which\s+(?:section|formation)", re.I)
_LIST = re.compile(r"main\s+.*problem|what\s+.*problem|list|summari[sz]", re.I)
_NPT = re.compile(r"npt|non[- ]productive|item\s*#?\s*\d+", re.I)
_SECTION = re.compile(r"section|hole\s+size|formation", re.I)
_WORD = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", re.I)
_STOP = {
    "what", "was", "were", "the", "a", "an", "in", "of", "did", "do",
    "is", "are", "which", "at", "for", "and", "or", "to", "by", "on",
    "well", "report", "reported", "according", "with", "from", "this",
}


def _role(item: EvidenceItem) -> str:
    metadata = item.metadata
    entity = str(metadata.get("entity_type") or "").upper()
    content = str(metadata.get("content_type") or "").lower()
    text = item.text
    if entity == "NPT_ITEM_ID" or "npt item" in text.lower():
        return "NPT_EVENT"
    if entity in {"FORMATION", "FORMATION_INFORMATION"} or "formation" in text.lower():
        return "FORMATION_INFORMATION"
    if metadata.get("logical_table_id") or metadata.get("table_id") or "table" in content or "field_value_pairs" in text:
        return "TABLE_ROW"
    if _PROBLEM.search(text):
        return "OPERATION_EVENT"
    return "TEXT_FACT"


def _dedupe(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[tuple[str, str, str]] = set()
    result: list[EvidenceItem] = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item.text).strip().lower()
        key = (normalized, str(item.metadata.get("pdf_page_start") or ""), str(item.metadata.get("logical_table_id") or item.metadata.get("table_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _sub_intents(query: str) -> dict[str, re.Pattern[str]]:
    q = query.lower()
    intents: dict[str, re.Pattern[str]] = {}
    if re.search(r"12¼|12\s*[-/]?\s*1\s*(?:[/\s]*4)?|bit\s+run", q):
        intents["hole_12_25"] = re.compile(r"12\s*[-/]?\s*1\s*(?:[/\s]*4)?|12¼|12-1/4|bit\s*run", re.I)
    if re.search(r"hole\s+condition|tight\s+hole|hard\s+back\s*ream|condition", q):
        intents["hole_condition"] = re.compile(r"tight\s+hole|hard\s+back\s*ream|hole\s+cond|condition", re.I)
    if re.search(r"stuck|sticking", q):
        intents["stuck_pipe"] = re.compile(r"stuck\s*(?:pipe|string)?|work\s+on\s+stuck", re.I)
    if "back-off" in q or "back off" in q:
        intents["back_off"] = re.compile(r"back[- ]?off|back[- ]?off\s+tools|electrical", re.I)
    if "sidetrack" in q or "side track" in q:
        intents["sidetrack"] = re.compile(r"side[- ]?track|sidetrack", re.I)
    if re.search(r"\bnpt\b|non[- ]productive|lost\s*time", q):
        intents["npt"] = re.compile(r"\bnpt\b|non[- ]productive|lost\s*time|work\s+on\s+stuck", re.I)
    if re.search(r"bit\s+run|rop|total\s+meters", q):
        intents["bit_runs"] = re.compile(r"bit\s*(?:run|#)|average\s*rop|rop", re.I)
    if "drilling fluid" in q or "mud" in q:
        intents["drilling_fluid"] = re.compile(r"drilling\s+fluid|mud|circulation|solids|viscosity|gpm|surface\s+line", re.I)
    if "drilling data" in q:
        intents["drilling_data"] = re.compile(r"drilling\s+data|rop|bit\s*\d+", re.I)
    return intents


def select_context(query: str, evidence: Sequence[EvidenceItem], budget: int = 12) -> list[EvidenceItem]:
    """Select evidence by multi-intent coverage before similarity or retrieval rank."""
    candidates = _dedupe(evidence)
    if not candidates or budget <= 0:
        return []
    intents = _sub_intents(query)
    q = query.lower()
    direct = {name: [item for item in candidates if pattern.search(item.text)] for name, pattern in intents.items()}
    chosen: list[EvidenceItem] = []

    def add(item: EvidenceItem) -> None:
        if item not in chosen and len(chosen) < budget:
            chosen.append(item)

    def page(item: EvidenceItem) -> int:
        try:
            return int(item.metadata.get("pdf_page_number") or item.metadata.get("pdf_page_start") or 0)
        except (TypeError, ValueError):
            return 0

    # Preserve complete comparison groups and required cross-section evidence.
    if "bit_runs" in intents:
        for item in candidates:
            if re.search(r"bit\s*(?:run|#)\s*(?:[4-9]|1[0-2])\b", item.text, re.I):
                add(item)
    if "drilling_fluid" in intents and "drilling_fluid" in direct:
        for item in direct["drilling_fluid"]: add(item)
    if "drilling_data" in intents and "drilling_data" in direct:
        for item in direct["drilling_data"]: add(item)
    # Reserve Well Summary/Montage anchors before broad NPT matches.
    if any(name in intents for name in ("hole_12_25", "stuck_pipe", "back_off", "sidetrack")):
        for item in sorted(candidates, key=lambda x: (0 if page(x) in (6, 7) else 1, page(x))):
            if page(item) in (6, 7) and (page(item) == 7 or any(pattern.search(item.text) for pattern in intents.values())):
                add(item)
    if "npt" in intents:
        for item in direct["npt"]: add(item)
    for name in ("hole_condition", "stuck_pipe", "back_off", "sidetrack", "npt"):
        for item in direct.get(name, []): add(item)

    # Fill remaining slots by direct sub-intent coverage, then retrieval order.
    for item in candidates:
        if len(chosen) >= budget: break
        if any(pattern.search(item.text) for pattern in intents.values()): add(item)
    if not any(any(pattern.search(item.text) for pattern in intents.values()) for item in candidates):
        for item in candidates:
            if len(chosen) >= budget: break
            add(item)
    return chosen[:budget]
