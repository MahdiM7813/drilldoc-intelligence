from __future__ import annotations

import re
from typing import Iterable


FRONT_PATTERNS = [
    ("cover", re.compile(r"INTERNATIONAL\s+STANDARD", re.I)),
    ("copyright", re.compile(r"COPYRIGHT PROTECTED DOCUMENT|All rights reserved", re.I)),
    ("table_of_contents", re.compile(r"^Contents\s*$", re.I | re.M)),
    ("foreword", re.compile(r"^Foreword\s*$", re.I | re.M)),
    ("introduction", re.compile(r"^Introduction\s*$", re.I | re.M)),
]
SECTION_TITLES = {
    "scope": "Scope",
    "normative references": "Normative references",
    "terms and definitions": "Terms and definitions",
    "symbols": "Symbols",
    "bibliography": "Bibliography",
}
HEADING_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\s+([^\n]{3,160})\s*$")
ANNEX_RE = re.compile(r"(?mi)^\s*Annex\s+([A-Z])(?:\s*\([^)]*\))?\s*$")
TABLE_RE = re.compile(r"(?mi)^\s*Table\s+([A-Z]?\.?\d+)\s*[-—:]\s*(.+)$")
FIGURE_RE = re.compile(r"(?mi)^\s*Figure\s+([A-Z]?\.?\d+)\s*[-—:]\s*(.+)$")


def _item(document_id: str, kind: str, label: str, title: str, page: int, confidence: float, method: str) -> dict:
    return {"document_id": document_id, "type": kind, "label": label, "title": title.strip(), "start_pdf_page": page, "end_pdf_page": None, "confidence": confidence, "detection_method": method, "warnings": []}


def detect_structure(pages: list[dict], document_id: str) -> list[dict]:
    items: list[dict] = []
    seen: set[tuple] = set()
    for page in pages:
        idx, text = page["pdf_page_index"], page["normalized_text"]
        for label, pattern in FRONT_PATTERNS:
            key = ("front_matter", label)
            if key not in seen and pattern.search(text):
                items.append(_item(document_id, "front_matter", label, label.replace("_", " ").title(), idx, .92, "regex")); seen.add(key)
        for label, title in SECTION_TITLES.items():
            match = re.search(rf"(?mi)^\s*(\d+(?:\.\d+)*)?\s*{re.escape(title)}\s*$", text)
            if match and ("section", match.group(1) or label) not in seen:
                number = match.group(1) or label
                items.append(_item(document_id, "section", number, title, idx, .9, "regex")); seen.add(("section", number))
        if not page["has_tables"]:
            for match in HEADING_RE.finditer(text):
                number, title = match.groups(); key = ("section", number)
                if key not in seen and not title.lower().startswith(("iso ", "© ")):
                    items.append(_item(document_id, "section", number, title, idx, .72, "regex")); seen.add(key)
        for match in ANNEX_RE.finditer(text):
            label = match.group(1); key = ("annex", label)
            if key not in seen:
                tail = text[match.end():].splitlines()
                title = next((x.strip() for x in tail if x.strip()), f"Annex {label}")
                items.append(_item(document_id, "annex", f"Annex {label}", title, idx, .88, "regex")); seen.add(key)
        for kind, pattern in (("table", TABLE_RE), ("figure", FIGURE_RE)):
            for match in pattern.finditer(text):
                key = (kind, match.group(1))
                if key not in seen:
                    items.append(_item(document_id, kind, match.group(1), match.group(2), idx, .82, "regex")); seen.add(key)
    ordered = sorted(items, key=lambda x: (x["start_pdf_page"], x["type"], x["label"]))
    for kind in {x["type"] for x in ordered}:
        group = [x for x in ordered if x["type"] == kind]
        for current, nxt in zip(group, group[1:]):
            current["end_pdf_page"] = max(current["start_pdf_page"], nxt["start_pdf_page"] - 1)
    return ordered


def detect_clauses(pages: list[dict], document_id: str) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for page in pages:
        text = page["normalized_text"]
        if page["has_tables"]:
            continue
        for match in HEADING_RE.finditer(text):
            number, title = match.groups()
            if number in seen or title.lower().startswith(("iso ", "© ")):
                continue
            seen.add(number)
            candidates.append({
                "clause_id": f"{document_id.rsplit('-EN', 1)[0]}-{number}",
                "document_id": document_id,
                "clause_number": number,
                "title": title.strip(),
                "pdf_page_start": page["pdf_page_index"],
                "printed_page_start": page["printed_page_number"],
                "raw_text": "",
                "parent_clause": number.rsplit(".", 1)[0] if "." in number else None,
                "warnings": ["regex-derived candidate; verify against page layout"],
            })
    candidates.sort(key=lambda x: x["pdf_page_start"])
    for i, clause in enumerate(candidates):
        start = clause["pdf_page_start"]
        end = candidates[i + 1]["pdf_page_start"] if i + 1 < len(candidates) else len(pages)
        clause["raw_text"] = "\n\n".join(p["raw_text"] for p in pages[start:end]).strip()
    return candidates


def related_clause(page_index: int, clauses: Iterable[dict]) -> str | None:
    prior = [c for c in clauses if c["pdf_page_start"] <= page_index]
    return prior[-1]["clause_number"] if prior else None
