from __future__ import annotations

import re
from collections import defaultdict

from .models import HierarchyItem


MAIN_RE = re.compile(r"^(\d+)\s+([^\n]{2,160})$")
SUB_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+)+)\s+([^\n]{2,160})$")
ANNEX_RE = re.compile(r"^Annex\s+([A-Z])\s*$", re.I)
SPECIAL_RE = re.compile(r"^(Foreword|Introduction|Bibliography|Contents)\s*$", re.I)
EMBEDDED_DEFINITION_RE = re.compile(r"3[A-Za-z ]{0,3}\.\s*(\d+)")


def _parent(number: str) -> str | None:
    if "." in number:
        return number.rsplit(".", 1)[0]
    return None


def resolve_hierarchy(pages: list[dict], stage01_clauses: list[dict], table_page_indexes: set[int]) -> list[HierarchyItem]:
    body_start = next((p["pdf_page_index"] for p in pages if "INTERNATIONAL STANDARD" in p["normalized_text"] and "1 Scope" in p["normalized_text"]), 0)
    stage_numbers = {str(c.get("clause_number")) for c in stage01_clauses}
    items: list[HierarchyItem] = []
    seen: set[tuple[str, str]] = set()
    for page in pages:
        index = page["pdf_page_index"]
        if index in table_page_indexes:
            continue
        lines = page["normalized_text"].splitlines()
        for line in lines:
            stripped = line.strip()
            match = ANNEX_RE.fullmatch(stripped)
            if match:
                number = f"Annex {match.group(1).upper()}"
                title = next((x.strip() for x in lines[lines.index(line) + 1:] if x.strip() and not re.match(r"^\([^)]*\)$", x.strip())), number)
                key = ("annex", number)
                if key not in seen:
                    items.append(HierarchyItem(number, title, index, page["printed_page_number"], "annex", "regex", .94, [], None)); seen.add(key)
                continue
            match = SUB_RE.fullmatch(stripped)
            if match and index >= body_start:
                number, title = match.groups(); key = ("clause", number)
                if key not in seen:
                    items.append(HierarchyItem(number, title, index, page["printed_page_number"], "clause", "regex", .91, [], _parent(number))); seen.add(key)
                continue
            match = MAIN_RE.fullmatch(stripped)
            if match and index >= body_start:
                number, title = match.groups()
                if number == "5817" or len(title.split()) < 1:
                    continue
                key = ("section", number)
                if key not in seen:
                    source = "stage01" if number in stage_numbers else "regex"
                    confidence = .96 if re.match(r"^(Scope|Normative references|Terms and definitions|Symbols|Assessment of imperfections)$", title, re.I) else .78
                    warnings = [] if confidence >= .9 else ["heading candidate requires manual confirmation"]
                    items.append(HierarchyItem(number, title, index, page["printed_page_number"], "section", source, confidence, warnings, None)); seen.add(key)
                continue
            match = SPECIAL_RE.fullmatch(stripped)
            if match:
                title = match.group(1).title(); key = ("front" if index < body_start else "section", title)
                if key not in seen:
                    items.append(HierarchyItem(title.lower(), title, index, page["printed_page_number"], "front_matter" if index < body_start else "section", "regex", .9, [], None)); seen.add(key)
    return sorted(items, key=lambda x: (x.pdf_page, _line_position(pages[x.pdf_page]["normalized_text"], x), x.number))


def _line_position(text: str, item: HierarchyItem) -> int:
    candidates = [item.number, item.title, f"{item.number} {item.title}"]
    positions = [text.find(x) for x in candidates if text.find(x) >= 0]
    return min(positions) if positions else len(text)


def items_by_page(items: list[HierarchyItem]) -> dict[int, list[HierarchyItem]]:
    result: dict[int, list[HierarchyItem]] = defaultdict(list)
    for item in items:
        result[item.pdf_page].append(item)
    return result


def extract_definition_candidates(pages: list[dict], hierarchy: list[HierarchyItem]) -> list[dict]:
    """Recover definitions only where clause numbers visibly occur; preserve contaminated source."""
    section3 = next((x for x in hierarchy if x.number == "3"), None)
    if not section3 or not re.fullmatch(r"Terms and definitions", section3.title.strip(), re.I):
        return []
    page = pages[section3.pdf_page]
    lines = page["raw_text"].splitlines()
    start = next((i for i, line in enumerate(lines) if re.search(r"\b3\s+Terms and definitions\b", line)), 0)
    candidates: list[dict] = []
    markers: list[tuple[int, str]] = []
    for i, line in enumerate(lines[start:], start):
        match = EMBEDDED_DEFINITION_RE.search(line)
        if match:
            markers.append((i, f"3.{match.group(1)}"))
    for pos, (line_index, number) in enumerate(markers):
        term_index = line_index + 1
        if term_index >= len(lines):
            continue
        term = lines[term_index].strip()
        if not term or re.match(r"^(?:Note|©|\d+)$", term, re.I):
            continue
        next_line = markers[pos + 1][0] if pos + 1 < len(markers) else len(lines)
        definition_lines = lines[term_index + 1: next_line + (1 if pos + 1 < len(markers) else 0)]
        definition_lines = [x for x in definition_lines if not re.fullmatch(r"\s*(?:\d+|©.*)", x)]
        source = "\n".join([term] + definition_lines).strip()
        if source:
            candidates.append({
                "number": number,
                "term": term,
                "source_text": source,
                "definition_text": "\n".join(definition_lines).strip(),
                "pdf_page": page["pdf_page_index"],
                "printed_page": page["printed_page_number"],
                "detection_source": "layout_hint",
                "confidence": .62,
                "warnings": ["definition number was interleaved with adjacent text by PDF extraction; source preserved verbatim", "hierarchy assignment requires manual review"],
            })
    return candidates
