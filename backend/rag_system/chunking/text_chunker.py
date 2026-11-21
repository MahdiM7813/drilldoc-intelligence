from __future__ import annotations

import re
from collections import defaultdict

from .citation import make_citation
from .models import Chunk, HierarchyItem
from .normalizer import clean_page_furniture, normalize_for_retrieval, split_paragraphs, token_estimate


def document_label(registry: dict) -> str:
    year = (registry.get("publication_date") or "")[:4]
    return f"{registry['standard_code']}:{year}(E)" if year else registry["standard_code"]


def _heading_line(item: HierarchyItem) -> re.Pattern:
    if item.item_type == "annex":
        return re.compile(rf"^\s*{re.escape(item.number)}\s*$", re.I)
    if item.number in {"foreword", "introduction", "bibliography", "contents"}:
        return re.compile(rf"^\s*{re.escape(item.title)}\s*$", re.I)
    return re.compile(rf"^\s*{re.escape(item.number)}\s+{re.escape(item.title)}\s*$", re.I)


def page_segments(pages: list[dict], hierarchy: list[HierarchyItem], table_pages: set[int]) -> list[dict]:
    by_page: dict[int, list[HierarchyItem]] = defaultdict(list)
    for item in hierarchy:
        by_page[item.pdf_page].append(item)
    current: HierarchyItem | None = None
    result: list[dict] = []
    label = "ISO 5817:2014(E)"
    for page in pages:
        index = page["pdf_page_index"]
        if index in table_pages:
            continue
        source, furniture_warnings = clean_page_furniture(page["raw_text"], label)
        lines = source.splitlines()
        boundaries: list[tuple[int, HierarchyItem]] = []
        for item in by_page.get(index, []):
            pos = next((i for i, line in enumerate(lines) if _heading_line(item).fullmatch(line)), None)
            if pos is not None:
                boundaries.append((pos, item))
        boundaries.sort(key=lambda x: x[0])
        cursor = 0
        for pos, item in boundaries:
            if pos > cursor and current:
                result.append({"hierarchy": current, "page": page, "source_text": "\n".join(lines[cursor:pos]).strip(), "warnings": furniture_warnings})
            current = item
            cursor = pos
        if cursor < len(lines):
            if current:
                result.append({"hierarchy": current, "page": page, "source_text": "\n".join(lines[cursor:]).strip(), "warnings": furniture_warnings})
            elif source:
                fallback = HierarchyItem(f"front-page-{index}", "Front matter", index, page["printed_page_number"], "front_matter", "layout_hint", .6, ["front matter type uncertain"])
                result.append({"hierarchy": fallback, "page": page, "source_text": source, "warnings": furniture_warnings + fallback.warnings})
    return [x for x in result if x["source_text"].strip()]


def _embedding(label: str, item: HierarchyItem, content: str) -> str:
    lines = [label]
    if item.item_type == "annex" or item.number.startswith("Annex "):
        lines.append(f"{item.number} — {item.title}")
    else:
        lines.append(f"Section {item.number} — {item.title}")
        if "." in item.number:
            lines.append(f"Clause {item.number}")
    lines.extend(["Content:", content])
    return "\n".join(lines)


def _base_chunk(registry: dict, item: HierarchyItem, source: str, page_start: dict, page_end: dict, chunk_id: str, warnings: list[str], content_type: str | None = None) -> Chunk:
    normalized = normalize_for_retrieval(source)
    label = document_label(registry)
    merged_warnings = sorted(set(warnings + item.warnings))
    if page_start["printed_page_number"] is None:
        merged_warnings.append("missing printed page")
    if "�" in source:
        merged_warnings.append("corrupted Unicode")
    critical = ("hierarchy", "corrupted Unicode", "missing printed page", "very large chunk", "broken formula")
    needs_review = item.confidence < .8 or any(any(marker in warning for marker in critical) for warning in merged_warnings)
    status = "uncertain" if item.confidence < .75 else ("partial" if needs_review else "complete")
    pages = list(range(page_start["pdf_page_index"], page_end["pdf_page_index"] + 1))
    printed = [page_start["printed_page_number"], page_end["printed_page_number"]]
    clause = item.number if item.item_type in {"section", "clause"} and item.number not in {"bibliography", "contents"} else None
    citation = make_citation(registry=registry, document_label=label, clause=clause, table=None, pdf_pages=pages, printed_pages=printed)
    ctype = content_type or ("annex" if item.item_type == "annex" or item.number.startswith("Annex ") else ("front_matter" if item.item_type == "front_matter" else "clause"))
    return Chunk(
        chunk_id, registry["document_id"], registry["standard_code"], registry["edition"], registry.get("publication_date"), registry["language"],
        ctype, item.number, item.title, clause, item.title if clause else None, item.parent_clause,
        page_start["pdf_page_index"], page_end["pdf_page_index"], page_start["printed_page_number"], page_end["printed_page_number"],
        source, normalized, _embedding(label, item, normalized), None, None, citation,
        [f"page:{x}" for x in pages], token_estimate(normalized), len(source), status, merged_warnings,
        needs_review, item.detection_source, item.confidence,
    )


def _sentence_fallback(source: str, maximum: int) -> list[str]:
    if token_estimate(source) <= maximum:
        return [source]
    units = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])|\n(?=NOTE\b)", source) if x.strip()]
    if len(units) == 1:
        return [source]
    result: list[str] = []
    active = ""
    for unit in units:
        candidate = f"{active} {unit}".strip()
        if active and token_estimate(candidate) > maximum:
            result.append(active); active = unit
        else:
            active = candidate
    if active:
        result.append(active)
    return result


def build_prose_chunks(registry: dict, pages: list[dict], hierarchy: list[HierarchyItem], table_pages: set[int], config: dict) -> list[Chunk]:
    segments = page_segments(pages, hierarchy, table_pages)
    max_tokens = int(config["max_tokens"]); target = int(config["target_tokens"])
    chunks: list[Chunk] = []
    counters: dict[str, int] = defaultdict(int)
    for segment in segments:
        item, page, source = segment["hierarchy"], segment["page"], segment["source_text"]
        # Definition bodies get dedicated chunks; retain only the Section 3 heading/preamble here.
        if item.number == "3" and "3.1" in source:
            source = source.split("3.1", 1)[0].strip()
        if not source:
            continue
        parts = split_paragraphs(source)
        groups: list[str] = []
        active = ""
        for part in parts:
            candidate = f"{active}\n{part}".strip()
            if active and token_estimate(candidate) > target:
                groups.append(active); active = part
            else:
                active = candidate
        if active:
            groups.append(active)
        groups = [piece for group in groups for piece in _sentence_fallback(group, max_tokens)]
        for group in groups:
            warnings = list(segment["warnings"])
            if token_estimate(group) > max_tokens:
                warnings.append("very large chunk; semantic boundary preserved")
            key = re.sub(r"[^A-Za-z0-9.]+", "-", item.number.upper()).strip("-") or "UNSCOPED"
            counters[key] += 1
            chunk_id = f"{registry['document_id']}__CLAUSE-{key}__CHUNK-{counters[key]:03d}"
            chunks.append(_base_chunk(registry, item, group, page, page, chunk_id, warnings))
    return chunks


def build_definition_chunks(registry: dict, pages: list[dict], definitions: list[dict], config: dict) -> list[Chunk]:
    result: list[Chunk] = []
    label = document_label(registry)
    seen_ids: dict[str, int] = {}
    for definition in definitions:
        number, term, source = definition["number"], definition["term"], definition["source_text"]
        normalized = normalize_for_retrieval(source)
        page = pages[definition["pdf_page"]]
        warnings = list(definition["warnings"])
        if token_estimate(normalized) < int(config["min_tokens"]):
            warnings.append("very small chunk; definition isolation preserved")
        citation = make_citation(registry=registry, document_label=label, clause=number, table=None, pdf_pages=[definition["pdf_page"]], printed_pages=[definition["printed_page"]])
        embedding = f"{label}\nSection 3 — Terms and definitions\nClause {number} — {term}\nDefinition:\n{normalized}"
        base_id = f"{registry['document_id']}__DEFINITION-{number}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        result.append(Chunk(
            f"{base_id}__CHUNK-{seen_ids[base_id]:03d}", registry["document_id"], registry["standard_code"], registry["edition"], registry.get("publication_date"), registry["language"],
            "definition", "3", "Terms and definitions", number, term, "3", definition["pdf_page"], definition["pdf_page"], definition["printed_page"], definition["printed_page"],
            source, normalized, embedding, None, None, citation, [f"page:{definition['pdf_page']}:definition:{number}"], token_estimate(normalized), len(source), "uncertain", warnings, True,
            definition["detection_source"], definition["confidence"], term, definition["definition_text"],
        ))
    return result
