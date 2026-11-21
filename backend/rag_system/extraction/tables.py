from __future__ import annotations

import re
from typing import Any

from .structure import related_clause


CAPTION_RE = re.compile(r"(?mi)^\s*Table\s+([A-Z]?\.?\d+)\s*[-—:]\s*(.+)$")


def _cell(value: Any) -> dict:
    if value is None or not str(value).strip():
        return {"value": None, "uncertain": True}
    value = str(value).strip()
    return {"value": value, "uncertain": "�" in value}


def inspect_tables(pages: list[dict], page_tables: list[list], clauses: list[dict], document_id: str) -> list[dict]:
    runs: list[list[int]] = []
    active: list[int] = []
    for index, (page, detected) in enumerate(zip(pages, page_tables)):
        landscape = page.get("page_width_points", 0) > page.get("page_height_points", 0)
        is_table_page = (bool(detected) and landscape) or bool(CAPTION_RE.search(page["normalized_text"]))
        continuation = active and index == active[-1] + 1 and (is_table_page or "ISO 6520" in page["normalized_text"])
        if is_table_page or continuation:
            active.append(index)
        elif active:
            runs.append(active); active = []
    if active:
        runs.append(active)

    results: list[dict] = []
    for n, run in enumerate(runs, 1):
        combined = "\n\n".join(pages[i]["raw_text"] for i in run)
        caption_match = CAPTION_RE.search(combined)
        if caption_match:
            caption = f"Table {caption_match.group(1)} - {caption_match.group(2).strip()}"
        else:
            context = "\n".join(pages[i]["normalized_text"] for i in range(max(0, run[0] - 2), run[0]))
            mentions = re.findall(r"(?i)\bTable\s+([A-Z]?\.?\d+)\b", context)
            caption = f"Table {mentions[-1]} - caption inferred from preceding context" if mentions else f"Detected table region {n}"
        rows = []
        for page_index in run:
            for local_index, table in enumerate(page_tables[page_index]):
                for row_index, row in enumerate(table or []):
                    rows.append({"source_pdf_page": page_index, "source_table_index": local_index, "source_row_index": row_index, "cells": [_cell(cell) for cell in row]})
        uncertainty = sum(cell["uncertain"] for row in rows for cell in row["cells"])
        status = "failed" if not rows else ("partial" if uncertainty or len(run) > 1 else "complete")
        warnings = ["automated layout extraction; values must be visually verified before downstream use"]
        if len(run) > 1:
            warnings.append("multi-page table region; row continuity is uncertain")
        if uncertainty:
            warnings.append(f"uncertain_or_empty_cells={uncertainty}; no values were guessed")
        if not rows:
            warnings.append("caption/continuation detected but structured rows were not recovered")
        results.append({
            "table_id": f"{document_id.rsplit('-EN', 1)[0]}-TABLE-{n}",
            "document_id": document_id,
            "caption": caption,
            "pdf_pages": run,
            "printed_pages": [pages[i]["printed_page_number"] for i in run],
            "related_clause": related_clause(run[0], clauses),
            "raw_extracted_text": combined,
            "structured_rows": rows,
            "extraction_status": status,
            "warnings": warnings,
        })
    return results
