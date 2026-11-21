from __future__ import annotations

import re
from collections import defaultdict

from .citation import make_citation
from .models import Chunk
from .normalizer import normalize_for_retrieval, token_estimate
from .text_chunker import document_label


def _table_label(table: dict) -> str:
    match = re.search(r"\bTable\s+([A-Z]?\.?\d+)", table.get("caption", ""), re.I)
    return match.group(1) if match else table["table_id"].rsplit("-TABLE-", 1)[-1]


def _row_labels(rows: list[dict]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        cells = row.get("cells", [])
        value = cells[0].get("value") if cells else None
        if value and re.fullmatch(r"[A-Z]?\d+(?:\.\d+)+", value.strip()):
            labels.append(value.strip())
    return list(dict.fromkeys(labels))


def build_table_chunks(registry: dict, pages: list[dict], tables: list[dict], config: dict) -> tuple[list[Chunk], list[Chunk]]:
    parents: list[Chunk] = []
    children: list[Chunk] = []
    label = document_label(registry)
    maximum = int(config["max_tokens"])
    for table in tables:
        table_label = _table_label(table)
        table_id = table["table_id"]
        table_pages = table["pdf_pages"]
        printed = [pages[i]["printed_page_number"] for i in table_pages]
        parent_id = f"{registry['document_id']}__TABLE-{table_label}__PARENT"
        caption = table.get("caption") or f"Table {table_label}"
        parent_citation = make_citation(registry=registry, document_label=label, clause=table.get("related_clause"), table=table_label, pdf_pages=table_pages, printed_pages=printed)
        parent_norm = normalize_for_retrieval(caption)
        parent_warnings = list(table.get("warnings", []))
        parents.append(Chunk(
            parent_id, registry["document_id"], registry["standard_code"], registry["edition"], registry.get("publication_date"), registry["language"],
            "table", table.get("related_clause"), None, table.get("related_clause"), None, None,
            min(table_pages), max(table_pages), printed[0] if printed else None, printed[-1] if printed else None,
            caption, parent_norm, f"{label}\nTable {table_label} — {caption}\nContent:\n{parent_norm}", table_id, None, parent_citation,
            [f"table:{table_id}"], token_estimate(parent_norm), len(caption), table.get("extraction_status", "partial"), parent_warnings,
            table.get("extraction_status") != "complete", "stage01", .95, segment_type="table_summary",
        ))
        rows_by_page: dict[int, list[dict]] = defaultdict(list)
        for row in table.get("structured_rows", []):
            rows_by_page[int(row["source_pdf_page"])].append(row)
        previous_segment: str | None = None
        for index, page_index in enumerate(table_pages, 1):
            source = pages[page_index]["raw_text"].strip()
            if not source:
                source = next((x for x in table.get("raw_extracted_text", "").split("\n\n") if x.strip()), caption)
            normalized = normalize_for_retrieval(source)
            segment_id = f"{table_id}-SEG-{index:03d}"
            rows = rows_by_page.get(page_index, [])
            row_labels = _row_labels(rows)
            warnings = list(table.get("warnings", []))
            warnings.append("partial table extraction; verify merged cells, diagrams, formulas, and column alignment visually")
            if not rows:
                warnings.append("no structured rows recovered for this table page")
            if token_estimate(normalized) > maximum:
                warnings.append("very large chunk; table page boundary preserved")
            printed_page = pages[page_index]["printed_page_number"]
            if printed_page is None:
                warnings.append("missing printed page")
            citation = make_citation(registry=registry, document_label=label, clause=table.get("related_clause"), table=table_label, pdf_pages=[page_index], printed_pages=[printed_page])
            rows_context = ", ".join(row_labels) if row_labels else "not confidently identified"
            embedding = f"{label}\nTable {table_label} — {caption}\nRelated section: {table.get('related_clause') or 'not assigned'}\nRows or entries: {rows_context}\nSource PDF page: {page_index}\nContent:\n{normalized}"
            children.append(Chunk(
                f"{registry['document_id']}__TABLE-{table_label}__CHUNK-{index:03d}", registry["document_id"], registry["standard_code"], registry["edition"], registry.get("publication_date"), registry["language"],
                "table_segment", table.get("related_clause"), None, table.get("related_clause"), None, None,
                page_index, page_index, printed_page, printed_page, source, normalized, embedding, table_id, None, citation,
                [f"table:{table_id}:page:{page_index}"] + [f"table:{table_id}:row:{r.get('source_row_index')}@page:{page_index}" for r in rows],
                token_estimate(normalized), len(source), "partial", sorted(set(warnings)), True, "stage01", .8,
                segment_id=segment_id, segment_type="table_header" if index == 1 else ("table_continuation" if rows else "table_unstructured_segment"),
                row_labels=row_labels, structured_rows=rows, continuation_of=previous_segment, parent_table_chunk_id=parent_id,
            ))
            previous_segment = segment_id
    return parents, children
