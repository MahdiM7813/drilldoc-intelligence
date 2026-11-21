from __future__ import annotations


def edition_year(publication_date: str | None) -> str | None:
    return publication_date[:4] if publication_date else None


def make_citation(*, registry: dict, document_label: str, clause: str | None, table: str | None, pdf_pages: list[int], printed_pages: list[str | None]) -> dict:
    known_printed = [str(x) for x in printed_pages if x is not None]
    start, end = min(pdf_pages), max(pdf_pages)
    result = {
        "document_id": registry["document_id"],
        "standard_code": registry["standard_code"],
        "edition_year": edition_year(registry.get("publication_date")),
        "document_label": document_label,
        "clause": clause,
        "table": table,
        "pdf_pages": sorted(set(pdf_pages)),
        "printed_pages": list(dict.fromkeys(known_printed)),
        "pdf_page_start": start,
        "pdf_page_end": end,
        "printed_page_start": known_printed[0] if known_printed else None,
        "printed_page_end": known_printed[-1] if known_printed else None,
    }
    parts = [document_label]
    if clause:
        parts.append(f"Clause {clause}")
    if table:
        parts.append(f"Table {table}")
    if known_printed:
        parts.append(f"printed page {known_printed[0]}" if len(set(known_printed)) == 1 else f"printed pages {known_printed[0]}-{known_printed[-1]}")
    else:
        parts.append(f"PDF page {start}" if start == end else f"PDF pages {start}-{end}")
    result["display"] = ", ".join(parts)
    return result
