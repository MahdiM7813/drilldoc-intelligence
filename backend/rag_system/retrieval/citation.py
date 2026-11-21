from __future__ import annotations

from typing import Any


def citation_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose stored citation provenance without inventing missing values."""
    return {
        "display": str(metadata.get("citation_display") or ""),
        "clause": str(metadata.get("citation_clause") or ""),
        "table": str(metadata.get("citation_table") or ""),
        "pdf_page_start": int(metadata.get("pdf_page_start", 0)),
        "pdf_page_end": int(metadata.get("pdf_page_end", 0)),
        "printed_page_start": str(metadata.get("printed_page_start") or ""),
        "printed_page_end": str(metadata.get("printed_page_end") or ""),
    }
