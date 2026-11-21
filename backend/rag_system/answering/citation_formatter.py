from __future__ import annotations

from rag_system.retrieval.schemas import RetrievalResult


def citation_for_result(result: RetrievalResult) -> dict[str, str]:
    clause = result.clause_number
    page = result.citation.get("printed_page_start") or str(result.pdf_page_start)
    return {
        "document": result.document_id,
        "clause": clause,
        "page": str(page),
        "chunk_id": result.chunk_id,
    }


def preserve_allowed_citations(candidate: list[dict], evidence: list[RetrievalResult]) -> list[dict[str, str]]:
    allowed = {item.chunk_id: citation_for_result(item) for item in evidence}
    preserved: list[dict[str, str]] = []
    for citation in candidate:
        chunk_id = str(citation.get("chunk_id") or "")
        if chunk_id in allowed and allowed[chunk_id] not in preserved:
            preserved.append(allowed[chunk_id])
    return preserved
