from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .models import SUPPORTED_CONTENT_TYPES


def validate_chunks(chunks: list[dict], registry: dict, tables: list[dict], config: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    ids = [x.get("chunk_id") for x in chunks]
    duplicates = [x for x, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate chunk IDs: {duplicates}")
    valid_tables = {x["table_id"] for x in tables}
    text_hashes: dict[str, list[str]] = defaultdict(list)
    max_tokens = int(config["max_tokens"])
    for chunk in chunks:
        cid = chunk.get("chunk_id", "<missing>")
        if chunk.get("document_id") != registry.get("document_id"):
            errors.append(f"{cid}: invalid document reference")
        if not str(chunk.get("source_text", "")).strip():
            errors.append(f"{cid}: empty source text")
        if chunk.get("content_type") not in SUPPORTED_CONTENT_TYPES:
            errors.append(f"{cid}: unsupported content type")
        start, end = chunk.get("pdf_page_start"), chunk.get("pdf_page_end")
        if not isinstance(start, int) or not isinstance(end, int) or start > end:
            errors.append(f"{cid}: invalid page range")
        citation = chunk.get("citation", {})
        if citation.get("pdf_page_start") != start or citation.get("pdf_page_end") != end:
            errors.append(f"{cid}: citation/page mismatch")
        if chunk.get("content_type") in {"table", "table_segment"} and chunk.get("table_id") not in valid_tables:
            errors.append(f"{cid}: unknown table ID")
        if chunk.get("token_estimate", 0) > max_tokens and not any("very large chunk" in x for x in chunk.get("warnings", [])):
            errors.append(f"{cid}: exceeds maximum without warning")
        digest = hashlib.sha256(chunk.get("normalized_text", "").encode("utf-8")).hexdigest()
        text_hashes[digest].append(cid)
    duplicate_text_groups = [group for group in text_hashes.values() if len(group) > 1]
    if duplicate_text_groups:
        warnings.append(f"exact duplicate normalized text groups={len(duplicate_text_groups)}")
    warning_counts = Counter(w for chunk in chunks for w in chunk.get("warnings", []))
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "duplicate_chunk_ids": duplicates,
        "duplicate_text_groups": duplicate_text_groups,
        "duplicate_chunk_count": sum(len(x) - 1 for x in duplicate_text_groups),
        "warnings_by_category": dict(sorted(warning_counts.items())),
    }
