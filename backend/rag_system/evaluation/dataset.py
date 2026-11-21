"""Validation for human-annotated synthetic question fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED = {"question_id", "question", "expected_answer", "relevant_document_ids", "relevant_evidence_ids", "category", "answering_allowed", "expected_abstention"}
CATEGORIES = {"direct_factual", "table_based", "numerical_unit", "multi_chunk", "multi_step", "unanswerable"}


def validate_record(record: dict[str, Any]) -> None:
    missing = REQUIRED - record.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if not isinstance(record["question_id"], str) or not record["question_id"].strip():
        raise ValueError("question_id must be a non-empty string")
    if record["category"] not in CATEGORIES:
        raise ValueError("unknown question category")
    if not isinstance(record["answering_allowed"], bool) or not isinstance(record["expected_abstention"], bool):
        raise ValueError("answering_allowed and expected_abstention must be booleans")
    if record["answering_allowed"] == record["expected_abstention"]:
        raise ValueError("answerability and expected abstention must be opposites")
    for key in ("relevant_document_ids", "relevant_evidence_ids"):
        if not isinstance(record[key], list) or any(not isinstance(x, str) or not x for x in record[key]):
            raise ValueError(f"{key} must be a list of non-empty strings")
    if record["answering_allowed"] and (not record["expected_answer"] or not record["relevant_evidence_ids"]):
        raise ValueError("answerable questions require a gold answer and evidence IDs")
    if not record["answering_allowed"] and record["relevant_evidence_ids"]:
        raise ValueError("unanswerable questions must not have supporting evidence IDs")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        validate_record(row)
        if row["question_id"] in seen:
            raise ValueError(f"duplicate question_id: {row['question_id']}")
        seen.add(row["question_id"])
    return rows
