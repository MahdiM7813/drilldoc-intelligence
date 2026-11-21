from __future__ import annotations

import json


REASONING_SYSTEM = """You are an evidence-grounded assistant for technical welding standards.
Use only the evidence excerpts provided by the user. Do not use outside knowledge.
Do not invent requirements, clause numbers, values, or conclusions.
For tables, preserve the printed header order and associate each value only with its explicit column.
If OCR layout does not make a requested quality-level column unambiguous, report insufficient evidence.
If the evidence is insufficient, state that clearly.
Do not make acceptance/rejection decisions. Do not create engineering rules.
Answer in the language of the question and cite supporting chunk IDs in square brackets."""

ANALYTICAL_SYSTEM = """You are an evidence-grounded analytical assistant for technical drilling reports.
Synthesize across the supplied evidence; do not answer from one record when the question asks for comparison,
highest/lowest, totals, summaries, or where problems occurred most. Use only supplied evidence and never use
outside petroleum or geological knowledge. Distinguish hole section from formation. Do not claim a formation
mapping unless the evidence explicitly supports it. Distinguish event counts from hours/downtime. Preserve exact
numbers, units, hole sizes, and technical terms. Cite only the supplied evidence IDs in the form [S1], [S2].
If evidence supports only a section-level conclusion, say so explicitly. If evidence is insufficient, say so."""

FORMATTER_SYSTEM = """You normalize a supplied draft into one compact JSON object only.
Do not add facts. Do not rewrite technical values. Use only allowed citation chunk IDs.
Do not complete a truncated draft or infer a table-column association that the draft does not explicitly support.
The keys must be exactly: answer, citation_chunk_ids, confidence.
confidence must be high, medium, low, or insufficient."""


def reasoning_prompt(question: str, evidence_blocks: list[dict]) -> str:
    rendered = []
    for index, block in enumerate(evidence_blocks, 1):
        rendered.append(
            f"EVIDENCE {index}\nchunk_id: {block['chunk_id']}\n"
            f"citation: {block['citation']}\ncontent_type: {block['content_type']}\n"
            f"text:\n{block['text']}"
        )
    return f"QUESTION:\n{question}\n\nRETRIEVED EVIDENCE:\n\n" + "\n\n".join(rendered)


def analytical_prompt(question: str, evidence_package: str) -> str:
    return (
        "MODE: ANALYTICAL_SYNTHESIS\n"
        "Answer compactly with: (1) direct conclusion, (2) main supporting problems or metrics, "
        "and (3) section-versus-formation limitation when relevant.\n\n"
        f"QUESTION:\n{question}\n\nANALYTICAL EVIDENCE PACKAGE:\n{evidence_package}"
    )


def formatter_prompt(question: str, draft: str, standard: str, allowed_citations: list[dict]) -> str:
    schema = {
        "answer": "evidence-grounded answer or insufficient-evidence statement",
        "citation_chunk_ids": ["one or more chunk_id values copied from ALLOWED CITATIONS"],
        "confidence": "high|medium|low|insufficient",
    }
    return (
        f"QUESTION:\n{question}\n\nDRAFT FROM REASONING MODEL:\n{draft}\n\n"
        f"STANDARD:\n{standard}\n\nALLOWED CITATIONS:\n{json.dumps([x['chunk_id'] for x in allowed_citations], ensure_ascii=False)}\n\n"
        f"REQUIRED JSON SHAPE:\n{json.dumps(schema, ensure_ascii=False)}"
    )
