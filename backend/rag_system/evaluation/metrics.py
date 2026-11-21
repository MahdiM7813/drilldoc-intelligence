"""Small, deterministic metrics for synthetic RAG evaluation."""
from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence


def retrieval_metrics(ranked_ids: Sequence[str], relevant_ids: Iterable[str], *, k_values: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """Compute recall@k, precision@5, reciprocal rank and binary nDCG@5.

    Relevance is binary and chunk-ID based. Duplicate returned IDs count once at
    their first rank. Recall denominator is all annotated relevant IDs.
    """
    relevant = set(relevant_ids)
    ranked = list(dict.fromkeys(ranked_ids))
    out: dict[str, float] = {}
    for k in k_values:
        hits = sum(x in relevant for x in ranked[:k])
        out[f"recall@{k}"] = hits / len(relevant) if relevant else 0.0
    k = 5
    out["precision@5"] = sum(x in relevant for x in ranked[:k]) / k
    out["mrr"] = 1.0 / next((i for i, x in enumerate(ranked, 1) if x in relevant), math.inf)
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], 1) if x in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    out["ndcg@5"] = dcg / idcg if idcg else 0.0
    return out


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w.]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def numerical_matches(answer: str, expected_value: float, expected_unit: str) -> dict[str, bool]:
    """Conservative value/unit check; conversions and tolerance are not inferred."""
    normalized = answer.casefold().replace(",", ".")
    numbers = [float(x) for x in re.findall(r"(?<![\w-])\d+(?:\.\d+)?(?!\w)", normalized)]
    unit_present = bool(re.search(rf"(?<!\w){re.escape(expected_unit.casefold())}(?!\w)", normalized))
    return {"value_correct": any(math.isclose(x, expected_value, rel_tol=0, abs_tol=1e-9) for x in numbers),
            "unit_correct": unit_present}


def answer_match_metrics(answer: str, expected: str) -> dict[str, bool | None]:
    """Literal normalized/numeric/unit checks; these do not score semantic faithfulness."""
    number_pattern = r"(?<![\w-])\d+(?:[.,]\d+)?(?!\w)"
    unit_pattern = r"(?<!\w)(?:MPa|mL/min|L/min|minutes|minute|C|L)(?!\w)"
    expected_numbers = {x.replace(",", ".") for x in re.findall(number_pattern, expected)}
    answer_numbers = {x.replace(",", ".") for x in re.findall(number_pattern, answer)}
    expected_units = {x.casefold() for x in re.findall(unit_pattern, expected, re.I)}
    answer_units = {x.casefold() for x in re.findall(unit_pattern, answer, re.I)}
    return {
        "normalized_exact_match": bool(expected) and normalize_answer(answer) == normalize_answer(expected),
        "numeric_value_match": expected_numbers.issubset(answer_numbers) if expected_numbers else None,
        "unit_match": expected_units.issubset(answer_units) if expected_units else None,
    }


def abstention_marker(answer: str) -> bool:
    """Detect explicit uncertainty wording; confidence calibration is reported separately."""
    return bool(re.search(
        r"insufficient evidence|not specified|not provided|not recorded|does not state|does not mention|"
        r"cannot determine|no information|no evidence|no details|intentionally not specified",
        answer, re.I,
    ))


def citation_metrics(cited_ids: Sequence[str], relevant_ids: Iterable[str], evidence_ids: Iterable[str]) -> dict[str, float]:
    cited = list(dict.fromkeys(cited_ids))
    relevant, available = set(relevant_ids), set(evidence_ids)
    valid = [x for x in cited if x in available]
    return {
        "citation_precision": sum(x in relevant for x in valid) / len(valid) if valid else 0.0,
        "citation_coverage": len(set(valid) & relevant) / len(relevant) if relevant else 0.0,
        "citation_validity": len(valid) / len(cited) if cited else 0.0,
    }

