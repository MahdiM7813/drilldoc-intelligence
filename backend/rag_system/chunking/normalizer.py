from __future__ import annotations

import re
import unicodedata


WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)


def normalize_for_retrieval(text: str) -> str:
    """Conservative normalization; technical values and symbols remain untouched."""
    value = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()).strip()


def token_estimate(text: str) -> int:
    """Deterministic offline estimate: Unicode words and punctuation tokens."""
    return max(1, len(WORD_RE.findall(text)))


def split_paragraphs(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    blocks = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [x.rstrip() for x in text.splitlines()]
    groups: list[list[str]] = []
    active: list[str] = []
    for line in lines:
        is_boundary = bool(re.match(r"^\s*(?:[a-z]\)|\d+[.)]|—|NOTE\s+\d*\b)", line, re.I))
        if is_boundary and active:
            groups.append(active); active = []
        active.append(line)
    if active:
        groups.append(active)
    return ["\n".join(x).strip() for x in groups if any(y.strip() for y in x)]


def clean_page_furniture(text: str, document_label: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    warnings: list[str] = []
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in {document_label, "ISO 5817:2014(E)"}:
            warnings.append("probable header/footer contamination removed")
            continue
        if re.fullmatch(r"(?:[ivxlcdm]+|\d+)", stripped, re.I):
            warnings.append("probable header/footer contamination removed")
            continue
        if re.match(r"^©\s+ISO\b", stripped, re.I):
            warnings.append("probable header/footer contamination removed")
            continue
        kept.append(line)
    return "\n".join(kept).strip(), sorted(set(warnings))
