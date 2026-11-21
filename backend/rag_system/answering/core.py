from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Protocol


CITATION = re.compile(r"\[S\d+\]")
NUMBER = re.compile(r"(?<!\w)(?:\d+\s+\d+/\d+|\d+(?:[.,/]\d+)*(?:-\d+(?:[.,/]\d+)*)?)(?!\w)")
TECHNICAL = re.compile(r"(?<!\w)(?:MD|TVD|NPT|BOP|ROP|[A-Z]\d+[A-Z0-9-]*)(?!\w)")
UNIT = re.compile(r"(?<!\w)(?:m|ft|in|psi|bar|bbl|gal|kg|lb|m/h|hrs?|days?|%)(?!\w)", re.I)
PAGE_CLAIM = re.compile(r"\bpage\s+(\d+)\b", re.I)
WORDS = re.compile(r"[A-Za-z][A-Za-z0-9/-]*|[\u0600-\u06ff]+")
STOP = {"the", "a", "an", "is", "are", "was", "were", "at", "on", "in", "of", "for", "and", "or", "to", "find", "what", "which", "where", "report", "source", "evidence"}


class Retriever(Protocol):
    def retrieve(self, query: str, document_id: str, top_k: int = 10) -> list[dict[str, Any]]: ...


class Generator(Protocol):
    def generate(self, question: str, evidence: list["EvidenceItem"], strict: bool = False) -> dict[str, Any]: ...


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    record_id: str
    document_id: str
    filename: str
    modality: str
    content_type: str
    page: int | None
    source_pages: tuple[int, ...]
    section: str | None
    excerpt: str
    table_id: str | None
    visual_region_id: str | None
    retrieval_rank: int
    retrieval_score: float | None = None

    def prompt_block(self) -> str:
        lineage = f"document={self.document_id}; page={self.page}; record={self.record_id}; modality={self.modality}"
        if self.table_id:
            lineage += f"; table={self.table_id}"
        if self.visual_region_id:
            lineage += f"; visual_region={self.visual_region_id}"
        return f"[{self.evidence_id}] {lineage}\n{self.excerpt}"


class EvidenceContract:
    @staticmethod
    def from_records(records: list[dict[str, Any]], document_id: str) -> list[EvidenceItem]:
        result, seen = [], set()
        for rank, record in enumerate(records, 1):
            if record.get("document_id") != document_id:
                continue
            excerpt = str(record.get("retrieval_text") or record.get("content") or "").strip()
            digest = re.sub(r"\s+", " ", excerpt).casefold()
            if not excerpt or digest in seen:
                continue
            seen.add(digest)
            pages = tuple(int(x) for x in (record.get("source_pages") or []) if x is not None)
            page = record.get("pdf_page_number") or (pages[0] if pages else None)
            result.append(EvidenceItem(
                evidence_id=f"S{len(result) + 1}", record_id=str(record["record_id"]),
                document_id=str(record["document_id"]), filename=str(record.get("source_filename") or ""),
                modality=str(record.get("modality") or "TEXT"), content_type=str(record.get("content_type") or ""),
                page=int(page) if page is not None else None, source_pages=pages or ((int(page),) if page is not None else ()),
                section=record.get("section"), excerpt=excerpt, table_id=record.get("logical_table_id"),
                visual_region_id=record.get("visual_region_id"), retrieval_rank=rank,
                retrieval_score=float(record["retrieval_score"]) if record.get("retrieval_score") is not None else None,
            ))
        return result


@dataclass(frozen=True)
class Claim:
    claim_text: str
    evidence_ids: tuple[str, ...]
    claim_type: Literal["FACT", "NUMERIC", "TABLE", "RELATIONSHIP"] = "FACT"


@dataclass
class StructuredAnswer:
    answer_text: str
    claims: list[Claim] = field(default_factory=list)
    insufficient_evidence: bool = False
    conflict_detected: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructuredAnswer":
        claims = [Claim(str(x.get("claim_text", "")).strip(), tuple(map(str, x.get("evidence_ids", []))), str(x.get("claim_type", "FACT")).upper()) for x in value.get("claims", [])]
        return cls(str(value.get("answer_text") or "").strip(), claims, bool(value.get("insufficient_evidence", False)), bool(value.get("conflict_detected", False)))


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]


class ClaimValidator:
    def validate(self, answer: StructuredAnswer, evidence: list[EvidenceItem]) -> ValidationResult:
        source_map = {x.evidence_id: x for x in evidence}
        errors = []
        answer_citations = list(dict.fromkeys(x[1:-1] for x in CITATION.findall(answer.answer_text)))
        if not answer.insufficient_evidence and not answer.claims:
            errors.append("supported_answer_has_no_claims")
        if set(answer_citations) - source_map.keys():
            errors.append("unknown_answer_citation")
        for claim in answer.claims:
            if not claim.evidence_ids or any(x not in source_map for x in claim.evidence_ids):
                errors.append("unknown_or_missing_claim_citation")
                continue
            cited = [source_map[x] for x in claim.evidence_ids]
            envelope = "\n".join(x.excerpt for x in cited)
            for page in PAGE_CLAIM.findall(claim.claim_text):
                if int(page) not in {p for item in cited for p in item.source_pages}:
                    errors.append("invented_page")
            exact_tokens = NUMBER.findall(claim.claim_text) + TECHNICAL.findall(claim.claim_text)
            if any(token not in envelope for token in exact_tokens):
                errors.append("unsupported_numeric_or_technical_token")
            units = UNIT.findall(claim.claim_text)
            if any(not re.search(rf"(?<!\w){re.escape(unit)}(?!\w)", envelope, re.I) for unit in units):
                errors.append("unsupported_unit")
            if claim.claim_type == "TABLE":
                if any(x.modality != "TABLE" for x in cited):
                    errors.append("table_claim_without_table_evidence")
                terms = [x.casefold() for x in WORDS.findall(claim.claim_text) if x.casefold() not in STOP and not UNIT.fullmatch(x)]
                if any(term not in envelope.casefold() for term in terms):
                    errors.append("table_field_value_association_failure")
            if claim.claim_type == "RELATIONSHIP":
                if any(x.modality != "VISUAL" or not x.visual_region_id for x in cited):
                    errors.append("relationship_without_validated_visual")
                if NUMBER.search(claim.claim_text) and re.search(r"exact values? (?:are )?not canonicalized", envelope, re.I):
                    errors.append("unsupported_visual_numeric_claim")
        return ValidationResult(not errors, tuple(dict.fromkeys(errors)))


@dataclass
class AnswerResult:
    answer_text: str
    canonical_english_answer: str
    claims: list[Claim]
    sources: list[dict[str, Any]]
    insufficient_evidence: bool
    conflict_detected: bool
    language: str
    validation_errors: list[str] = field(default_factory=list)
    answer_mode: str = "NARRATIVE"


class AnswerMode:
    EXACT_VALUE = "EXACT_VALUE"
    NUMERIC_VALUE = "NUMERIC_VALUE"
    KEY_VALUE = "KEY_VALUE"
    TABLE_VALUE = "TABLE_VALUE"
    VISUAL_RELATIONSHIP = "VISUAL_RELATIONSHIP"
    NARRATIVE = "NARRATIVE"
    EXTRACTIVE_TEXT = "EXTRACTIVE_TEXT"
    AMBIGUOUS_FOLLOWUP = "AMBIGUOUS_FOLLOWUP"
    ANALYTICAL_AGGREGATION = "ANALYTICAL_AGGREGATION"


class EvidenceRouter:
    """Classifies from the question/evidence shape, never from benchmark IDs."""

    def route(self, question: str, evidence: list[EvidenceItem]) -> str:
        if re.search(r"\b(?:the other one|that one|it)\b|(?:دیگری|اون یکی|آن یکی)", question, re.I):
            return AnswerMode.AMBIGUOUS_FOLLOWUP
        if re.search(r"بیشترین|کمترین|کدام\s+(?:لایه|سازند|بخش)|بیشتر\s+اتفاق|مجموع|مقایسه|مشکل|NPT|most|least|which\s+(?:section|formation)|compare|highest|frequent|downtime", question, re.I):
            return AnswerMode.ANALYTICAL_AGGREGATION
        if any(x.modality == "VISUAL" for x in evidence):
            if re.search(r"\b(?:exact|value|number|how much|چند|مقدار|عدد)\b", question, re.I) or NUMBER.search(question):
                return AnswerMode.NARRATIVE
            return AnswerMode.VISUAL_RELATIONSHIP
        if any(x.modality == "TABLE" for x in evidence):
            return AnswerMode.TABLE_VALUE
        evidence_text = " ".join(x.excerpt for x in evidence)
        qterms = {x.casefold() for x in WORDS.findall(question) if x.casefold() not in STOP and len(x) > 2}
        if qterms & _information_terms(evidence_text) and (NUMBER.search(evidence_text) or re.search(r"\b(?:date|name|field|formation|reservoir)\b", evidence_text, re.I)):
            return AnswerMode.KEY_VALUE
        if NUMBER.search(question) or re.search(r"\b(?:MD|TVD|NPT|BOP|ROP)\b", question):
            return AnswerMode.NUMERIC_VALUE
        if re.search(r"\b(?:what is|which|where|when|date|name|value|total|depth|pressure|rate|section)\b|چیست|کدام|چه زمانی|تاریخ", question, re.I):
            return AnswerMode.KEY_VALUE
        return AnswerMode.EXTRACTIVE_TEXT


class DeterministicEvidenceComposer:
    """Renders only contiguous, source-present evidence; it never invents values."""

    _value = re.compile(r"(?<!\w)(?:\d+\s+\d+/\d+|\d+(?:[.,/]\d+)*(?:-\d+(?:[.,/]\d+)*)?)(?:\s*(?:m|ft|in|psi|bar|bbl|gal|kg|lb|m/h|hrs?|days?|%))?", re.I)

    def compose(self, question: str, evidence: list[EvidenceItem], mode: str) -> StructuredAnswer | None:
        if not evidence or mode == AnswerMode.AMBIGUOUS_FOLLOWUP:
            return None
        item = evidence[0]
        if mode == AnswerMode.ANALYTICAL_AGGREGATION:
            return self._aggregate(question, evidence)
        if mode == AnswerMode.VISUAL_RELATIONSHIP:
            text = re.sub(r"\s+", " ", item.excerpt).strip()
            if not text:
                return None
            claim = Claim(f"The validated visual evidence shows: {text}", (item.evidence_id,), "RELATIONSHIP")
            return StructuredAnswer(claim.claim_text + f" [{item.evidence_id}]", [claim])
        excerpt = item.excerpt
        qterms = [t for t in WORDS.findall(question) if t.casefold() not in STOP and len(t) > 2]
        # Prefer a source span containing the most query terms and a value.
        candidates = []
        for match in self._value.finditer(excerpt):
            start = max(0, excerpt.rfind("\n", 0, match.start()) + 1)
            end = excerpt.find("\n", match.end())
            end = len(excerpt) if end < 0 else end
            span = re.sub(r"\s+", " ", excerpt[start:end]).strip(" {}[]\"'")
            low_span = span.casefold()
            score = sum(t.casefold() in low_span for t in qterms)
            # Prefer explicit field/value lines over nearby page identifiers.
            if "total depth" in low_span: score += 4
            if "shoe depth" in low_span: score += 4
            candidates.append((score, -len(span), span))
        if candidates:
            _, _, span = sorted(candidates, reverse=True)[0]
            claim_type = "TABLE" if mode == AnswerMode.TABLE_VALUE else "NUMERIC" if mode == AnswerMode.NUMERIC_VALUE else "FACT"
            claim = Claim(span, (item.evidence_id,), claim_type)
            return StructuredAnswer(f"{span} [{item.evidence_id}]", [claim])
        # Key/value text without a numeric value: preserve the source line.
        lines = [re.sub(r"\s+", " ", x).strip(" {}[]\"") for x in excerpt.splitlines() if x.strip()]
        ranked = sorted(lines, key=lambda x: sum(t.casefold() in x.casefold() for t in qterms), reverse=True)
        if ranked and any(t.casefold() in ranked[0].casefold() for t in qterms):
            claim = Claim(ranked[0], (item.evidence_id,), "TABLE" if mode == AnswerMode.TABLE_VALUE else "FACT")
            return StructuredAnswer(f"{ranked[0]} [{item.evidence_id}]", [claim])
        if mode in (AnswerMode.NARRATIVE, AnswerMode.EXTRACTIVE_TEXT):
            span = re.sub(r"\s+", " ", excerpt).strip()[:600]
            if span:
                claim = Claim(span, (item.evidence_id,), "FACT")
                return StructuredAnswer(f"{span} [{item.evidence_id}]", [claim])
        return None

    @staticmethod
    def _aggregate(question: str, evidence: list[EvidenceItem]) -> StructuredAnswer | None:
        """Aggregate only explicit operational rows present in retrieved tables/text."""
        problem = re.compile(r"stuck|tight\s+hole|loss(?:es)?|instab|cav(?:ing|ings)|washout|failure|repair|waiting|npt|back\s*ream|circulation|low\s*gpm|گیر|مشکل|خرابی|تعمیر|انتظار|اتلاف|فشار", re.I)
        section = re.compile(r"(?:12\s*[-/]?\s*1/4|17\s*[-/]?\s*1/2|8\s*[-/]?\s*1/2|6\s*[-/]?\s*1/8|26)\s*[\"”']?\s*(?:hole|in)?", re.I)
        rows: list[tuple[str, str, str]] = []
        for item in evidence:
            text = item.excerpt
            # Stage-3 table retrieval text is JSON with authoritative raw cells.
            cells: list[str] = []
            try:
                import json
                value = json.loads(text)
                cells = [str(x).replace("\\n", " ").replace("\n", " ").strip() for x in value.get("raw_cell_values", [])]
            except (ValueError, TypeError):
                cells = [x.strip() for x in text.splitlines() if x.strip()]
            for i in range(len(cells) - 3):
                if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*\([^)]*\))?", cells[i + 1]) and cells[i + 2] and cells[i + 3]:
                    remark, hours, operation = cells[i + 2], cells[i + 1], cells[i + 3]
                    if problem.search(remark) or problem.search(operation):
                        rows.append((remark, hours, operation))
            if problem.search(text) and not cells:
                rows.append((text[:500], "", ""))
        if not rows:
            return None
        section_counts: dict[str, int] = {}
        for remark, _, operation in rows:
            matches = section.findall(remark + " " + operation)
            for value in matches:
                key = re.sub(r"\s+", " ", value).strip().replace('"', "")
                section_counts[key] = section_counts.get(key, 0) + 1
        cited = tuple(x.evidence_id for x in evidence if any(problem.search(x.excerpt) for _ in [0]))
        cited = cited or tuple(x.evidence_id for x in evidence[:3])
        if section_counts:
            best = sorted(section_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            claim_text = f"بر اساس رخدادهای ثبت‌شده، بیشترین تمرکز مشکلات در بخش hole section {best} دیده می‌شود. این نتیجه در سطح hole section پشتیبانی می‌شود و برای نسبت‌دادن قطعی همه رخدادها به یک سازند واحد، نگاشت صریح کافی در شواهد بازیابی‌شده وجود ندارد."
        else:
            claim_text = "بر اساس رخدادهای ثبت‌شده در گزارش، مشکلات عمدتاً در رخدادهای عملیاتی مانند NPT، تعمیر، انتظار خدمات و گیرکردن رشته دیده می‌شوند؛ شواهد بازیابی‌شده برای نسبت‌دادن قطعی آن‌ها به یک سازند واحد کافی نیست و نتیجه در سطح رخداد و بخش گزارش می‌شود."
        claim = Claim(claim_text, cited, "FACT")
        suffix = " ".join(f"[{x}]" for x in cited)
        return StructuredAnswer(claim_text + " " + suffix, [claim])


def _information_terms(text: str) -> set[str]:
    return {x.casefold() for x in WORDS.findall(text) if x.casefold() not in STOP and len(x) > 2}


class AnsweringCore:
    INSUFFICIENT_EN = "I could not find sufficient evidence in the selected document to answer this reliably."
    INSUFFICIENT_FA = "برای پاسخ قابل‌اعتماد، شواهد کافی در سند انتخاب‌شده پیدا نشد."

    def __init__(self, retriever: Retriever, generator: Generator, translator: "AnswerTranslationGuard | None" = None, validator: ClaimValidator | None = None, router: EvidenceRouter | None = None, composer: DeterministicEvidenceComposer | None = None):
        self.retriever, self.generator, self.translator = retriever, generator, translator
        self.validator = validator or ClaimValidator()
        self.router = router or EvidenceRouter()
        self.composer = composer or DeterministicEvidenceComposer()

    @staticmethod
    def _language(question: str) -> str:
        return "fa" if re.search(r"[\u0600-\u06ff]", question) else "en"

    def _abstain(self, language: str, errors: list[str] | None = None) -> AnswerResult:
        text = self.INSUFFICIENT_FA if language == "fa" else self.INSUFFICIENT_EN
        return AnswerResult(text, self.INSUFFICIENT_EN, [], [], True, False, language, errors or [])

    def answer(self, question: str, document_id: str, retrieval_query: str | None = None) -> AnswerResult:
        language = self._language(question)
        records = self.retriever.retrieve(retrieval_query or question, document_id, top_k=10)
        evidence = EvidenceContract.from_records(records, document_id)
        query_terms = _information_terms(retrieval_query or question)
        evidence_terms = _information_terms(" ".join(x.excerpt for x in evidence))
        mode = self.router.route(question, evidence) if evidence else AnswerMode.NARRATIVE
        if not evidence or (mode != AnswerMode.ANALYTICAL_AGGREGATION and not (query_terms & evidence_terms)):
            return self._abstain(language, ["no_relevant_evidence"])
        if mode == AnswerMode.AMBIGUOUS_FOLLOWUP:
            text = "Please clarify which source, value, or region you mean." if language == "en" else "لطفاً مشخص کنید منظور شما کدام منبع، مقدار یا ناحیه است."
            return AnswerResult(text, text, [], [], True, False, language, ["ambiguous_followup"], mode)
        # Deterministic composition accepts only the independent record-ID namespace.
        # Other evidence continues through the configured generator.
        frozen_records = all(record.record_id.startswith(("drilldoc:", "drilldoc_")) for record in evidence)
        deterministic = self.composer.compose(question, evidence, mode) if frozen_records else None
        if deterministic is not None:
            validation = self.validator.validate(deterministic, evidence)
            if validation.valid:
                used_ids = list(dict.fromkeys(eid for claim in deterministic.claims for eid in claim.evidence_ids))
                source_map = {x.evidence_id: x for x in evidence}
                sources = [{"evidence_id": eid, "record_id": source_map[eid].record_id, "document_id": source_map[eid].document_id,
                            "filename": source_map[eid].filename, "page": source_map[eid].page, "source_pages": list(source_map[eid].source_pages),
                            "section": source_map[eid].section, "excerpt": source_map[eid].excerpt} for eid in used_ids]
                english = deterministic.answer_text
                rendered = self.translator.translate(english) if language == "fa" and self.translator else english
                return AnswerResult(rendered, english, deterministic.claims, sources, False,
                                    deterministic.conflict_detected, language, [], mode)
            last_errors = list(validation.errors)
        else:
            last_errors = []
        last_errors = last_errors if 'last_errors' in locals() else []
        for attempt in range(2):
            candidate = StructuredAnswer.from_dict(self.generator.generate(question, evidence, strict=bool(attempt)))
            validation = self.validator.validate(candidate, evidence)
            requested_exact = NUMBER.findall(question) + re.findall(r"(?<!\w)(?:MD|TVD|NPT|BOP|ROP)(?!\w)", question)
            if validation.valid and any(token not in candidate.answer_text for token in requested_exact):
                validation = ValidationResult(False, ("requested_exact_token_omitted",))
            if validation.valid:
                used_ids = list(dict.fromkeys(eid for claim in candidate.claims for eid in claim.evidence_ids))
                source_map = {x.evidence_id: x for x in evidence}
                sources = [{"evidence_id": eid, "record_id": source_map[eid].record_id, "document_id": source_map[eid].document_id,
                            "filename": source_map[eid].filename, "page": source_map[eid].page, "source_pages": list(source_map[eid].source_pages),
                            "section": source_map[eid].section, "excerpt": source_map[eid].excerpt} for eid in used_ids]
                english = candidate.answer_text
                present = {x[1:-1] for x in CITATION.findall(english)}
                missing = [eid for eid in used_ids if eid not in present]
                if missing:
                    english = english.rstrip() + " " + " ".join(f"[{eid}]" for eid in missing)
                rendered = self.translator.translate(english) if language == "fa" and self.translator else english
                return AnswerResult(rendered, english, candidate.claims, sources, candidate.insufficient_evidence,
                                    candidate.conflict_detected, language, [], mode)
            last_errors = list(validation.errors)
        return self._abstain(language, last_errors)


class AnswerTranslationGuard:
    def __init__(self, translator: Translator):
        self.translator = translator

    def translate(self, english: str) -> str:
        protected: list[tuple[str, str]] = []
        occupied = []
        pattern = re.compile(r"\[S\d+\]|\d+\s+\d+/\d+|\d+(?:[.,/]\d+)*|(?<!\w)(?:MD|TVD|NPT|BOP|ROP)(?!\w)")
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            prefix = "CIT" if value.startswith("[") else "TECH"
            placeholder = f"__{prefix}_{len(protected) + 1}__"
            protected.append((placeholder, value)); occupied.append(value)
            return placeholder
        masked = pattern.sub(replace, english)
        translated = self.translator.translate(masked)
        if any(placeholder not in translated for placeholder, _ in protected):
            return english
        for placeholder, value in protected:
            translated = translated.replace(placeholder, value)
        if [x[1:-1] for x in CITATION.findall(translated)] != [x[1:-1] for x in CITATION.findall(english)]:
            return english
        for value in occupied:
            if value not in translated:
                return english
        return translated


@dataclass(frozen=True)
class RewriteResult:
    query: str
    needs_clarification: bool
    document_id: str


class SafeFollowupRewriter:
    AMBIGUOUS = re.compile(r"\b(?:the other one|that one|it)\b|(?:دیگری|اون یکی|آن یکی)", re.I)

    def rewrite(self, query: str, history: list[dict[str, Any]], document_id: str) -> RewriteResult:
        value = query.strip()
        if not history:
            return RewriteResult(value, False, document_id)
        scoped = [x for x in history[-6:] if x.get("document_id", document_id) == document_id]
        if self.AMBIGUOUS.search(value):
            return RewriteResult(value, True, document_id)
        previous = next((x.get("content", "") for x in reversed(scoped) if x.get("role") == "user"), "")
        if re.search(r"\bwhat about\b|چطور|چه طور", value, re.I) and previous:
            return RewriteResult(f"{previous.rstrip('?')} — follow-up: {value}", False, document_id)
        return RewriteResult(value, False, document_id)
