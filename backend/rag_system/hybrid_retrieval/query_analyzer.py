from __future__ import annotations
import re
import unicodedata
from rag_system.retrieval.query_processor import process_query
from .schemas import QueryAnalysis

DOMAIN_TERMS = {
    "عمق نهایی": ("total depth", "TD", "final depth", "well summary", "general information"),
    "مخزن هدف": ("target reservoir", "reservoir", "target formation", "general information"),
    "عمق آب": ("water depth", "depth of water", "general information"),
    "مدت زمان چاه": ("well duration", "duration", "general information"),
    "مشکلات حفاری": ("drilling problems", "drilling problems occurred", "NPT", "non productive time", "well summary"),
    "در حفاری چه مشکلاتی": ("drilling problems", "NPT", "non productive time", "well summary"),
    "در حین حفاری": ("drilling problems", "NPT", "non productive time", "well summary"),
    "سازند": ("formation", "formations drilled", "section"),
    "سازند": ("formation", "formations drilled"),
    "بخش": ("section",),
    "عدم ذوب": ("lack of fusion", "incomplete fusion", "401"),
    "تخلخل": ("porosity", "gas pore", "surface pore", "pore"),
    "سطح کیفی": ("quality level",), "سطح کيفی": ("quality level",),
    "سطح تصویرشده": ("projected area",), "سطح تصويرشده": ("projected area",),
}

def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"[^\w\u0600-\u06ff.]+", " ", value).strip()

def tokenize(value: str) -> tuple[str, ...]:
    stopwords = {"what", "was", "were", "is", "are", "the", "of", "a", "an", "how", "which", "in", "on", "for", "to", "and"}
    return tuple(token for token in normalize_text(value).split() if token not in stopwords and (len(token) > 1 or token.isalnum()))

class QueryAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        original = process_query(query); normalized = normalize_text(original)
        expansions, phrases = [], []
        for persian, terms in DOMAIN_TERMS.items():
            if normalize_text(persian) in normalized:
                expansions.extend(terms); phrases.extend(terms)
        if "total depth" in normalized:
            expansions.extend(("TD", "well summary")); phrases.append("total depth")
        if "target reservoir" in normalized:
            expansions.extend(("reservoir", "general information")); phrases.append("target reservoir")
        if "water depth" in normalized:
            expansions.extend(("general information",)); phrases.append("water depth")
        if "drilling problems" in normalized:
            expansions.extend(("NPT", "well summary")); phrases.append("drilling problems")
        for phrase in ("lack of fusion", "quality level", "projected area", "gas pore", "surface pore"):
            if phrase in normalized: phrases.append(phrase)
        expanded = " ".join([original, *expansions]).strip()
        intent = "definition" if any(x in normalized for x in ("what is", "definition", "تعریف", "چیست")) else "general"
        language = "fa" if re.search(r"[\u0600-\u06ff]", original) else "en"
        return QueryAnalysis(original, expanded, tokenize(expanded), tuple(dict.fromkeys(phrases)), language, intent,
                             ("definition",) if intent == "definition" else ())
