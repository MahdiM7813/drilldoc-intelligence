from __future__ import annotations
class QueryRewriter:
    """Small deterministic rewrite layer; an LLM rewrite provider can be injected later."""
    def rewrite(self, query: str, history: list[dict]) -> str:
        value=query.strip()
        if not history: return value
        pronouns=('آن','این','it','that','this','its','همان','او')
        if any(token.lower() in value.lower().split() for token in pronouns):
            previous=next((m['content'] for m in reversed(history) if m['role']=='user'), '')
            if value.lower().strip().startswith('what was its ') and previous:
                subject = value.strip()[len('What was its '):].rstrip('?').strip()
                return f"What was the {subject} at the context of: {previous}"
            if 'آن' in value and previous:
                subject = value.replace('آن', '').replace('؟', '').strip()
                return f"{subject} با توجه به: {previous}"
            return f"{previous}\nFollow-up question: {value}"
        return value
