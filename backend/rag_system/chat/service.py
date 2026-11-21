from __future__ import annotations
import re
import json
from .query_rewriter import QueryRewriter
from .repository import ChatRepository
from .schemas import ChatResponse, Source
from rag_system.answering.core import AnsweringCore, SafeFollowupRewriter

INSUFFICIENT='اطلاعات کافی برای پاسخ در اسناد موجود پیدا نشد.'
FA_QUERY_BRIDGE = {"عمق کل": "total depth", "عمق": "depth", "آب": "water", "جدول": "table", "لوله جداری": "casing", "کفشک": "shoe", "فشار": "pressure", "بیشترین": "most highest", "کمترین": "least lowest", "مشکلات": "problems issues", "مشکل": "problem issue", "کدام": "which", "لایه": "layer section formation", "سازند": "formation", "بخش": "section", "اتفاق": "occurred events", "مقایسه": "compare", "مجموع": "total", "گیرکردن": "stuck pipe", "رشته": "string", "حفاری": "drilling"}

def _user_answer(text: str) -> str:
    """Remove internal citation/audit scaffolding from the visible answer only."""
    value = re.sub(r"\s*\[S\d+\]", "", str(text or ""))
    value = re.sub(r"^بر اساس رخدادهای ثبت‌شده(?: و شواهد بازیابی‌شده)?،?\s*", "", value)
    value = value.replace("برای نسبت‌دادن قطعی همه رخدادها به یک سازند واحد، نگاشت صریح کافی در شواهد بازیابی‌شده وجود ندارد.",
                          "گزارش اطلاعات کافی برای نسبت دادن قطعی همه این مشکلات به یک سازند مشخص ارائه نمی‌کند.")
    return re.sub(r"\s{2,}", " ", value).strip()

def _source_excerpt(e) -> str:
    """Return a compact human-readable source preview; raw table JSON stays internal."""
    raw = str(getattr(e, "text", getattr(e, "content", "")) or "")
    if e.metadata.get("content_type") != "table_segment":
        return re.sub(r"\s+", " ", raw)[:1200]
    try:
        cells = [str(x).replace("\\n", " ").replace("\n", " ").strip() for x in json.loads(raw).get("raw_cell_values", [])]
    except (ValueError, TypeError):
        return "جدول ساختاریافته — جزئیات در پاسخ داخلی نگهداری می‌شود."
    lines = []
    for i in range(len(cells) - 3):
        if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*\([^)]*\))?", cells[i + 1]):
            lines.append(f"عملیات: {cells[i + 3]}\nمدت: {cells[i + 1]} ساعت\nشرح: {cells[i + 2]}")
    if not lines:
        pairs = [f"{cells[i]}: {cells[i + 1]}" for i in range(0, min(len(cells) - 1, 12), 2)]
        lines = ["\n".join(pairs)]
    return "\n\n".join(lines[:4])[:1200]
class ChatService:
    def __init__(self, repository: ChatRepository, retrieval):
        self.repository=repository; self.retrieval=retrieval; self.rewriter=QueryRewriter(); self.safe_rewriter=SafeFollowupRewriter()
    async def answer(self, conversation_id: str | None, query: str, document_ids=None, language: str = "auto"):
        cid=conversation_id or self.repository.create_conversation(query[:80])
        history=self.repository.messages(cid)
        safe=self.safe_rewriter.rewrite(query,[{"role":x.get("role"),"content":x.get("content"),"document_id":(document_ids[0] if document_ids else "")} for x in history],document_ids[0] if document_ids else "")
        if safe.needs_clarification:
            text="Please clarify which source, value, or region you mean." if language != "fa" else "لطفاً مشخص کنید منظور شما کدام منبع، مقدار یا ناحیه است."
            self.repository.add_message(cid,'user',query); mid=self.repository.add_message(cid,'assistant',text,[])
            return ChatResponse(conversation_id=cid,message_id=mid,answer=text,sources=[],language="fa" if language=="fa" else "en",answer_mode="AMBIGUOUS_FOLLOWUP",insufficient_evidence=True,clarification_required=True,llama_used=False)
        rewritten=safe.query
        for fa_term, en_term in FA_QUERY_BRIDGE.items():
            if fa_term in rewritten:
                rewritten = f"{rewritten} {en_term}"
        if rewritten == query:
            rewritten=self.rewriter.rewrite(query,history)
        result=await self.retrieval.query(rewritten, standard_id=None)
        allowed=set(document_ids or [])
        evidence=[e for e in result.evidence if not allowed or e.metadata.get('document_id',e.document) in allowed]
        # LLMEnhancedRetrievalService owns generation and backend validation.
        # Do not pass its grounded answer back through the legacy deterministic
        # AnsweringCore; that path is retained only for explicit legacy callers.
        if result.answer is not None:
            answer = result.answer
            cited_ids = {citation.chunk_id for citation in answer.citations}
            source_evidence = [e for e in evidence if e.chunk_id in cited_ids] or evidence
            sources=[Source(document_id=e.document,filename=e.metadata.get('filename',e.document),
                             page=int(e.metadata.get('pdf_page_start',0) or 0),
                             section=e.metadata.get('clause_title'),chunk_id=e.chunk_id,
                             score=float(getattr(e,'score',0.0)),excerpt=_source_excerpt(e),
                             modality=str(e.metadata.get('modality',e.metadata.get('content_type','TEXT'))),
                             record_id=e.metadata.get('record_id')) for e in source_evidence]
            is_fa=language == "fa" or (language == "auto" and any('\u0600' <= c <= '\u06ff' for c in query))
            text=_user_answer(answer.answer)
            self.repository.add_message(cid,'user',query); mid=self.repository.add_message(cid,'assistant',text,[s.model_dump() for s in sources])
            return ChatResponse(conversation_id=cid,message_id=mid,answer=text,sources=sources,
                                language="fa" if is_fa else "en",answer_mode="LLM_GROUNDED",
                                insufficient_evidence=answer.confidence == "insufficient",
                                clarification_required=False,llama_used=True)
        # Stage-5 may return a broad candidate set; select the authoritative
        # modality first for structured or visual questions while retaining text
        # evidence as secondary context.
        low=rewritten.casefold()
        want_visual=any(t in low for t in ("chart", "plot", "visual", "relationship", "shown"))
        analytical=bool(re.search(r"بیشترین|کمترین|مشکل|کدام|most|least|problem|compare|highest|frequent|downtime", query, re.I))
        want_table=not want_visual and any(t in low for t in ("table", "depth", "date", "status", "casing", "duration", "water", "rkb", "md", "tvd"))
        preferred="VISUAL" if want_visual else "TABLE" if want_table else None
        if preferred or analytical:
            terms={t for t in re.findall(r"[a-z0-9]+",low) if len(t)>2}
            def relevance(e):
                text=str(getattr(e,"text","") or getattr(e,"content","")).casefold()
                overlap=sum(t in text for t in terms)
                boosts=0
                if "total depth" in low and "total depth" in text: boosts+=8
                if "tvd" in low and "tvd" in text: boosts+=3
                if "rop" in low and "rop" in text: boosts+=8
                if analytical and "main problem" in text: boosts+=20
                if analytical and "12 1/4" in text: boosts+=16
                if analytical and any(t in text for t in ("stuck", "npt", "loss", "repair")): boosts+=6
                return (0 if preferred and e.metadata.get("modality")==preferred else 1, -(overlap+boosts), float(getattr(e,"score",0)))
            evidence=sorted(evidence,key=relevance)[:20 if analytical else 10]
        else:
            evidence=evidence[:10]
        sources=[Source(document_id=e.metadata.get('document_id',e.document),filename=e.metadata.get('filename',e.document),page=int(e.metadata.get('pdf_page_start',0)),section=e.metadata.get('clause_title'),chunk_id=e.chunk_id,score=float(getattr(e,'score',0.0)),excerpt=_source_excerpt(e),modality=str(e.metadata.get('modality','TEXT')),record_id=e.metadata.get('record_id')) for e in evidence]
        records=[]
        for e in evidence:
            meta=e.metadata
            modality="TABLE" if meta.get("content_type")=="table_segment" else ("VISUAL" if meta.get("content_type")=="visual" else "TEXT")
            records.append({"record_id":f"{modality}::drilldoc_default__{e.chunk_id}","document_id":e.document,"source_filename":meta.get("filename",e.document),"modality":modality,"content_type":meta.get("content_type","text_chunk"),"pdf_page_number":meta.get("pdf_page_start"),"source_pages":[meta.get("pdf_page_start")],"section":meta.get("clause_title"),"retrieval_text":getattr(e,"text","")})
        class StaticRetriever:
            def __init__(self, rows): self.rows=rows
            def retrieve(self, _query, document_id, top_k=10): return [x for x in self.rows if x["document_id"]==document_id][:top_k]
        class NoLlama:
            def generate(self, *args, **kwargs): return {"answer_text":"","claims":[],"insufficient_evidence":True}
        core=AnsweringCore(StaticRetriever(records),NoLlama())
        core_result=core.answer(query,(document_ids or ["drilldoc_default"])[0],rewritten)
        text=core_result.answer_text
        is_fa=language == "fa" or (language == "auto" and any('\u0600' <= c <= '\u06ff' for c in query))
        self.repository.add_message(cid,'user',query); mid=self.repository.add_message(cid,'assistant',text,[s.model_dump() for s in sources])
        return ChatResponse(conversation_id=cid,message_id=mid,answer=text,sources=sources,language="fa" if is_fa else "en",answer_mode=core_result.answer_mode,insufficient_evidence=core_result.insufficient_evidence,clarification_required=False,llama_used=False)
