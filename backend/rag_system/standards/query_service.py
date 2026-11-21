from __future__ import annotations

import asyncio

from rag_system.api.schemas import EvidenceItem, RetrievalFilters, RetrievalResponse
from rag_system.api.service import InvalidQuery, RetrievalServiceProtocol, RetrievalUnavailable

from .registry import JsonStandardRegistry, StandardNotFound
from .schemas import StandardEntry, StandardStatus


class StandardAwareRetrievalService:
    """Routes to an existing per-standard retrieval service; it does not rerank within a standard."""

    def __init__(self, registry: JsonStandardRegistry, services: dict[str, RetrievalServiceProtocol]):
        self.registry, self.services = registry, services

    async def query(self, query: str, top_k: int | None = None, filters: RetrievalFilters | None = None,
                    standard_id: str | None = None) -> RetrievalResponse:
        targets = self._targets(standard_id)
        responses = await asyncio.gather(*(self._query_standard(entry, query, top_k, filters) for entry in targets))
        evidence = [item for response in responses for item in response.evidence]
        evidence.sort(key=lambda item: (item.score, item.chunk_id))
        limit = top_k if top_k is not None else max((len(response.evidence) for response in responses), default=0)
        return RetrievalResponse(query=query.strip(), evidence=evidence[:limit])

    def _targets(self, standard_id: str | None) -> list[StandardEntry]:
        if standard_id:
            try:
                entry = self.registry.get(standard_id)
            except StandardNotFound as exc:
                raise InvalidQuery(f"unknown standard_id: {standard_id}") from exc
            if entry.status != StandardStatus.active:
                raise InvalidQuery(f"standard is not active: {standard_id}")
            return [entry]
        active = self.registry.active()
        if not active:
            raise RetrievalUnavailable("no active standards are configured")
        return active

    async def _query_standard(self, entry: StandardEntry, query: str, top_k: int | None,
                              filters: RetrievalFilters | None) -> RetrievalResponse:
        service = self.services.get(entry.collection_id or "")
        if service is None:
            raise RetrievalUnavailable(f"retrieval service unavailable for standard: {entry.standard_id}")
        response = await service.query(query, top_k, filters)
        enriched = [self._enrich(item, entry) for item in response.evidence]
        return RetrievalResponse(query=response.query, evidence=enriched)

    @staticmethod
    def _enrich(item: EvidenceItem, entry: StandardEntry) -> EvidenceItem:
        citation = item.citation
        metadata = dict(item.metadata)
        metadata.update({
            "standard_id": entry.standard_id,
            "standard_name": entry.name,
            "version": entry.version,
            "document": item.document,
            "page": citation.get("printed_page_start", ""),
            "table": citation.get("table", ""),
            "clause": citation.get("clause", ""),
            "chunk_id": item.chunk_id,
        })
        return item.model_copy(update={"metadata": metadata, "citation": dict(citation)})

    def close(self) -> None:
        seen = set()
        for service in self.services.values():
            if id(service) in seen:
                continue
            seen.add(id(service))
            close = getattr(service, "close", None)
            if callable(close):
                close()
