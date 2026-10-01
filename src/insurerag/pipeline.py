"""Etape 7 - le pipeline qui relie recherche + generation.

On peut passer notre propre retriever / generator, ca sert pour les tests
ou si on veut changer de LLM plus tard.
"""

from __future__ import annotations

from functools import lru_cache

from config import EXTRACT_CHARS, LLM_MODEL, OLLAMA_HOST, TOP_K_RERANK
from .generation import NO_ANSWER, OllamaGenerator
from .retrieval import HybridRetriever
from .schemas import AskResult, Source


class RAGPipeline:
    """Le pipeline complet : question -> extraits -> reponse avec sources."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: OllamaGenerator | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or OllamaGenerator()

    def ask(self, question: str, top_k: int = TOP_K_RERANK) -> AskResult:
        question = (question or "").strip()
        if not question:
            return AskResult(question=question, answer="", sources=[])

        hits = self.retriever.retrieve(question, top_k=top_k)
        if not hits:
            return AskResult(question=question, answer=NO_ANSWER, sources=[])

        chunks = self.retriever.get_chunks([h.chunk_id for h in hits])
        if not chunks:
            return AskResult(question=question, answer=NO_ANSWER, sources=[])

        answer = self.generator.answer(question, chunks)

        # hits c'est des objets Hit, il faut prendre hit.score et pas hit
        sources = [
            Source(
                ref=i,
                source=chunk.source,
                pages=chunk.pages,
                headings=chunk.headings,
                score=float(hit.score),
                extract=chunk.raw_text[:EXTRACT_CHARS]
                + ("..." if len(chunk.raw_text) > EXTRACT_CHARS else ""),
            )
            for i, (chunk, hit) in enumerate(zip(chunks, hits), start=1)
        ]

        return AskResult(question=question, answer=answer, sources=sources)

    def close(self) -> None:
        self.retriever.close()


@lru_cache(maxsize=1)
def get_pipeline(
    use_rerank: bool = True, model: str = LLM_MODEL, host: str = OLLAMA_HOST
) -> RAGPipeline:
    """Le pipeline est garde en cache pour pas recharger les modeles a chaque
    rerun de streamlit."""
    return RAGPipeline(
        retriever=HybridRetriever(use_rerank=use_rerank),
        generator=OllamaGenerator(model=model, host=host),
    )
