"""Etape 5 - la recherche : dense + BM25, fusion RRF, et reranking.

On renvoie des Hit (id + score), le texte on le relit apres dans les payloads
qdrant. Comme ca on depend pas d'une liste de chunks en memoire et l'app peut
tourner dans un autre processus que l'index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from config import (
    COLLECTION,
    DEVICE,
    QDRANT_DIR,
    RERANK_MODEL,
    RRF_K,
    TOP_K_BM25,
    TOP_K_DENSE,
    TOP_K_FUSION,
    TOP_K_RERANK,
)
from .embedding import Encoder
from .indexing import BM25Index, open_qdrant
from .schemas import Chunk, Hit


def rrf_fusion(
    rankings: list[list[Hit]], k: int = RRF_K, top_n: int = TOP_K_FUSION
) -> list[Hit]:
    """Reciprocal Rank Fusion : on additionne 1/(k+rang) pour chaque classement.

    Du coup un chunk bien classe par les 2 methodes remonte. C'est une fonction
    pure, on peut la tester tout seul.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [Hit(chunk_id=cid, score=score) for cid, score in ordered]


class HybridRetriever:
    """Charge tout a la demande : qdrant, bm25, l'encodeur et le reranker.

    Comme ca on peut creer le retriever sans attendre, et on paie le chargement
    qu'a la premiere question.
    """

    def __init__(
        self,
        collection: str = COLLECTION,
        device: str = DEVICE,
        use_rerank: bool = True,
        qdrant_path: Path = QDRANT_DIR,
    ) -> None:
        self.collection = collection
        self.device = device
        self.use_rerank = use_rerank
        self.qdrant_path = qdrant_path
        self._client: QdrantClient | None = None
        self._bm25: BM25Index | None = None
        self._encoder: Encoder | None = None
        self._reranker: CrossEncoder | None = None

    # -- les 4 trucs, charges a la demande --
    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = open_qdrant(self.qdrant_path)
        return self._client

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load()
        return self._bm25

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(device=self.device)
        return self._encoder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(
                RERANK_MODEL, device=self.device, max_length=512
            )
        return self._reranker

    # -- les etapes de la recherche --
    def dense_search(self, query: str, k: int = TOP_K_DENSE) -> list[Hit]:
        """La recherche par vecteurs (similarite cosinus)."""
        vector = self.encoder.encode_query(query).tolist()
        points = self.client.query_points(
            collection_name=self.collection, query=vector, limit=k, with_payload=False
        ).points
        # L'ID du point EST le chunk_id (UUID pose a l'indexation).
        return [Hit(chunk_id=str(p.id), score=float(p.score)) for p in points]

    def bm25_search(self, query: str, k: int = TOP_K_BM25) -> list[Hit]:
        """La recherche par mots."""
        return self.bm25.search(query, k=k)

    def hybrid_search(
        self,
        query: str,
        top_n: int = TOP_K_FUSION,
        k_dense: int = TOP_K_DENSE,
        k_bm25: int = TOP_K_BM25,
    ) -> list[Hit]:
        """On melange le dense et le bm25 avec RRF."""
        return rrf_fusion(
            [self.dense_search(query, k=k_dense), self.bm25_search(query, k=k_bm25)],
            top_n=top_n,
        )

    def rerank(
        self, query: str, candidates: list[Hit], top_k: int = TOP_K_RERANK
    ) -> list[Hit]:
        """Reranking des candidats avec le cross-encoder.

        On fait les paires avec les textes relus dans qdrant mais on garde
        l'id de chaque chunk, comme ca si un chunk a disparu de l'index les
        scores restent dans le bon ordre.
        """
        if not candidates:
            return []
        texts = self.get_texts([h.chunk_id for h in candidates])
        pairs = [(h.chunk_id, texts[h.chunk_id]) for h in candidates if h.chunk_id in texts]
        if not pairs:
            return []

        scores = self.reranker.predict(
            [(query, text) for _, text in pairs], batch_size=16, show_progress_bar=False
        )
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(chunk_id=pairs[i][0], score=float(scores[i])) for i in order
        ]

    def retrieve(
        self, query: str, top_k: int = TOP_K_RERANK, use_rerank: bool | None = None
    ) -> list[Hit]:
        """dense + bm25 -> RRF -> reranking (ou pas si on le desactive)."""
        if use_rerank is None:
            use_rerank = self.use_rerank
        fused = self.hybrid_search(query)
        if use_rerank:
            return self.rerank(query, fused, top_k=top_k)
        return fused[:top_k]

    # -- relire les contenus --
    def get_payloads(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Relit les payloads (texte, titres, pages, source) dans qdrant."""
        if not chunk_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.collection, ids=chunk_ids, with_payload=True
        )
        return {str(r.id): (r.payload or {}) for r in records}

    def get_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """Le texte (avec les titres) de chaque chunk_id."""
        return {
            cid: payload.get("text", "")
            for cid, payload in self.get_payloads(chunk_ids).items()
        }

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fabrique les Chunk a partir des payloads, dans l'ordre demande."""
        payloads = self.get_payloads(chunk_ids)
        chunks: list[Chunk] = []
        for cid in chunk_ids:
            payload = payloads.get(cid)
            if payload is None:
                continue
            chunks.append(
                Chunk(
                    id=cid,
                    text=payload.get("text", ""),
                    raw_text=payload.get("raw_text", ""),
                    headings=list(payload.get("headings") or []),
                    pages=list(payload.get("pages") or []),
                    source=payload.get("source", ""),
                )
            )
        return chunks

    def close(self) -> None:
        """Ferme qdrant, sinon le dossier reste locke."""
        if self._client is not None:
            self._client.close()
            self._client = None

