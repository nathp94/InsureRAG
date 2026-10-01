"""Etape 4 - l'indexation : qdrant pour le dense + BM25 pour le lexical.

Qdrant c'est la base vectorielle (en mode fichier, pas besoin de serveur).
BM25 ca Complete bien parce que sur les contrats y a plein de mots exacts
genre "franchise" ou "delai" que le dense rate parfois.

Les index sont sur disque et on utilise des uuid pour les chunks, pas des
positions dans une liste. Du coup l'app peut demarrer dans un autre
processus que celui qui a construit l'index (cas de streamlit).
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from config import BM25_PATH, COLLECTION, QDRANT_DIR, ensure_dirs

from .embedding import Encoder
from .schemas import Chunk, Hit


FR_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "en", "au", "aux",
    "à", "a", "ce", "ces", "cet", "cette", "que", "qui", "quoi", "dans", "par", "pour", "sur",
    "avec", "sans", "est", "sont", "être", "avoir", "il", "elle", "ils", "elles", "je", "vous",
    "nous", "se", "sa", "son", "ses", "leur", "leurs", "ne", "pas", "plus", "si", "ou", "y",
    "mon", "ma", "mes", "ton", "ta", "tes", "votre", "vos", "notre", "nos", "c", "s", "n", "qu",
}


def strip_accents(text: str) -> str:
    # on decompose en NFD et on vire les caracteres de categorie "Mn"
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def tokenize_fr(text: str) -> list[str]:
    """Minuscules, sans accents, sans mots vides, et on retire le "s" des pluriels.

    Le stemming est hyper basique mais ca suffit pour des contrats et ca evite
    de rajouter une dependance.
    """
    normalized = strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", normalized)
    stop = {strip_accents(w) for w in FR_STOPWORDS}
    return [
        t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens if t not in stop
    ]


# ---- l'index BM25 ----
class BM25Index:
    """BM25 + la liste des ids de chunks.

    rank_bm25 marche qu'avec des positions (des entiers), du coup on garde
    self.ids pour pouvoir retourner des uuid a la place.
    """

    def __init__(self, ids: list[str], bm25: BM25Okapi) -> None:
        self.ids = ids
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        corpus = [tokenize_fr(c.text) for c in chunks]
        if not corpus:
            # BM25Okapi divise par corpus_size, donc ZeroDivisionError sur vide
            return cls(ids=[], bm25=None)
        return cls(ids=[c.id for c in chunks], bm25=BM25Okapi(corpus))

    def search(self, query: str, k: int = 20) -> list[Hit]:
        """Les k meilleurs, on garde que ceux qui ont un score > 0."""
        if not self.ids or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize_fr(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            Hit(chunk_id=self.ids[int(i)], score=float(scores[i]))
            for i in top
            if scores[i] > 0
        ]

    def save(self, path: Path = BM25_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self.bm25}, f)
        return path

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> "BM25Index":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Index BM25 absent : {path}. Lancez `python scripts/build_index.py`."
            )
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(ids=data["ids"], bm25=data["bm25"])



# ---- la base vectorielle ----
def open_qdrant(path: Path = QDRANT_DIR) -> QdrantClient:
    """Le client qdrant local. Pas de serveur a lancer.

    Par contre qdrant locke le dossier, du coup on en ouvre qu'un par
    processus (c'est pour ca que le retriever le garde en attribut).
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def rebuild_collection(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: np.ndarray,
    collection: str = COLLECTION,
) -> int:
    """On recree la collection et on push les vecteurs. Renvoie le compte."""
    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
    )

    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=c.id,  # UUID deterministe -> upsert idempotent
                vector=v.tolist(),
                payload={
                    "text": c.text,
                    "raw_text": c.raw_text,
                    "headings": list(c.headings),
                    "pages": list(c.pages),
                    "source": c.source,
                },
            )
            for c, v in zip(chunks, vectors)
        ],
    )
    return int(client.count(collection).count)


def build_index(
    chunks: list[Chunk],
    encoder: Encoder | None = None,
    collection: str = COLLECTION,
) -> dict[str, int]:
    """Cree les 2 index et les sauvegarde. Renvoie un petit resume."""
    if not chunks:
        raise ValueError("Aucun chunk a indexer.")

    ensure_dirs()
    encoder = encoder or Encoder()

    print(f"[index] Embedding de {len(chunks)} chunks ...", flush=True)
    vectors = encoder.encode_passages([c.text for c in chunks])

    print("[index] Ecriture dans Qdrant ...", flush=True)
    client = open_qdrant()
    try:
        n_vectors = rebuild_collection(client, chunks, vectors, collection=collection)
    finally:
        client.close()  # important, sinon le dossier reste locke

    print("[index] Construction de l'index BM25 ...", flush=True)
    BM25Index.build(chunks).save()

    return {
        "n_chunks": len(chunks),
        "n_vectors": n_vectors,
        "n_sources": len({c.source for c in chunks}),
    }
