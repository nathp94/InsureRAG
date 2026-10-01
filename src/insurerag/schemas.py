"""Les dataclasses utilisees partout dans le projet.

Je les mets dans un fichier a part sinon pipeline.py devrait importer
chunking.py et ca fait n'importe quoi niveau dependances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Un bout de texte qu'on a indexe.

    text = le texte avec les titres devant, c'est ca qu'on indexe et qu'on
    cherche. raw_text = le texte tout seul, c'est ca qu'on envoie au LLM
    (sinon il se repete les titres partout).
    """

    id: str
    text: str
    raw_text: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "raw_text": self.raw_text,
            "headings": list(self.headings),
            "pages": list(self.pages),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            raw_text=data.get("raw_text", ""),
            headings=list(data.get("headings") or []),
            pages=list(data.get("pages") or []),
            source=data.get("source", ""),
        )


@dataclass
class Hit:
    """Un resultat de recherche. Juste l'id et le score, le texte on le relit
    apres dans qdrant."""

    chunk_id: str
    score: float


@dataclass
class Source:
    """Une source qu'on affiche dans l'UI."""

    ref: int
    source: str
    pages: list[int]
    headings: list[str]
    score: float
    extract: str

    def label(self) -> str:
        pages = ", ".join(str(p) for p in self.pages) or "n/a"
        headings = " > ".join(self.headings) or "sans titre"
        return f"[{self.ref}] {self.source} - p. {pages} | {headings} ({self.score:.3f})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "source": self.source,
            "pages": list(self.pages),
            "headings": list(self.headings),
            "score": round(self.score, 3),
            "extract": self.extract,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        return cls(
            ref=data["ref"],
            source=data.get("source", ""),
            pages=list(data.get("pages") or []),
            headings=list(data.get("headings") or []),
            score=float(data.get("score", 0.0)),
            extract=data.get("extract", ""),
        )


@dataclass
class AskResult:
    """Ce que renvoie pipeline.ask()."""

    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
        }
