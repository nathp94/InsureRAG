"""Etape 2 - decoupe des documents en chunks avec le HybridChunker de docling.

Le chunker est cale sur le tokenizer du modele d'embedding, comme ca on est
sur que ca rentre dedans sans etre coupe au milieu.
Chaque chunk garde 2 versions du texte :
  - text     : avec les titres devant, pour indexer et chercher
  - raw_text : le texte tout seul, pour envoyer au LLM
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from config import (
    CHUNKS_PATH,
    CHUNK_MAX_TOKENS,
    EMBED_MODEL,
    RAW_DIR,
    ensure_dirs,
    list_raw_pdfs,
)
from .schemas import Chunk


def build_chunker(
    max_tokens: int = CHUNK_MAX_TOKENS, embed_model: str = EMBED_MODEL
) -> HybridChunker:
    """Le chunker, cale sur le tokenizer du modele d'embedding."""
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
    hf_tokenizer = HuggingFaceTokenizer(tokenizer=embed_tokenizer, max_tokens=max_tokens)
    return HybridChunker(tokenizer=hf_tokenizer, merge_peers=True)


def _extract_pages(chunk) -> list[int]:
    # va chercher les pages dans les "prov" de chaque item du chunk
    pages: set[int] = set()
    for item in chunk.meta.doc_items:
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(page_no)
    return sorted(pages)


def build_chunks(
    doc: DoclingDocument, source_name: str, chunker: HybridChunker | None = None
) -> list[Chunk]:
    """Decoupe un document. Les ids sont des uuid5 donc ils sont toujours les
    memes si on relance sur les memes PDF (ca evite de dupliquer dans qdrant)."""
    chunker = chunker or build_chunker()

    chunks: list[Chunk] = []
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        chunks.append(
            Chunk(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_name}-{i}")),
                text=chunker.contextualize(chunk=ch),
                raw_text=ch.text,
                headings=list(ch.meta.headings or []),
                pages=_extract_pages(ch),
                source=source_name,
            )
        )
    return chunks


def chunk_document(
    doc: DoclingDocument, source_name: str, chunker: HybridChunker | None = None
) -> list[Chunk]:
    # meme chose que build_chunks, nom plus clair quand on l'appelle du pipeline
    return build_chunks(doc, source_name, chunker)


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> Path:
    """Un chunk par ligne, en JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    """Relit les chunks depuis le JSONL. Liste vide si le fichier n'existe pas."""
    path = Path(path)
    if not path.is_file():
        return []
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def chunk_corpus(
    pdf_paths: list[Path] | None = None,
) -> list[Chunk]:
    """Parse tout le corpus et le decoupe. Renvoie les chunks de tous les PDF.

    On construit un seul chunker et on le reutilise partout.
    """
    from .parsing import build_converter, convert_pdf, export_document

    ensure_dirs()
    pdfs = [Path(p) for p in (pdf_paths if pdf_paths is not None else list_raw_pdfs())]
    if not pdfs:
        raise FileNotFoundError(f"Aucun PDF source trouve dans {RAW_DIR}")

    converter = build_converter()
    chunker = build_chunker()
    all_chunks: list[Chunk] = []

    for pdf in pdfs:
        doc, elapsed = convert_pdf(converter, pdf)
        export_document(doc, pdf.stem)
        chunks = build_chunks(doc, pdf.name, chunker)
        print(
            f"[chunk] {pdf.name} : {len(chunks)} chunks ({elapsed:.1f}s de parsing)",
            flush=True,
        )
        all_chunks.extend(chunks)

    return all_chunks
