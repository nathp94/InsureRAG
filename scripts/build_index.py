"""Construit l'index complet : parsing -> chunking -> embeddings -> Qdrant + BM25.

    python scripts/build_index.py
    python scripts/build_index.py --rebuild   # sans confirmation interactive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insurerag.chunking import chunk_corpus, save_chunks  # noqa: E402
from insurerag.indexing import build_index  # noqa: E402
from config import (  # noqa: E402
    CHUNKS_PATH,
    CHUNK_MAX_TOKENS,
    COLLECTION,
    DEVICE,
    EMBED_MODEL,
    RERANK_MODEL,
    ensure_dirs,
    index_is_built,
    list_raw_pdfs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Construction de l'index InsureRAG.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruit meme si un index existe deja (par defaut : on confirme).",
    )
    args = parser.parse_args()

    pdfs = list_raw_pdfs()
    if not pdfs:
        print("Aucun PDF dans data/raw. Deposez-y vos contrats puis relancez.")
        return 1

    if index_is_built() and not args.rebuild:
        answer = input(
            "Un index existe deja. Le reconstruire ? [o/N] "
        ).strip().lower()
        if answer not in {"o", "oui", "y", "yes"}:
            print("Annule.")
            return 0

    ensure_dirs()

    print("=" * 70)
    print("InsureRAG - construction de l'index")
    print(f"  appareil   : {DEVICE}")
    print(f"  modele     : {EMBED_MODEL}")
    print(f"  reranker   : {RERANK_MODEL}")
    print(f"  collection : {COLLECTION}")
    print(f"  chunking   : {CHUNK_MAX_TOKENS} tokens max")
    print(f"  sources    : {len(pdfs)} PDF")
    for pdf in pdfs:
        print(f"                - {pdf.name}")
    print("=" * 70)

    start = time.perf_counter()

    # 1. on parse et on decoupe
    chunks = chunk_corpus(pdfs)
    if not chunks:
        print("Aucun chunk genere.")
        return 1
    save_chunks(chunks)

    lengths = [len(c.raw_text) for c in chunks]
    print(
        f"\n[chunks] {len(chunks)} chunks | "
        f"caracteres min {min(lengths)} / moyen {sum(lengths) // len(lengths)} / max {max(lengths)}"
    )

    # 2. la on fait les embeddings et on remplit qdrant + bm25
    stats = build_index(chunks)

    elapsed = time.perf_counter() - start
    print("=" * 70)
    print(
        f"Termine en {elapsed / 60:.1f} min : "
        f"{stats['n_chunks']} chunks, {stats['n_vectors']} vecteurs, "
        f"{stats['n_sources']} documents."
    )
    print(f"  chunks : {CHUNKS_PATH}")
    print(f"\nPostez des questions avec :  python scripts/ask.py \"ma question\"")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
