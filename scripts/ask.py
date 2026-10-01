"""CLI de questionnement.

    python scripts/ask.py "Quelles sont les exclusions en cas de degat des eaux ?"
    python scripts/ask.py "..." --top-k 8
    python scripts/ask.py "..." --no-rerank      # plus rapide, moins precis
    python scripts/ask.py --eval                # hit@k / MRR sur 5 questions
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import TOP_K_RERANK, index_is_built  # noqa: E402
from insurerag.generation import OllamaUnavailable  # noqa: E402
from insurerag.indexing import strip_accents  # noqa: E402
from insurerag.pipeline import RAGPipeline  # noqa: E402
from insurerag.retrieval import HybridRetriever  # noqa: E402
from insurerag.schemas import AskResult  # noqa: E402

# quelques questions pour tester la recherche.
# un passage est ok si il contient TOUS les mots-cles.
EVAL_SET = [
    {"question": "Dans quel délai faut-il declarer un vol ?", "keywords": ["vol", "jour"]},
    {"question": "Dans quel délai faut-il declarer un sinistre ?", "keywords": ["jour", "sinistre"]},
    {"question": "Que couvre la garantie degats des eaux ?", "keywords": ["degats", "eau"]},
    {"question": "Qu'est-ce qu'une franchise ?", "keywords": ["franchise"]},
    {"question": "Comment resilier le contrat ?", "keywords": ["resil"]},
]


def render(result: AskResult) -> None:
    print("=" * 78)
    print("QUESTION :", result.question)
    print("-" * 78)
    print(result.answer)
    print("-" * 78)
    print("SOURCES :")
    for source in result.sources:
        print(f"  {source.label()}")
    print("=" * 78)


def evaluate(pipeline: RAGPipeline, k: int) -> None:
    """hit@k et MRR. On teste que la recherche, on appelle pas le LLM."""
    hits: list[bool] = []
    rr: list[float] = []

    for example in EVAL_SET:
        question = example["question"]
        keywords = [kw.lower() for kw in example["keywords"]]

        ranked = pipeline.retriever.retrieve(question, top_k=k)
        chunks = pipeline.retriever.get_chunks([h.chunk_id for h in ranked])

        rank = None
        for position, chunk in enumerate(chunks, start=1):
            haystack = (chunk.raw_text + " " + " ".join(chunk.headings)).lower()
            # on enleve les accents pour comparer ("degats" == "degâts")
            haystack = strip_accents(haystack)
            if all(strip_accents(kw) in haystack for kw in keywords):
                rank = position
                break

        hits.append(rank is not None)
        rr.append(1.0 / rank if rank else 0.0)
        mark = "OK " if rank else "KO "
        print(f"  {mark}{question}" + (f"  (rang {rank})" if rank else "  (hors top k)"))

    n = len(EVAL_SET)
    print("-" * 78)
    print(f"  hit@{k} = {sum(hits) / n:.2f}   |   MRR = {sum(rr) / n:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Questionner le corpus InsureRAG.")
    parser.add_argument("question", nargs="?", help="La question a poser.")
    parser.add_argument("--top-k", type=int, default=TOP_K_RERANK, help="Passages retenus.")
    parser.add_argument("--no-rerank", action="store_true", help="Desactive le reranking.")
    parser.add_argument("--eval", action="store_true", help="Lance l'evaluation du retrieval.")
    args = parser.parse_args()

    if not args.question and not args.eval:
        parser.print_help()
        return 1

    if not index_is_built():
        print(
            "Index absent ou incomplet.\n"
            "Lancez d'abord :  python scripts/build_index.py"
        )
        return 1

    pipeline = RAGPipeline(
        retriever=HybridRetriever(use_rerank=not args.no_rerank)
    )

    try:
        if args.eval:
            print(f"\nEVALUATION DU RETRIEVAL (top_k={args.top_k})\n")
            evaluate(pipeline, args.top_k)
            return 0

        start = time.perf_counter()
        result = pipeline.ask(args.question, top_k=args.top_k)
        render(result)
        print(f"\n({time.perf_counter() - start:.1f}s)")
        return 0

    except OllamaUnavailable as exc:
        print(f"\n[LLM indisponible] {exc}")
        return 2
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
