"""Config du projet. Tout est modifiable avec des variables d'environnement."""

from __future__ import annotations

import os
from pathlib import Path

# chemin du fichier config.py = racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent

# on essaie de lire le .env mais ca doit pas planter si le module manque
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


# ---- chemins ----
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QDRANT_DIR = PROCESSED_DIR / "qdrant"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"


# ---- gpu ou cpu ----
def detect_device() -> str:
    # si la var d'env est forcee on respecte, sinon on regarde torch
    forced = os.getenv("INSURERAG_DEVICE")
    if forced:
        return forced.strip().lower()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = detect_device()


# ---- modeles ----
EMBED_MODEL = os.getenv("INSURERAG_EMBED_MODEL", "intfloat/multilingual-e5-base")
RERANK_MODEL = os.getenv("INSURERAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_BATCH_SIZE = int(os.getenv("INSURERAG_EMBED_BATCH_SIZE", "32"))

# le LLM tourne sur ollama, c'est lui qui gere le GPU
LLM_MODEL = os.getenv("INSURERAG_LLM_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("INSURERAG_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_TEMPERATURE = float(os.getenv("INSURERAG_LLM_TEMPERATURE", "0"))
LLM_NUM_CTX = int(os.getenv("INSURERAG_LLM_NUM_CTX", "8192"))
LLM_NUM_PREDICT = int(os.getenv("INSURERAG_LLM_NUM_PREDICT", "512"))
LLM_TIMEOUT = float(os.getenv("INSURERAG_LLM_TIMEOUT", "300"))


# ---- indexation ----
COLLECTION = os.getenv("INSURERAG_COLLECTION", "insurerag")
CHUNK_MAX_TOKENS = int(os.getenv("INSURERAG_CHUNK_MAX_TOKENS", "400"))


# ---- recherche ----
# on recupere plein de chunks puis on en garde que 5 a la fin
TOP_K_DENSE = int(os.getenv("INSURERAG_TOP_K_DENSE", "20"))
TOP_K_BM25 = int(os.getenv("INSURERAG_TOP_K_BM25", "20"))
TOP_K_FUSION = int(os.getenv("INSURERAG_TOP_K_FUSION", "20"))
TOP_K_RERANK = int(os.getenv("INSURERAG_TOP_K_RERANK", "5"))
RRF_K = int(os.getenv("INSURERAG_RRF_K", "60"))

# combien de caracteres on affiche dans l'UI pour chaque source
EXTRACT_CHARS = int(os.getenv("INSURERAG_EXTRACT_CHARS", "300"))


# ---- petits utilitaires ----
def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def index_is_built() -> bool:
    # sert a dire a l'app "pas encore d'index, va le construire"
    return (
        CHUNKS_PATH.is_file()
        and BM25_PATH.is_file()
        and QDRANT_DIR.is_dir()
        and any(QDRANT_DIR.iterdir())
    )


def list_raw_pdfs() -> list[Path]:
    if not RAW_DIR.is_dir():
        return []
    return sorted(RAW_DIR.glob("*.pdf"))
