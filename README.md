# InsureRAG

Pipeline RAG sur des **conditions générales d'assurance habitation** (PDF français) :

```
PDF  →  Docling (parsing)  →  HybridChunker (chunking)  →  embeddings e5
     →  Qdrant (dense) + BM25 (lexical)  →  fusion RRF  →  reranking cross-encoder
     →  LLM local Ollama  →  réponse citée [1] [2]
```

Interface Streamlit, aucune clé API, aucun serveur de base de données à lancer.

---

## Documents source

Les contrats d'assurance ne sont pas versionnés dans ce dépôt. Déposez vos PDF
dans `data/raw/`, puis lancez `python scripts/build_index.py`.

Les documents utilisés pour le développement :

| Contrat | Lien |
| --- | --- |
| Macif — habitation | _(à compléter)_ |
| GMG / AG2R | _(à compléter)_ |
| MRH4 | _(à compléter)_ |
| Ma Maison | _(à compléter)_ |

---

## Installation

```powershell
uv venv .venv
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

# torch doit être installé depuis l'index CUDA pour utiliser le GPU
uv pip install torch torchvision --python .venv\Scripts\python.exe `
    --index-url https://download.pytorch.org/whl/cu128 `
    --reinstall-package torch --reinstall-package torchvision
```

Le LLM est servi par **Ollama** (à installer séparément) :

```powershell
ollama serve                       # si le service ne tourne pas déjà
ollama pull llama3.2:3b            # modèle par défaut
```

## Utilisation

```powershell
# 1. Construire l'index (parsing -> chunking -> embeddings -> Qdrant + BM25)
.\.venv\Scripts\python.exe scripts\build_index.py

# 2. Poser une question
.\.venv\Scripts\python.exe scripts\ask.py "Quelles sont les exclusions en cas de dégât des eaux ?"
.\.venv\Scripts\python.exe scripts\ask.py "..." --top-k 8
.\.venv\Scripts\python.exe scripts\ask.py "..." --no-rerank    # plus rapide, moins précis
.\.venv\Scripts\python.exe scripts\ask.py --eval                # hit@k / MRR

# 3. Interface web
.\.venv\Scripts\streamlit.exe run app.py

# Tests
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `config.py` | Chemins, modèles, hyper-paramètres — surchargeables par `.env` |
| `src/insurerag/schemas.py` | `Chunk`, `Hit`, `Source`, `AskResult` |
| `src/insurerag/parsing.py` | Conversion Docling + exports markdown / JSON |
| `src/insurerag/chunking.py` | `HybridChunker` aligné sur le tokenizer d'embedding |
| `src/insurerag/embedding.py` | Encodeur dense (prefixes e5 `passage:` / `query:`) |
| `src/insurerag/indexing.py` | Qdrant + BM25, tokenisation française |
| `src/insurerag/retrieval.py` | Dense, BM25, fusion RRF, reranking |
| `src/insurerag/generation.py` | Prompt cité + client Ollama |
| `src/insurerag/pipeline.py` | `RAGPipeline.ask()` |
| `app.py` | Interface Streamlit |
| `scripts/` | `build_index.py`, `ask.py` |

Chaque étape est un module à responsabilité unique, sans état global : les
modèles lourds sont chargés à la demande et chaque composant est injectable.

## Provenance

Ce code a succédé à deux scripts expérimentaux (`ingestion.py` et `rag.py`),
supprimés après refonte. Toute la logique a été portée dans `src/insurerag/`,
avec trois corrections par rapport à l'original :

- le chemin du PDF n'est plus codé en dur, et le parsing ne s'exécute plus à
  l'import ;
- les index sont associés à des **UUID stables** et non à des positions dans
  une liste Python, ce qui permet de servir l'index depuis un autre processus
  (cas de Streamlit) ;
- une seule collection couvre les 4 contrats, au lieu d'un nom de collection
  par document.

## Données

```
data/raw/         PDF sources (non versionnés) — déposez-y vos contrats
data/processed/   exports Docling (*.md, *.json) + chunks.jsonl
                  + bm25.pkl + qdrant/   (non versionnés)
```

## Configuration

Copiez `.env.example` en `.env` puis adaptez. Aucune valeur n'est obligatoire.

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `INSURERAG_DEVICE` | auto (`cuda`/`cpu`) | Forcer l'appareil |
| `INSURERAG_EMBED_MODEL` | `intfloat/multilingual-e5-base` | Embeddings |
| `INSURERAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder |
| `INSURERAG_LLM_MODEL` | `llama3.2:3b` | Modèle Ollama |
| `INSURERAG_OLLAMA_HOST` | `http://localhost:11434` | Serveur Ollama |
| `INSURERAG_LLM_NUM_CTX` | `8192` | Fenêtre de contexte |
| `INSURERAG_CHUNK_MAX_TOKENS` | `400` | Taille max d'un chunk |
| `INSURERAG_TOP_K_RERANK` | `5` | Passages conservés |
| `INSURERAG_RRF_K` | `60` | Constante de fusion RRF |

Un modèle plus fort s'obtient sans toucher au code :
`ollama pull qwen2.5:7b` puis `INSURERAG_LLM_MODEL=qwen2.5:7b` dans `.env`.

## Notes

- **Qdrant verrouille son dossier en mode local** : un seul client est ouvert
  par processus. `build_index.py` et `app.py` ne doivent pas tourner en même
  temps, et c'est pourquoi le client est fermé explicitement après l'écriture.
- **Les identifiants de chunks sont des UUID5 déterministes**
  (`nom_du_fichier-index_du_chunk`) : reconstruire l'index sur les mêmes PDF
  réécrit exactement les mêmes points, l'opération est idempotente.
- **Les chunks sont servis depuis les payloads Qdrant**, pas depuis une liste en
  mémoire : l'application peut donc démarrer dans un processus différent de
  celui qui a construit l'index (c'est le cas de Streamlit).
- `llama3.2:3b` est volontairement plus littéral qu'un 7B : les réponses citent
  strictement les extraits fournis.
