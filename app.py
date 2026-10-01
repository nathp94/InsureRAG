"""L'interface streamlit.

    streamlit run app.py

On met le pipeline en cache avec st.cache_resource, sinon a chaque clic il
recharge l'encodeur, le reranker et qdrant et ca prend 30 secondes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    COLLECTION,
    DEVICE,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_HOST,
    RERANK_MODEL,
    TOP_K_RERANK,
    index_is_built,
)
from insurerag.generation import OllamaUnavailable  # noqa: E402
from insurerag.pipeline import get_pipeline  # noqa: E402

st.set_page_config(page_title="InsureRAG", page_icon="🏠", layout="centered")

EXAMPLES = [
    "Quelles sont les exclusions en cas de dégât des eaux ?",
    "Dans quel délai dois-je déclarer un sinistre ?",
    "Qu'est-ce qu'une franchise ?",
    "Comment puis-je résilier mon contrat ?",
]


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_pipeline(use_rerank: bool):
    return get_pipeline(use_rerank=use_rerank)


def render_sources(sources) -> None:
    """Les extraits cites, dans un expander replie."""
    if not sources:
        return
    with st.expander(f"📄 {len(sources)} source(s) utilisée(s)"):
        for source in sources:
            st.markdown(f"**[{source.ref}]** `{source.source}` · score `{source.score:.3f}`")
            meta = []
            if source.pages:
                meta.append("p. " + ", ".join(str(p) for p in source.pages))
            if source.headings:
                meta.append(" > ".join(source.headings))
            if meta:
                st.caption(" · ".join(meta))
            st.text(source.extract)
            st.divider()


# ---- la barre sur le cote ----
with st.sidebar:
    st.header("⚙️ Paramètres")

    top_k = st.slider("Passages retenus", 1, 12, TOP_K_RERANK)
    use_rerank = st.toggle("Reranking (cross-encoder)", value=True)

    st.divider()
    st.caption("Index")
    st.write(f"Collection : `{COLLECTION}`")
    st.write(f"Embeddings : `{EMBED_MODEL}`")
    st.write(f"Reranker : `{RERANK_MODEL}`")
    st.write(f"Appareil : `{DEVICE.upper()}`")
    st.write(f"LLM : `{LLM_MODEL}` (Ollama)")

    if st.button("🗑️ Vider la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- le reste de la page ----
st.title("🏠 InsureRAG")
st.caption(
    "Posez une question sur vos conditions générales d'assurance habitation. "
    "Les réponses s'appuient uniquement sur les documents indexés."
)

# pas d'index -> on dit comment le faire
if not index_is_built():
    st.error("Aucun index disponible.")
    st.code("python scripts/build_index.py", language="bash")
    st.stop()

# ollama doit tourner sinon on peut pas repondre
pipeline = load_pipeline(use_rerank)
if not pipeline.generator.is_available():
    st.warning(
        f"Le serveur Ollama ne répond pas sur `{OLLAMA_HOST}`.\n\n"
        "Démarrez-le avec `ollama serve` (ou lancez l'application Ollama sur Windows), "
        f"puis rechargez cette page. Modèle attendu : `{LLM_MODEL}`."
    )
    st.stop()

# l'historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

# quelques questions d'exemple, seulement au premier message
if not st.session_state.messages:
    st.markdown("**Exemples :**")
    for example in EXAMPLES:
        if st.button(example, key=f"ex_{example[:20]}"):
            st.session_state.pending = example
            break

prompt = st.chat_input("Votre question…")

if prompt is None and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et rédaction…"):
            try:
                result = pipeline.ask(prompt, top_k=top_k)
            except OllamaUnavailable as exc:
                st.error(str(exc))
                st.stop()

        st.markdown(result.answer)
        render_sources(result.sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
