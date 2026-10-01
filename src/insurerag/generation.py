"""Etape 6 - la reponse du LLM.

On envoie juste des requetes http a ollama (/api/chat) au lieu de charger un
modele en local, comme ca pas besoin de bitsandbytes et le GPU reste libre pour
la recherche.

Le prompt oblige le LLM a citer ses sources [1] [2] et a dire "je ne trouve
pas" quand l'info est pas dans les extraits.
"""

from __future__ import annotations

import requests

from config import (
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_HOST,
)
from .schemas import Chunk

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans l'analyse de contrats d'assurance habitation. "
    "Tu réponds UNIQUEMENT à partir des extraits fournis. "
    "Si l'information n'est pas dans les extraits, réponds : "
    "\"Je ne trouve pas cette information dans les documents fournis.\" "
    "Cite systématiquement tes sources sous la forme [1], [2]... correspondant aux extraits. "
    "Réponds en français, de façon claire et précise."
)

NO_ANSWER = "Je ne trouve pas cette information dans les documents fournis."


class OllamaUnavailable(RuntimeError):
    """Ollama repond pas, ou le modele demande est pas installe."""


def build_prompt(query: str, passages: list[Chunk]) -> list[dict]:
    """Colle les extraits numerotes + la question dans un message user."""
    context = "\n\n".join(
        f"[{i}] (source {p.source} | pages {', '.join(map(str, p.pages)) or 'n/a'} "
        f"| {' > '.join(p.headings) or 'sans titre'})\n{p.raw_text}"
        for i, p in enumerate(passages, start=1)
    )
    user = (
        f"EXTRAITS :\n{context}\n\n"
        f"QUESTION : {query}\n\n"
        f"RÉPONSE (avec citations [n]) :"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class OllamaGenerator:
    """Petit client pour parler au serveur ollama."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = LLM_TEMPERATURE,
        num_predict: int = LLM_NUM_PREDICT,
        num_ctx: int = LLM_NUM_CTX,
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.timeout = timeout

    def is_available(self) -> bool:
        # /api/tags existe sur toutes les versions, c'est le plus simple a taper
        try:
            requests.get(f"{self.host}/api/tags", timeout=5)
            return True
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.host}/api/tags", timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def generate(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                f"Serveur Ollama injoignable sur {self.host}.\n"
                f"Demarrez-le avec `ollama serve` (ou lancez l'application Ollama "
                f"sur Windows). Detail : {exc}"
            ) from exc

        if response.status_code == 404:
            raise OllamaUnavailable(
                f"Modele '{self.model}' absent d'Ollama.\n"
                f"Modeles disponibles : {self.list_models()}\n"
                f"Installez-le avec `ollama pull {self.model}` "
                f"ou changez INSURERAG_LLM_MODEL dans .env"
            )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def answer(self, query: str, passages: list[Chunk]) -> str:
        # pas d'extraits -> pas de reponse, on dit direct que c'est pas trouve
        if not passages:
            return NO_ANSWER
        return self.generate(build_prompt(query, passages))
