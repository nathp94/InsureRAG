"""Etape 3 - les embeddings denses.

ATTENTION le modele e5 veut un prefixe different selon ce qu'on encode :
"passage: " pour les documents et "query: " pour les questions.
Si on oublie ca les resultats sont vraiment mauvais.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DEVICE, EMBED_BATCH_SIZE, EMBED_MODEL

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class Encoder:
    """Le modele est charge qu'a la premiere utilisation, pas avant.

    Comme ca streamlit peut creer le retriever direct et on paie le chargement
    qu'a la premiere question.
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str = DEVICE,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.device == "cuda":
                # en float16 ca prend 2x moins de vram
                self._model.half()
        return self._model

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [f"{PASSAGE_PREFIX}{t}" for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            f"{QUERY_PREFIX}{query}", normalize_embeddings=True
        )
