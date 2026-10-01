"""Tests sur les fonctions qui ont pas besoin de modele.

On teste surtout la tokenisation et le RRF, ca tourne en 1 seconde.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest  # noqa: E402

from insurerag.indexing import BM25Index, strip_accents, tokenize_fr  # noqa: E402
from insurerag.retrieval import rrf_fusion  # noqa: E402
from insurerag.schemas import Chunk, Hit, Source  # noqa: E402


def make_chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(id=f"id-{i}", text=t, raw_text=t, headings=[], pages=[i + 1], source="test.pdf")
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_strip_accents(self):
        assert strip_accents("dégâts des eaux") == "degats des eaux"
        assert strip_accents("résiliation") == "resiliation"
        assert strip_accents("çà") == "ca"

    def test_lowercases_and_splits(self):
        assert tokenize_fr("Franchise DE 300 EUROS") == ["franchise", "300", "euro"]

    def test_removes_stopwords(self):
        tokens = tokenize_fr("les garanties de la franchise")
        assert "les" not in tokens and "de" not in tokens and "la" not in tokens
        assert "franchise" in tokens

    def test_removes_simple_plural(self):
        # "garanties" (>= 5 lettres) devient "garantie" ; "vol" (3 lettres) reste.
        assert tokenize_fr("les garanties contre le vol") == ["garantie", "contre", "vol"]

    def test_accented_stopwords_removed(self):
        # "à" est un mot vide accentue : il doit disparaitre.
        assert "a" not in tokenize_fr("à partir de 2024")

    def test_empty(self):
        assert tokenize_fr("") == []


# ---------------------------------------------------------------------------
# Fusion RRF
# ---------------------------------------------------------------------------
class TestRRF:
    def test_single_ranking_preserves_order(self):
        ranking = [Hit("a", 0.9), Hit("b", 0.5), Hit("c", 0.1)]
        fused = rrf_fusion([ranking], k=60, top_n=3)
        assert [h.chunk_id for h in fused] == ["a", "b", "c"]

    def test_scores_decrease(self):
        fused = rrf_fusion([[Hit("a", 9.0), Hit("b", 0.001)]], k=60, top_n=2)
        assert fused[0].score > fused[1].score

    def test_agreement_between_strategies_wins(self):
        # un chunk bien classe par les 2 methodes doit passer devant un chunk
        # qui est 1er d'une seule methode, c'est tout l'interet du RRF
        dense = [Hit("x", 0.99), Hit("shared", 0.98)]
        lexical = [Hit("shared", 5.0), Hit("y", 4.0)]
        fused = rrf_fusion([dense, lexical], k=60, top_n=3)
        assert fused[0].chunk_id == "shared"

    def test_top_n_truncates(self):
        ranking = [Hit(str(i), 1.0) for i in range(10)]
        assert len(rrf_fusion([ranking], top_n=4)) == 4

    def test_empty_inputs(self):
        assert rrf_fusion([], top_n=5) == []
        assert rrf_fusion([[], []], top_n=5) == []

    def test_ignores_raw_scores(self):
        # le score de depart compte pas, seul le rang compte
        a = rrf_fusion([[Hit("x", 0.001)]], k=60, top_n=1)
        b = rrf_fusion([[Hit("x", 999.0)]], k=60, top_n=1)
        assert a[0].score == b[0].score

    def test_duplicates_accumulate(self):
        fused = rrf_fusion([[Hit("a", 1.0), Hit("a", 0.5)]], k=60, top_n=1)
        expected = 1 / 61 + 1 / 62
        assert fused[0].score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class TestBM25:
    def test_ranks_relevant_chunk_first(self):
        # Corpus volontairement plus large que 2 documents : avec un corpus
        # minuscule, l'IDF de rank_bm25 s'annule et tous les scores sont nuls.
        index = BM25Index.build(
            make_chunks(
                "La garantie dommages aux biens couvre le vol de choses.",
                "La garantie responsabilite civile couvre les dommages causes.",
                "Le sinistre doit etre declare sous 5 jours ouvres.",
                "Les frais de dossier sont fixes par le contrat annuel.",
                "Le remboursement intervient apres expertise du bien endommage.",
                "La renovation couvre les toitures et les facades.",
                "Toute declaration tardive peut entrainer une reduction.",
                "Les garanties spatiales ne concernent pas les biens mobiles.",
            )
        )
        hits = index.search("combien de jours pour declarer un sinistre", k=8)
        assert hits, "aucun resultat : l'IDF est annule sur un corpus trop petit"
        assert hits[0].chunk_id == "id-2"
        assert hits[0].score > 0

    def test_returns_nothing_when_no_match(self):
        index = BM25Index.build(
            make_chunks(
                "La garantie vol couvre les biens de l'assure.",
                "La responsabilite civile couvre les dommages causes a autrui.",
                "Les franchises s'appliquent par sinistre et par annee.",
            )
        )
        assert index.search("aviation commerciale de fret", k=3) == []

    def test_ids_align_with_corpus(self):
        chunks = make_chunks("premier texte", "deuxieme texte", "troisieme texte")
        index = BM25Index.build(chunks)
        assert index.ids == [c.id for c in chunks]

    def test_empty_index_is_inert(self):
        """Un corpus vide ne doit pas lever ZeroDivisionError."""
        index = BM25Index.build([])
        assert index.search("anything", k=3) == []


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TestSource:
    # Source.label() c'est ce que la CLI et l'UI streamlit affichent

    def _source(self) -> Source:
        return Source(
            ref=1,
            source="CG-Macif-Habitation.pdf",
            pages=[12],
            headings=["Franchise"],
            score=0.5,  # doit etre un float : un Hit ici leve TypeError
            extract="extrait...",
        )

    # si on met un Hit la au lieu d'un float ca leve une TypeError,
    # c'est un bug qu'on a deja eu donc on garde le test
    def test_label_formats_score(self):
        label = self._source().label()
        assert "[1]" in label
        assert "CG-Macif-Habitation.pdf" in label
        assert "0.500" in label
        assert "Franchise" in label

    def test_label_handles_missing_metadata(self):
        source = self._source()
        source.pages = []
        source.headings = []
        label = source.label()
        assert "n/a" in label and "sans titre" in label

    def test_roundtrip_to_dict(self):
        assert Source.from_dict(self._source().to_dict()).ref == 1


class TestChunkSchema:
    def test_roundtrip(self):
        chunk = Chunk(
            id="abc", text="t", raw_text="r", headings=["H"], pages=[3], source="s.pdf"
        )
        assert Chunk.from_dict(chunk.to_dict()) == chunk

    def test_from_dict_tolerates_missing_optionals(self):
        chunk = Chunk.from_dict({"id": "abc", "text": "t"})
        assert chunk.headings == [] and chunk.pages == [] and chunk.source == ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
