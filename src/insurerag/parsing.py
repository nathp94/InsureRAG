"""Etape 1 - lecture des PDF avec docling.

Le converter se construit une fois et on le reutilise pour tous les PDF, ca
prend du temps a loader donc vaut mieux pas le refaire 4 fois.
Chaque PDF est aussi ecrit en .md et en .json dans data/processed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from config import DEVICE, PROCESSED_DIR, RAW_DIR, ensure_dirs, list_raw_pdfs


def build_converter() -> DocumentConverter:
    """Le converter, sur GPU si on en a un."""
    device = AcceleratorDevice.CUDA if DEVICE == "cuda" else AcceleratorDevice.CPU
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def convert_pdf(converter: DocumentConverter, pdf_path: Path) -> tuple[DoclingDocument, float]:
    """Renvoie (le document, combien de temps ca a pris)."""
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    start = time.perf_counter()
    result = converter.convert(pdf_path)
    elapsed = time.perf_counter() - start
    return result.document, elapsed


def export_document(
    doc: DoclingDocument, stem: str, out_dir: Path = PROCESSED_DIR
) -> tuple[Path, Path]:
    """Ecrit le .md et le .json, renvoie les 2 chemins."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(doc.export_to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    return md_path, json_path


def load_document(stem: str, out_dir: Path = PROCESSED_DIR) -> DoclingDocument:
    """Relit un document depuis le .json, ca evite de re-parser le PDF."""
    json_path = Path(out_dir) / f"{stem}.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Export JSON absent : {json_path}. Lancez d'abord le parsing."
        )
    return DoclingDocument.load_from_json(json_path)


def parse_corpus(
    pdf_paths: list[Path] | None = None, out_dir: Path = PROCESSED_DIR
) -> dict[str, DoclingDocument]:
    """Parse tous les PDF de data/raw. Renvoie {nom_pdf: document}."""
    ensure_dirs()
    pdfs = [Path(p) for p in (pdf_paths if pdf_paths is not None else list_raw_pdfs())]
    if not pdfs:
        raise FileNotFoundError(f"Aucun PDF trouve dans {RAW_DIR}")

    converter = build_converter()
    documents: dict[str, DoclingDocument] = {}

    for pdf in pdfs:
        print(f"[parse] {pdf.name} ...", flush=True)
        doc, elapsed = convert_pdf(converter, pdf)
        md_path, json_path = export_document(doc, pdf.stem, out_dir)
        print(
            f"         {elapsed:.1f}s -> {md_path.name} "
            f"({md_path.stat().st_size / 1024:.1f} KB), {json_path.name}",
            flush=True,
        )
        documents[pdf.name] = doc

    return documents
