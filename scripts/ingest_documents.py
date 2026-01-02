import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from pdfminer.high_level import extract_text as pdf_extract_text
from PIL import Image
from pypdf import PdfReader
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.append(str(BACKEND_PATH))

from app.config import settings
from app.rag.vector_store import VectorStore  # noqa: E402


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return pdf_extract_text(str(path))


def read_image(path: Path) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown", ".png", ".jpg", ".jpeg"}:
            yield path


def load_and_chunk(path: Path) -> Tuple[str, List[str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content = read_pdf(path)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        content = read_image(path)
    else:
        content = read_text_file(path)
    return str(path), chunk_text(content)


def ingest(input_dir: Path, db_path: Path) -> None:
    store = VectorStore(path=str(db_path), embedding_model=settings.embedding_model)
    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[dict] = []

    for path in iter_files(input_dir):
        source, chunks = load_and_chunk(path)
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{source}#chunk-{idx}")
            metadatas.append({"source": source, "chunk": idx})

    if documents:
        store.add_documents(contents=documents, ids=ids, metadatas=metadatas)
        print(f"Ingested {len(documents)} chunks from {len(set(m['source'] for m in metadatas))} files.")
    else:
        print("No documents found to ingest.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw", help="Directory containing documents.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "chroma", help="Chroma DB directory.")
    args = parser.parse_args()

    os.makedirs(args.db, exist_ok=True)
    ingest(args.input, args.db)


if __name__ == "__main__":
    main()

