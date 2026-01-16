from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer


SUPPORTED_EXT = {".txt", ".md", ".rst"}


@dataclass(frozen=True)
class RagChunk:
    source: str
    chunk_id: int
    text: str


@dataclass
class RagIndex:
    backend: str  # "tfidf" or "sentence_transformers"
    vectorizer: object
    matrix: np.ndarray
    chunks: List[RagChunk]

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        dump(self, index_dir / "rag_index.joblib")

    @staticmethod
    def load(index_dir: str | Path) -> "RagIndex":
        index_dir = Path(index_dir)
        obj = load(index_dir / "rag_index.joblib")
        if not isinstance(obj, RagIndex):
            raise TypeError("Loaded object is not a RagIndex")
        return obj

    def query(self, question: str, *, top_k: int = 5) -> List[Tuple[float, RagChunk]]:
        q = question.strip()
        if not q:
            return []

        if self.backend == "tfidf":
            qv = self.vectorizer.transform([q]).toarray().astype(float)
            qn = np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
            qv = qv / qn
            M = self.matrix
        elif self.backend == "sentence_transformers":
            qv = self.vectorizer.encode([q], normalize_embeddings=True)
            M = self.matrix
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # cosine similarity with normalized vectors
        sims = (M @ qv.T).reshape(-1)
        if len(sims) == 0:
            return []
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.chunks[i]) for i in idx]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Chunk text by character count with overlap.

    This is intentionally simple and robust for mixed doc types.
    """

    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
        if j == len(text):
            break
    return chunks


def build_index(
    docs_dir: str | Path,
    *,
    backend: str = "tfidf",
    chunk_size: int = 800,
    overlap: int = 120,
) -> RagIndex:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(docs_dir)

    chunks: List[RagChunk] = []

    paths = [p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    paths.sort()

    for p in paths:
        txt = _read_text_file(p)
        for k, c in enumerate(_chunk_text(txt, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(RagChunk(source=str(p), chunk_id=k, text=c))

    if not chunks:
        raise ValueError(f"No supported documents found in {docs_dir}. Supported: {sorted(SUPPORTED_EXT)}")

    texts = [c.text for c in chunks]

    if backend == "tfidf":
        vec = TfidfVectorizer(max_features=20000)
        X = vec.fit_transform(texts).toarray().astype(float)
        # Normalize rows for cosine sim via dot product
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        return RagIndex(backend=backend, vectorizer=vec, matrix=Xn, chunks=chunks)

    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install -e .[embeddings]"
            ) from e

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts, normalize_embeddings=True)
        return RagIndex(backend=backend, vectorizer=model, matrix=np.asarray(emb, dtype=float), chunks=chunks)

    raise ValueError(f"Unknown backend: {backend}")
