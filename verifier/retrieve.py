"""Hybrid retrieval: BM25Okapi + dense embeddings, fused after per-claim min-max
normalisation (thesis Section 3.2.3).

    F = alpha * B_norm + (1 - alpha) * D_norm

``alpha`` is the weight on the normalised BM25 score. Both score vectors cover
every passage in the corpus, are normalised per claim, and the top-k passages by
fused score are returned with their raw and fused scores for the audit log.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Config
from .textbook import Passage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalise one score vector; a constant vector maps to zeros."""
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


# ---------------------------------------------------------------------------
# dense encoders
# ---------------------------------------------------------------------------
class DenseEncoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:  # returns L2-normalised rows
        ...


class SentenceTransformerEncoder:
    """sentence-transformers/all-MiniLM-L6-v2 (pretrained, frozen)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("pip install sentence-transformers to use the real backend") from exc
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEncoder:
    """Deterministic bag-of-words hashing encoder. STUB: for wiring tests and demos only."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"stub-hashing-{dim}"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                matrix[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def make_encoder(cfg: Config) -> DenseEncoder:
    if cfg.backend == "stub":
        return HashingEncoder()
    return SentenceTransformerEncoder(cfg.dense_model)


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------
@dataclass
class Retrieved:
    rank: int
    passage: Passage
    bm25: float          # raw BM25Okapi score
    dense: float         # raw cosine similarity
    bm25_norm: float
    dense_norm: float
    fused: float


class Retriever:
    def __init__(
        self,
        passages: list[Passage],
        embeddings: np.ndarray,
        encoder: DenseEncoder,
        cfg: Config,
    ) -> None:
        from rank_bm25 import BM25Okapi

        if len(passages) != len(embeddings):
            raise ValueError("embeddings do not match passages; rebuild the index")
        self.passages = passages
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.encoder = encoder
        self.cfg = cfg
        self._bm25 = BM25Okapi(
            [tokenize(p.text) for p in passages],
            k1=cfg.bm25_k1,
            b=cfg.bm25_b,
            epsilon=cfg.bm25_epsilon,
        )

    def retrieve(self, claim: str) -> list[Retrieved]:
        bm25 = np.asarray(self._bm25.get_scores(tokenize(claim)), dtype=float)
        query = self.encoder.encode([claim])[0]
        dense = (self.embeddings @ query).astype(float)
        bm25_norm, dense_norm = minmax(bm25), minmax(dense)
        fused = self.cfg.alpha * bm25_norm + (1.0 - self.cfg.alpha) * dense_norm
        # stable order: ties broken by passage position
        order = np.lexsort((np.arange(len(fused)), -fused))[: self.cfg.top_k]
        return [
            Retrieved(
                rank=rank,
                passage=self.passages[i],
                bm25=float(bm25[i]),
                dense=float(dense[i]),
                bm25_norm=float(bm25_norm[i]),
                dense_norm=float(dense_norm[i]),
                fused=float(fused[i]),
            )
            for rank, i in enumerate(order, start=1)
        ]
