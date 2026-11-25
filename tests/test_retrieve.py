from dataclasses import replace

import numpy as np
import pytest

from verifier.config import Config
from verifier.retrieve import HashingEncoder, Retriever, minmax, tokenize
from verifier.textbook import Passage

PASSAGES = [
    Passage("p1", "A primary key uniquely identifies each tuple in a relation.", "Chapter 1", "1.1", 1, 1, 10),
    Passage("p2", "Normalization removes redundancy by decomposing relations into smaller relations.", "Chapter 2", "2.1", 2, 2, 9),
    Passage("p3", "Transactions are atomic, consistent, isolated and durable.", "Chapter 3", "3.1", 3, 3, 8),
    Passage("p4", "A view is a virtual relation defined by a query.", "Chapter 1", "1.2", 1, 1, 10),
    Passage("p5", "Two-phase locking guarantees conflict serializability of schedules.", "Chapter 3", "3.2", 3, 3, 8),
    Passage("p6", "Indexes speed up the lookup of tuples by a search key.", "Chapter 1", "1.3", 2, 2, 10),
]


def make(alpha=0.5, top_k=5):
    cfg = Config(backend="stub", alpha=alpha, top_k=top_k)
    encoder = HashingEncoder()
    embeddings = encoder.encode([p.text for p in PASSAGES])
    return Retriever(PASSAGES, embeddings, encoder, cfg)


def test_minmax_range_and_constant_vector():
    out = minmax(np.array([2.0, 4.0, 6.0]))
    assert out.min() == 0.0 and out.max() == 1.0 and out[1] == pytest.approx(0.5)
    assert np.all(minmax(np.array([3.0, 3.0, 3.0])) == 0.0)  # no spread, no signal


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("A Primary-Key, e.g. K1!") == ["a", "primary", "key", "e", "g", "k1"]


def test_hashing_encoder_is_deterministic_and_normalised():
    a = HashingEncoder().encode(["primary key", "primary key"])
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_returns_top_k_in_rank_order():
    hits = make(top_k=3).retrieve("What uniquely identifies a tuple in a relation?")
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.fused for h in hits] == sorted([h.fused for h in hits], reverse=True)
    assert hits[0].passage.passage_id == "p1"


def test_alpha_one_is_bm25_only_and_alpha_zero_is_dense_only():
    claim = "Two-phase locking guarantees serializability."
    bm25_only = make(alpha=1.0).retrieve(claim)
    dense_only = make(alpha=0.0).retrieve(claim)
    assert [h.fused for h in bm25_only] == pytest.approx([h.bm25_norm for h in bm25_only])
    assert [h.fused for h in dense_only] == pytest.approx([h.dense_norm for h in dense_only])
    assert bm25_only[0].passage.passage_id == "p5"


def test_fusion_formula():
    for hit in make(alpha=0.3).retrieve("A view is defined by a query."):
        assert hit.fused == pytest.approx(0.3 * hit.bm25_norm + 0.7 * hit.dense_norm)


def test_ties_are_broken_by_passage_order():
    hits = make().retrieve("zzzz qqqq")  # no overlap with anything: all scores tie
    ids = [h.passage.passage_id for h in hits]
    assert ids == sorted(ids)


def test_mismatched_embeddings_are_rejected():
    cfg = Config(backend="stub")
    encoder = HashingEncoder()
    with pytest.raises(ValueError):
        Retriever(PASSAGES, encoder.encode(["only one"]), encoder, cfg)
