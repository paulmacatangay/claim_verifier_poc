"""Shared fixtures. Every test runs on the stub backend, so no pretrained model is needed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verifier.config import Config
from verifier.retrieve import make_encoder
from verifier.textbook import build_manifest, build_passages, save_passages

ROOT = Path(__file__).resolve().parent.parent
MINI_TEXTBOOK = ROOT / "samples" / "mini_textbook.txt"
SAMPLE_TEXT = ROOT / "samples" / "sample_ai_text.txt"


def build_stub_index(out: Path, textbook: Path = MINI_TEXTBOOK) -> Config:
    cfg = Config(backend="stub", index_dir=str(out / "index"), log_dir=str(out / "logs"))
    passages, info = build_passages(textbook, cfg)
    encoder = make_encoder(cfg)
    index = out / "index"
    index.mkdir(parents=True, exist_ok=True)
    save_passages(passages, index / "passages.jsonl")
    np.save(index / "embeddings.npy", encoder.encode([p.text for p in passages]))
    manifest = build_manifest(textbook, cfg, passages, info, encoder.name)
    (index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cfg


@pytest.fixture(scope="session")
def stub_workspace(tmp_path_factory) -> tuple[Config, Path]:
    out = tmp_path_factory.mktemp("poc")
    return build_stub_index(out), out


@pytest.fixture()
def fresh_workspace(tmp_path) -> tuple[Config, Path]:
    """A private index and private logs, for tests that inspect or change the logs."""
    return build_stub_index(tmp_path), tmp_path


@pytest.fixture(scope="session")
def spacy_model() -> str:
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed")
    return "en_core_web_sm"
