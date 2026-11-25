"""Configuration for the proof of concept.

Every tunable value lives here so the POC and the benchmark scripts read the same
settings. Values marked PLACEHOLDER must be replaced with the validation-selected
values from the experiments (Section 3.5.2) before any reported run; set
``tuned`` to true only after that.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

BACKENDS = ("real", "stub")


@dataclass
class Config:
    # --- corpus and segmentation (Section 3.2.2) -----------------------------
    textbook_id: str = "Database System Concepts"
    edition: str = "7th"
    passage_min_words: int = 100
    passage_max_words: int = 150
    passage_drop_below_words: int = 30  # headers, captions and stray fragments
    first_pdf_page: int = 1             # 1-based PDF page where the body starts
    page_offset: int = 0                # printed page = PDF page + page_offset
    toc_chapter_level: int = 1          # outline level that holds chapters
    toc_section_level: int = 2          # outline level that holds sections

    # --- retrieval (Section 3.2.3) ------------------------------------------
    top_k: int = 5
    alpha: float = 0.5                  # PLACEHOLDER: weight on normalised BM25
    bm25_k1: float = 1.5                # rank_bm25 0.2.2 BM25Okapi defaults
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- NLI and decision rule (Section 3.2.4, Eq. 3.11) ---------------------
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    tau_entail: float = 0.80            # PLACEHOLDER: entailment threshold
    tau_contra: float = 0.80            # PLACEHOLDER: contradiction threshold

    # --- claim extraction -----------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    max_input_chars: int = 20000

    # --- runtime ---------------------------------------------------------------
    backend: str = "real"               # "real" = pretrained models, "stub" = wiring tests only
    index_dir: str = "data/index"
    log_dir: str = "data/logs"
    tuned: bool = False                 # true only after alpha and thresholds are validation-selected
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        for name in ("tau_entail", "tau_contra"):
            value = getattr(self, name)
            if not 0.5 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.50, 1.00), got {value}")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < self.passage_min_words <= self.passage_max_words:
            raise ValueError("need 0 < passage_min_words <= passage_max_words")

    # ---------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load overrides from a JSON file; unknown keys are an error."""
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
