"""End-to-end verification run: extract claims, retrieve evidence, score with NLI,
apply the decision rule, and write the two audit logs.

The same classes are meant to be imported by the benchmark scripts, so the proof
of concept and the experiments run one code path.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .audit import (
    STATUS_CONTRADICTING,
    STATUS_NONE,
    STATUS_RETRIEVED,
    STATUS_SUPPORTING,
    AuditLogs,
    ClaimRow,
    EvidenceRow,
    RunRow,
)
from .config import Config
from .decision import CONTRADICTED, INSUFFICIENT, SUPPORTED, decide
from .extract import ClaimExtractor, SkippedSentence
from .nli import NLIModel, make_nli
from .retrieve import DenseEncoder, Retriever, make_encoder
from .textbook import Passage, load_passages


@dataclass
class EvidenceItem:
    rank: int
    passage_id: str
    locator: str
    text: str
    bm25: float
    dense: float
    fused: float
    entailment: float
    contradiction: float
    neutral: float
    status: str


@dataclass
class ClaimResult:
    claim_id: str
    claim: str
    sentence: str
    source_start: int
    source_end: int
    label: str
    review_flag: bool
    entailment: float
    contradiction: float
    reason: str
    latency_ms: float
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    timestamp: str
    submitted_text: str
    pipeline_version: str
    tuned: bool
    backend: str
    extraction_ms: float
    claims: list[ClaimResult]
    skipped: list[SkippedSentence]

    def counts(self) -> dict[str, int]:
        out = {SUPPORTED: 0, CONTRADICTED: 0, INSUFFICIENT: 0}
        for claim in self.claims:
            out[claim.label] += 1
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:8]


class VerificationPipeline:
    def __init__(
        self,
        cfg: Config,
        passages: list[Passage],
        retriever: Retriever,
        nli: NLIModel,
        extractor: ClaimExtractor,
        logs: AuditLogs,
    ) -> None:
        self.cfg = cfg
        self.passages = passages
        self.retriever = retriever
        self.nli = nli
        self.extractor = extractor
        self.logs = logs
        self.pipeline_version = f"poc-{__version__}+{cfg.backend}+{config_hash(cfg)}"

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config, base_dir: str | Path = ".") -> "VerificationPipeline":
        base = Path(base_dir)
        index_dir = base / cfg.index_dir
        passages_path = index_dir / "passages.jsonl"
        embeddings_path = index_dir / "embeddings.npy"
        manifest_path = index_dir / "manifest.json"
        if not passages_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"no index in {index_dir}. Build one first: python -m verifier build-index --textbook <file>"
            )
        encoder: DenseEncoder = make_encoder(cfg)
        if manifest_path.exists():
            built_with = json.loads(manifest_path.read_text(encoding="utf-8")).get("encoder")
            if built_with and built_with != encoder.name:
                raise ValueError(
                    f"the index was built with encoder {built_with!r} but the current backend uses "
                    f"{encoder.name!r}; rebuild the index or change the backend"
                )
        passages = load_passages(passages_path)
        embeddings = np.load(embeddings_path)
        retriever = Retriever(passages, embeddings, encoder, cfg)
        return cls(
            cfg,
            passages,
            retriever,
            make_nli(cfg),
            ClaimExtractor(cfg.spacy_model),
            AuditLogs(base / cfg.log_dir),
        )

    # ------------------------------------------------------------------
    def verify(self, text: str) -> RunResult:
        text = text.strip()
        if not text:
            raise ValueError("no text submitted")
        if len(text) > self.cfg.max_input_chars:
            raise ValueError(f"text is longer than {self.cfg.max_input_chars} characters")

        now = datetime.now(timezone.utc)
        timestamp = now.isoformat(timespec="seconds")
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        started = time.perf_counter()
        extraction = self.extractor.extract(text)
        extraction_ms = (time.perf_counter() - started) * 1000

        record = RunRow(
            verification_run_id=run_id,
            timestamp=timestamp,
            submitted_text=text,
            text_hash=text_hash,
            pipeline_version=self.pipeline_version,
            config_json=json.dumps(self.cfg.to_dict(), sort_keys=True),
        )
        results: list[ClaimResult] = []
        for index, extracted in enumerate(extraction.claims, start=1):
            claim_id = f"C{index:03d}"
            started = time.perf_counter()
            retrieved = self.retriever.retrieve(extracted.text)
            scores = self.nli.score([r.passage.text for r in retrieved], extracted.text)
            decision = decide(scores, self.cfg.tau_entail, self.cfg.tau_contra)
            latency_ms = (time.perf_counter() - started) * 1000

            evidence: list[EvidenceItem] = []
            for i, (item, score) in enumerate(zip(retrieved, scores)):
                if decision.label == SUPPORTED and i == decision.best_entail_index:
                    status = STATUS_SUPPORTING
                elif decision.label == CONTRADICTED and i == decision.best_contra_index:
                    status = STATUS_CONTRADICTING
                elif decision.label == INSUFFICIENT:
                    status = STATUS_NONE
                else:
                    status = STATUS_RETRIEVED
                evidence.append(
                    EvidenceItem(
                        rank=item.rank,
                        passage_id=item.passage.passage_id,
                        locator=item.passage.locator(),
                        text=item.passage.text,
                        bm25=item.bm25,
                        dense=item.dense,
                        fused=item.fused,
                        entailment=score.entailment,
                        contradiction=score.contradiction,
                        neutral=score.neutral,
                        status=status,
                    )
                )
            result = ClaimResult(
                claim_id=claim_id,
                claim=extracted.text,
                sentence=extracted.sentence,
                source_start=extracted.source_start,
                source_end=extracted.source_end,
                label=decision.label,
                review_flag=decision.review_flag,
                entailment=decision.entailment,
                contradiction=decision.contradiction,
                reason=decision.reason,
                latency_ms=latency_ms,
                warnings=extracted.warnings,
                notes=extracted.notes,
                evidence=evidence,
            )
            results.append(result)

            record.claims.append(
                ClaimRow(
                    verification_run_id=run_id,
                    claim_id=claim_id,
                    timestamp=timestamp,
                    text_hash=text_hash,
                    source_start=extracted.source_start,
                    source_end=extracted.source_end,
                    extracted_claim=extracted.text,
                    predicted_label=decision.label,
                    review_flag=decision.review_flag,
                    latency_ms=latency_ms,
                    pipeline_version=self.pipeline_version,
                )
            )
            for item, score, ev in zip(retrieved, scores, evidence):
                record.evidence.append(
                    EvidenceRow(
                        verification_run_id=run_id,
                        claim_id=claim_id,
                        textbook_id=self.cfg.textbook_id,
                        edition=self.cfg.edition,
                        passage_id=item.passage.passage_id,
                        chapter=item.passage.chapter,
                        section=item.passage.section,
                        page_start=item.passage.page_start,
                        page_end=item.passage.page_end,
                        retrieval_rank=item.rank,
                        bm25_score=item.bm25,
                        dense_score=item.dense,
                        fused_score=item.fused,
                        nli_entailment=score.entailment,
                        nli_contradiction=score.contradiction,
                        nli_neutral=score.neutral,
                        evidence_status=ev.status,
                        reason=decision.reason,
                    )
                )

        self.logs.write_run(record)
        return RunResult(
            run_id=run_id,
            timestamp=timestamp,
            submitted_text=text,
            pipeline_version=self.pipeline_version,
            tuned=self.cfg.tuned,
            backend=self.cfg.backend,
            extraction_ms=extraction_ms,
            claims=results,
            skipped=extraction.skipped,
        )
