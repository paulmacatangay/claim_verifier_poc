"""Two physically separate audit logs (thesis Section 3.8, Table 3.8).

* Content Examination Log (``content_examination.sqlite``): what was submitted,
  which claims were extracted, the predicted label and the review flag.
* Source Evidence Log (``source_evidence.sqlite``): which textbook passages were
  retrieved and scored, and why a label was assigned.

The two files are joined only by ``verification_run_id`` and ``claim_id``. The
evidence log stores passage identifiers and scores, never passage text, so the
licensed textbook is not copied into either log. For Insufficient Evidence
claims the evidence log keeps the attempted retrieval ranks with a
no-sufficient-evidence reason instead of inventing a supporting passage.

The submitted text is stored once per run (``runs`` table) and each claim row
carries the text hash, so the text is not duplicated for every claim.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

CONTENT_DB = "content_examination.sqlite"
EVIDENCE_DB = "source_evidence.sqlite"

_CONTENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    verification_run_id TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    submitted_text      TEXT NOT NULL,
    text_hash           TEXT NOT NULL,
    pipeline_version    TEXT NOT NULL,
    config_json         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    verification_run_id TEXT NOT NULL,
    claim_id            TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    text_hash           TEXT NOT NULL,
    source_start        INTEGER,
    source_end          INTEGER,
    extracted_claim     TEXT NOT NULL,
    predicted_label     TEXT NOT NULL,
    review_flag         INTEGER NOT NULL,
    latency_ms          REAL,
    pipeline_version    TEXT NOT NULL,
    PRIMARY KEY (verification_run_id, claim_id)
);
"""

_EVIDENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    verification_run_id TEXT NOT NULL,
    claim_id            TEXT NOT NULL,
    textbook_id         TEXT NOT NULL,
    edition             TEXT NOT NULL,
    passage_id          TEXT NOT NULL,
    chapter             TEXT,
    section             TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    retrieval_rank      INTEGER NOT NULL,
    bm25_score          REAL,
    dense_score         REAL,
    fused_score         REAL,
    nli_entailment      REAL,
    nli_contradiction   REAL,
    nli_neutral         REAL,
    evidence_status     TEXT NOT NULL,
    reason              TEXT,
    PRIMARY KEY (verification_run_id, claim_id, retrieval_rank)
);
"""

STATUS_SUPPORTING = "supporting"
STATUS_CONTRADICTING = "contradicting"
STATUS_RETRIEVED = "retrieved"
STATUS_NONE = "no_sufficient_evidence"


@dataclass
class ClaimRow:
    verification_run_id: str
    claim_id: str
    timestamp: str
    text_hash: str
    source_start: int
    source_end: int
    extracted_claim: str
    predicted_label: str
    review_flag: bool
    latency_ms: float
    pipeline_version: str


@dataclass
class EvidenceRow:
    verification_run_id: str
    claim_id: str
    textbook_id: str
    edition: str
    passage_id: str
    chapter: str | None
    section: str | None
    page_start: int
    page_end: int
    retrieval_rank: int
    bm25_score: float
    dense_score: float
    fused_score: float
    nli_entailment: float
    nli_contradiction: float
    nli_neutral: float
    evidence_status: str
    reason: str


@dataclass
class RunRow:
    verification_run_id: str
    timestamp: str
    submitted_text: str
    text_hash: str
    pipeline_version: str
    config_json: str
    claims: list[ClaimRow] = field(default_factory=list)
    evidence: list[EvidenceRow] = field(default_factory=list)


def _insert(table: str, row: object) -> tuple[str, list]:
    data = {k: (int(v) if isinstance(v, bool) else v) for k, v in vars(row).items() if k not in ("claims", "evidence")}
    columns = ", ".join(data)
    marks = ", ".join("?" for _ in data)
    return f"INSERT INTO {table} ({columns}) VALUES ({marks})", list(data.values())


class AuditLogs:
    """Writes and reads the two separate logs."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.content_path = self.log_dir / CONTENT_DB
        self.evidence_path = self.log_dir / EVIDENCE_DB
        with self._connect(self.content_path) as db:
            db.executescript(_CONTENT_SCHEMA)
        with self._connect(self.evidence_path) as db:
            db.executescript(_EVIDENCE_SCHEMA)
        for path in (self.content_path, self.evidence_path):
            with contextlib.suppress(OSError):  # best effort; not supported on every platform
                os.chmod(path, 0o600)

    @staticmethod
    @contextlib.contextmanager
    def _connect(path: Path):
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    # -- writing --------------------------------------------------------------
    def write_run(self, run: RunRow) -> None:
        with self._connect(self.content_path) as db:
            sql, values = _insert("runs", run)
            db.execute(sql, values)
            for claim in run.claims:
                sql, values = _insert("claims", claim)
                db.execute(sql, values)
        with self._connect(self.evidence_path) as db:
            for row in run.evidence:
                sql, values = _insert("evidence", row)
                db.execute(sql, values)

    # -- reading --------------------------------------------------------------
    def list_runs(self, limit: int = 20) -> list[dict]:
        with self._connect(self.content_path) as db:
            rows = db.execute(
                "SELECT r.verification_run_id, r.timestamp, r.pipeline_version, "
                "(SELECT COUNT(*) FROM claims c WHERE c.verification_run_id = r.verification_run_id) AS n_claims "
                "FROM runs r ORDER BY r.timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def read_claims(self, run_id: str) -> list[dict]:
        with self._connect(self.content_path) as db:
            rows = db.execute(
                "SELECT * FROM claims WHERE verification_run_id = ? ORDER BY claim_id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def read_run(self, run_id: str) -> dict | None:
        with self._connect(self.content_path) as db:
            row = db.execute("SELECT * FROM runs WHERE verification_run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def read_evidence(self, run_id: str, claim_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM evidence WHERE verification_run_id = ?"
        params: list = [run_id]
        if claim_id is not None:
            sql += " AND claim_id = ?"
            params.append(claim_id)
        sql += " ORDER BY claim_id, retrieval_rank"
        with self._connect(self.evidence_path) as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def joined(self, run_id: str) -> list[dict]:
        """Join the two logs in Python on run ID and claim ID (they are separate files)."""
        evidence = self.read_evidence(run_id)
        by_claim: dict[str, list[dict]] = {}
        for row in evidence:
            by_claim.setdefault(row["claim_id"], []).append(row)
        return [{**claim, "evidence": by_claim.get(claim["claim_id"], [])} for claim in self.read_claims(run_id)]

    # -- retention ------------------------------------------------------------
    def purge_older_than(self, days: int) -> int:
        """Delete runs older than ``days`` from both logs. Returns the number of runs removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._connect(self.content_path) as db:
            ids = [r[0] for r in db.execute("SELECT verification_run_id FROM runs WHERE timestamp < ?", (cutoff,))]
            for run_id in ids:
                db.execute("DELETE FROM claims WHERE verification_run_id = ?", (run_id,))
                db.execute("DELETE FROM runs WHERE verification_run_id = ?", (run_id,))
        with self._connect(self.evidence_path) as db:
            for run_id in ids:
                db.execute("DELETE FROM evidence WHERE verification_run_id = ?", (run_id,))
        return len(ids)
