import sqlite3
import stat

import pytest

from verifier.audit import AuditLogs, ClaimRow, EvidenceRow, RunRow


def make_run(run_id="R1", n_claims=2):
    run = RunRow(
        verification_run_id=run_id,
        timestamp="2026-01-01T00:00:00+00:00",
        submitted_text="SECRET SUBMITTED TEXT",
        text_hash="abc",
        pipeline_version="v",
        config_json="{}",
    )
    for i in range(1, n_claims + 1):
        cid = f"C{i:03d}"
        run.claims.append(
            ClaimRow(run_id, cid, run.timestamp, "abc", 0, 5, f"claim {i}", "Supported", False, 1.5, "v")
        )
        run.evidence.append(
            EvidenceRow(run_id, cid, "Book", "7th", f"p{i}", "Chapter 1", "1.1", 3, 3, 1,
                        1.0, 0.5, 0.9, 0.9, 0.05, 0.05, "supporting", "reason")
        )
    return run


def tables(path):
    with sqlite3.connect(path) as db:
        return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_two_physically_separate_files_with_the_right_tables(tmp_path):
    logs = AuditLogs(tmp_path)
    assert logs.content_path != logs.evidence_path
    assert logs.content_path.exists() and logs.evidence_path.exists()
    assert tables(logs.content_path) == {"runs", "claims"}
    assert tables(logs.evidence_path) == {"evidence"}


def test_evidence_log_holds_no_passage_text_and_no_submitted_text(tmp_path):
    logs = AuditLogs(tmp_path)
    logs.write_run(make_run())
    with sqlite3.connect(logs.evidence_path) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(evidence)")}
    assert not columns & {"passage_text", "text", "submitted_text", "extracted_claim"}
    assert b"SECRET SUBMITTED TEXT" not in logs.evidence_path.read_bytes()


def test_submitted_text_is_stored_once_per_run_not_per_claim(tmp_path):
    logs = AuditLogs(tmp_path)
    logs.write_run(make_run(n_claims=3))
    assert logs.content_path.read_bytes().count(b"SECRET SUBMITTED TEXT") == 1


def test_logs_join_only_on_run_and_claim_ids(tmp_path):
    logs = AuditLogs(tmp_path)
    logs.write_run(make_run("R1"))
    logs.write_run(make_run("R2", n_claims=1))
    joined = logs.joined("R1")
    assert [r["claim_id"] for r in joined] == ["C001", "C002"]
    assert [[e["passage_id"] for e in r["evidence"]] for r in joined] == [["p1"], ["p2"]]
    assert all(e["verification_run_id"] == "R1" for r in joined for e in r["evidence"])
    assert len(logs.read_evidence("R2")) == 1


def test_runs_are_listed_with_claim_counts(tmp_path):
    logs = AuditLogs(tmp_path)
    logs.write_run(make_run("R1", n_claims=2))
    assert logs.list_runs()[0]["n_claims"] == 2
    assert logs.read_run("R1")["submitted_text"] == "SECRET SUBMITTED TEXT"
    assert logs.read_run("missing") is None


def test_log_files_are_private(tmp_path):
    logs = AuditLogs(tmp_path)
    for path in (logs.content_path, logs.evidence_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_purge_removes_old_runs_from_both_logs(tmp_path):
    logs = AuditLogs(tmp_path)
    old = make_run("OLD")
    old.timestamp = "2020-01-01T00:00:00+00:00"
    logs.write_run(old)
    logs.write_run(make_run("NEW"))
    # make_run uses a fixed 2026 timestamp, so use a cutoff between the two
    removed = logs.purge_older_than(365 * 2)
    assert removed == 1
    assert logs.read_run("OLD") is None and logs.read_evidence("OLD") == []
    assert logs.read_run("NEW") is not None and len(logs.read_evidence("NEW")) == 2
