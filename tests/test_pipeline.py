"""End-to-end tests on the stub backend. They check the wiring, not the quality of the labels."""

import json
import subprocess
import sys

import pytest

from verifier.config import Config
from verifier.decision import LABELS
from verifier.pipeline import VerificationPipeline

from conftest import ROOT, SAMPLE_TEXT, MINI_TEXTBOOK, build_stub_index

TEXT = SAMPLE_TEXT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pipeline(stub_workspace, spacy_model):
    cfg, out = stub_workspace
    return VerificationPipeline.from_config(cfg, out)


def test_run_produces_claims_with_evidence_and_locators(pipeline):
    result = pipeline.verify(TEXT)
    assert len(result.claims) >= 5
    assert [c.claim_id for c in result.claims] == [f"C{i:03d}" for i in range(1, len(result.claims) + 1)]
    for claim in result.claims:
        assert claim.label in LABELS
        assert len(claim.evidence) == pipeline.cfg.top_k
        assert [e.rank for e in claim.evidence] == list(range(1, pipeline.cfg.top_k + 1))
        assert all("Chapter" in e.locator and "p" in e.locator for e in claim.evidence)
        assert claim.latency_ms > 0
        assert claim.review_flag == (claim.label != "Supported")
    assert sum(result.counts().values()) == len(result.claims)


def test_content_is_never_rewritten(pipeline):
    text = "  A relation has a primary key.  "
    result = pipeline.verify(text)
    assert result.submitted_text == text.strip()
    for claim in result.claims:
        assert result.submitted_text[claim.source_start : claim.source_end] == claim.sentence


def test_evidence_status_matches_the_label(pipeline):
    for claim in pipeline.verify(TEXT).claims:
        statuses = [e.status for e in claim.evidence]
        if claim.label == "Supported":
            assert statuses.count("supporting") == 1
        elif claim.label == "Contradicted":
            assert statuses.count("contradicting") == 1
        else:
            assert set(statuses) == {"no_sufficient_evidence"}


def test_both_logs_are_written_and_linked(fresh_workspace, spacy_model):
    cfg, out = fresh_workspace
    pipe = VerificationPipeline.from_config(cfg, out)
    result = pipe.verify(TEXT)
    logs = pipe.logs
    claim_rows = logs.read_claims(result.run_id)
    evidence_rows = logs.read_evidence(result.run_id)
    assert [r["claim_id"] for r in claim_rows] == [c.claim_id for c in result.claims]
    assert len(evidence_rows) == len(result.claims) * cfg.top_k
    assert {r["claim_id"] for r in evidence_rows} == {r["claim_id"] for r in claim_rows}
    assert {r["predicted_label"] for r in claim_rows} <= set(LABELS)
    # Table 3.8 fields are present in each log
    assert {"extracted_claim", "predicted_label", "review_flag", "latency_ms", "pipeline_version"} <= set(claim_rows[0])
    assert {"passage_id", "chapter", "section", "page_start", "retrieval_rank", "fused_score",
            "nli_entailment", "nli_contradiction", "evidence_status", "reason"} <= set(evidence_rows[0])


def test_evidence_log_does_not_contain_textbook_text(fresh_workspace, spacy_model):
    cfg, out = fresh_workspace
    pipe = VerificationPipeline.from_config(cfg, out)
    result = pipe.verify(TEXT)
    evidence_bytes = pipe.logs.evidence_path.read_bytes()
    for item in result.claims[0].evidence:
        assert item.text[:60].encode() not in evidence_bytes
    assert TEXT[:40].encode() not in evidence_bytes


def test_empty_and_oversized_input_are_rejected(pipeline):
    with pytest.raises(ValueError, match="no text"):
        pipeline.verify("   ")
    with pytest.raises(ValueError, match="longer than"):
        pipeline.verify("x" * (pipeline.cfg.max_input_chars + 1))


def test_text_without_claims_gives_an_empty_run(pipeline):
    result = pipeline.verify("What is a primary key?")
    assert result.claims == [] and result.skipped


def test_same_input_gives_same_labels(pipeline):
    first = [(c.claim, c.label, [e.passage_id for e in c.evidence]) for c in pipeline.verify(TEXT).claims]
    second = [(c.claim, c.label, [e.passage_id for e in c.evidence]) for c in pipeline.verify(TEXT).claims]
    assert first == second


def test_result_is_json_serialisable(pipeline):
    json.dumps(pipeline.verify(TEXT).to_dict())


def test_version_string_names_the_backend_and_changes_with_the_config(stub_workspace, spacy_model):
    cfg, out = stub_workspace
    a = VerificationPipeline.from_config(cfg, out).pipeline_version
    b = VerificationPipeline.from_config(Config(**{**cfg.to_dict(), "alpha": 0.7}), out).pipeline_version
    assert "stub" in a and a != b


def test_index_built_with_a_different_encoder_is_refused(fresh_workspace):
    cfg, out = fresh_workspace
    manifest_path = out / "index" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoder"] = "sentence-transformers/all-MiniLM-L6-v2"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rebuild the index"):
        VerificationPipeline.from_config(cfg, out)


def test_missing_index_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-index"):
        VerificationPipeline.from_config(Config(backend="stub", index_dir="nowhere"), tmp_path)


def test_config_rejects_unknown_keys_and_bad_values(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"alpa": 0.5}))
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(bad)
    with pytest.raises(ValueError):
        Config(alpha=1.5)
    with pytest.raises(ValueError):
        Config(tau_entail=0.3)
    with pytest.raises(ValueError):
        Config(backend="magic")


def test_command_line_end_to_end(tmp_path, spacy_model):
    cfg_path = tmp_path / "cfg.json"
    Config(backend="stub", index_dir=str(tmp_path / "idx"), log_dir=str(tmp_path / "log")).save(cfg_path)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "verifier", *args], cwd=ROOT, capture_output=True, text=True, timeout=120
        )

    built = run("build-index", "--config", str(cfg_path), "--textbook", str(MINI_TEXTBOOK))
    assert built.returncode == 0, built.stderr
    assert "Passages per chapter" in built.stdout

    verified = run("verify", "--config", str(cfg_path), "--file", str(SAMPLE_TEXT), "--json")
    assert verified.returncode == 0, verified.stderr
    payload = json.loads(verified.stdout[verified.stdout.index("{"):])
    assert payload["backend"] == "stub" and payload["claims"]

    shown = run("show-log", "--config", str(cfg_path))
    assert shown.returncode == 0 and payload["run_id"] in shown.stdout
