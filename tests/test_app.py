"""Drives the Streamlit page headlessly on the stub backend."""

import pytest

from verifier.config import Config

from conftest import ROOT, build_stub_index

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture()
def app(tmp_path, monkeypatch, spacy_model):
    cfg = build_stub_index(tmp_path)
    cfg_path = tmp_path / "config.stub.json"
    cfg.save(cfg_path)
    monkeypatch.setenv("VERIFIER_CONFIG", str(cfg_path))
    # st.cache_resource is keyed by the config path, so every test has its own pipeline
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_page_loads_and_shows_the_stub_banner(app):
    app.run()
    assert not app.exception
    assert any("STUB MODE" in e.value for e in app.error)
    assert button(app, "Verify claims").disabled  # nothing to verify yet


def test_sample_text_can_be_verified_and_shows_evidence(app):
    app.run()
    button(app, "Load sample text").click().run()
    assert app.text_area[0].value.strip()
    button(app, "Verify claims").click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert int(metrics["Claims"]) >= 5
    total = sum(int(metrics[k]) for k in ("Supported", "Contradicted", "Insufficient Evidence"))
    assert total == int(metrics["Claims"])
    assert app.expander, "each claim has an expander with its evidence"
    texts = [m.value for e in app.expander for m in e.markdown] + [c.value for e in app.expander for c in e.caption]
    assert any("Chapter" in x and "p." in x for x in texts), "evidence must show chapter, section and page"


def test_audit_log_tab_lists_the_run_after_verification(app):
    app.run()
    button(app, "Load sample text").click().run()
    button(app, "Verify claims").click().run()
    assert app.selectbox, "the log tab should offer the run that was just written"
    assert len(app.dataframe) >= 3  # summary table, content log, evidence log


def test_missing_index_shows_an_error_not_a_traceback(tmp_path, monkeypatch):
    cfg_path = tmp_path / "c.json"
    Config(backend="stub", index_dir=str(tmp_path / "none"), log_dir=str(tmp_path / "l")).save(cfg_path)
    monkeypatch.setenv("VERIFIER_CONFIG", str(cfg_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("build-index" in e.value for e in at.error)
