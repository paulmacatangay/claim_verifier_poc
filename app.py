"""Streamlit interface for the proof of concept.

Run:  streamlit run app.py
Use a different configuration with:  VERIFIER_CONFIG=config.stub.json streamlit run app.py
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pandas as pd
import streamlit as st

from verifier.audit import AuditLogs
from verifier.config import Config
from verifier.decision import CONTRADICTED, INSUFFICIENT, SUPPORTED
from verifier.pipeline import VerificationPipeline

BASE = Path(__file__).resolve().parent
CONFIG_FILE = os.environ.get("VERIFIER_CONFIG", "config.json")
SAMPLE = BASE / "samples" / "sample_ai_text.txt"

st.set_page_config(page_title="Claim Verification POC", layout="wide")


@st.cache_resource(show_spinner="Loading the index and models...")
def load_pipeline(config_file: str) -> VerificationPipeline:
    path = BASE / config_file
    cfg = Config.load(path) if path.exists() else Config()
    return VerificationPipeline.from_config(cfg, BASE)


def show_table(frame: pd.DataFrame) -> None:
    """Full-width table; newer Streamlit uses width="stretch", older versions use_container_width."""
    try:
        st.dataframe(frame, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(frame, hide_index=True, use_container_width=True)


def label_markup(label: str) -> str:
    return {
        SUPPORTED: ":green[**Supported**]",
        CONTRADICTED: ":red[**Contradicted**]",
        INSUFFICIENT: ":orange[**Insufficient Evidence**]",
    }[label]


try:
    pipeline = load_pipeline(CONFIG_FILE)
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.title("Claim Verification POC")
    st.error(str(error))
    st.stop()

cfg = pipeline.cfg
st.title("Claim Verification POC")
st.caption(
    f"Checks the claims in AI-generated text against one textbook only: "
    f"{cfg.textbook_id}, {cfg.edition} edition. A label reflects what that textbook establishes, "
    "and Insufficient Evidence does not mean that a claim is false."
)
if cfg.backend == "stub":
    st.error(
        "STUB MODE: no pretrained models are loaded. Labels only check that the software is wired "
        "together and must not be reported as results."
    )
elif not cfg.tuned:
    st.warning(
        "The fusion weight and the NLI thresholds are placeholder defaults. Replace them with the "
        "validation-selected values in the configuration file before any reported run."
    )

with st.sidebar:
    st.header("Active settings")
    st.write(f"Backend: **{cfg.backend}**")
    st.write(f"Passages indexed: **{len(pipeline.passages)}**")
    st.write(f"Dense model: `{cfg.dense_model}`" if cfg.backend == "real" else "Dense model: stub")
    st.write(f"NLI model: `{cfg.nli_model}`" if cfg.backend == "real" else "NLI model: stub")
    st.write(f"BM25 weight (alpha): **{cfg.alpha}**")
    st.write(f"Thresholds: entailment **{cfg.tau_entail}**, contradiction **{cfg.tau_contra}**")
    st.write(f"Passages per claim: **{cfg.top_k}**")
    st.caption(f"Version: {pipeline.pipeline_version}")
    st.caption("Parameters are fixed here. Change them in the configuration file so that every run is reproducible.")

verify_tab, logs_tab = st.tabs(["Verify", "Audit logs"])

with verify_tab:
    if "text" not in st.session_state:
        st.session_state["text"] = ""
    left, right = st.columns([3, 1])
    with right:
        if st.button("Load sample text", disabled=not SAMPLE.exists()):
            st.session_state["text"] = SAMPLE.read_text(encoding="utf-8")
            st.rerun()
        uploaded = st.file_uploader("or upload a .txt file", type=["txt"], label_visibility="visible")
        if uploaded is not None:
            content = uploaded.getvalue().decode("utf-8", errors="replace")
            if content != st.session_state.get("uploaded_content"):
                st.session_state["uploaded_content"] = content
                st.session_state["text"] = content
                st.rerun()
    with left:
        st.text_area(
            "AI-generated lesson, explanation or quiz text",
            key="text",
            height=220,
            max_chars=cfg.max_input_chars,
        )
    if st.button("Verify claims", type="primary", disabled=not st.session_state["text"].strip()):
        with st.spinner("Extracting claims and checking them against the textbook..."):
            try:
                st.session_state["result"] = pipeline.verify(st.session_state["text"])
            except ValueError as error:
                st.error(str(error))

    result = st.session_state.get("result")
    if result is not None:
        counts = result.counts()
        flagged = sum(c.review_flag for c in result.claims)
        latencies = [c.latency_ms for c in result.claims]
        cols = st.columns(6)
        cols[0].metric("Claims", len(result.claims))
        cols[1].metric("Supported", counts[SUPPORTED])
        cols[2].metric("Contradicted", counts[CONTRADICTED])
        cols[3].metric("Insufficient Evidence", counts[INSUFFICIENT])
        cols[4].metric("Flagged for review", flagged)
        cols[5].metric("Median ms per claim", f"{statistics.median(latencies):.0f}" if latencies else "-")
        st.caption(
            f"Run ID {result.run_id}. Claim extraction took {result.extraction_ms:.0f} ms. "
            "Contradicted and Insufficient Evidence claims are flagged for instructor review; "
            "the tool does not correct or rewrite content."
        )

        if result.claims:
            show_table(
                pd.DataFrame(
                    {
                        "Claim ID": [c.claim_id for c in result.claims],
                        "Label": [c.label for c in result.claims],
                        "Review": ["Review" if c.review_flag else "" for c in result.claims],
                        "Claim": [c.claim for c in result.claims],
                        "Best passage": [c.evidence[0].locator if c.evidence else "" for c in result.claims],
                    }
                )
            )
        else:
            st.info("No verifiable claims were found in the text.")

        for claim in result.claims:
            with st.expander(f"{claim.claim_id}: {claim.label}  |  {claim.claim}", expanded=claim.review_flag):
                st.markdown(f"{label_markup(claim.label)}" + ("  •  **Flag for instructor review**" if claim.review_flag else ""))
                st.write(claim.claim)
                st.caption(f"From the sentence: {claim.sentence}")
                for warning in claim.warnings:
                    st.warning(warning)
                for note in claim.notes:
                    st.caption(f"Context added: {note}")
                st.caption(claim.reason)
                for item in claim.evidence:
                    st.markdown(
                        f"**#{item.rank}  {item.passage_id}**  —  {item.locator}  —  *{item.status.replace('_', ' ')}*"
                    )
                    st.caption(
                        f"BM25 {item.bm25:.2f}  |  dense {item.dense:.2f}  |  fused {item.fused:.2f}  |  "
                        f"entailment {item.entailment:.2f}  |  contradiction {item.contradiction:.2f}  |  "
                        f"neutral {item.neutral:.2f}"
                    )
                    st.text(item.text)

        if result.skipped:
            with st.expander(f"{len(result.skipped)} sentence(s) not treated as claims"):
                for skipped in result.skipped:
                    st.write(f"- {skipped.text}  ({skipped.reason})")

        st.download_button(
            "Download this run as JSON",
            data=json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            file_name=f"{result.run_id}.json",
            mime="application/json",
        )

with logs_tab:
    st.write(
        "Two separate logs are kept. They are joined only by the run ID and the claim ID, and the "
        "evidence log stores passage identifiers and scores, not textbook text."
    )
    logs: AuditLogs = pipeline.logs
    runs = logs.list_runs(50)
    if not runs:
        st.info("No runs have been logged yet.")
    else:
        choice = st.selectbox(
            "Run",
            [r["verification_run_id"] for r in runs],
            format_func=lambda rid: next(
                f"{r['verification_run_id']}  ({r['n_claims']} claims)" for r in runs if r["verification_run_id"] == rid
            ),
        )
        run_row = logs.read_run(choice)
        st.subheader("Content Examination Log")
        st.caption(f"Submitted text hash: {run_row['text_hash']}  |  version: {run_row['pipeline_version']}")
        show_table(pd.DataFrame(logs.read_claims(choice)))
        with st.expander("Submitted text"):
            st.text(run_row["submitted_text"])
        st.subheader("Source Evidence Log")
        show_table(pd.DataFrame(logs.read_evidence(choice)))
