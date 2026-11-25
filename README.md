# Claim Verification POC

Working prototype for the thesis *An Evidence-Grounded Verification Pipeline for AI-Generated Academic Content Using Claim-Level Natural Language Inference*.

It takes AI-generated text, splits it into claims, finds the closest passages in **one textbook only**, and labels each claim **Supported**, **Contradicted** or **Insufficient Evidence**. It shows the passage with chapter, section and page, flags everything except Supported for instructor review, and writes two audit logs. It never rewrites or corrects the text.

## What is and is not verified in this build

| Part | Status |
|---|---|
| Passage segmentation, retrieval fusion, decision rule (Eq. 3.11), claim extraction, two audit logs, CLI, Streamlit page | Written and covered by 67 automated tests, all run on the **stub** backend |
| `real` backend (all-MiniLM-L6-v2 and cross-encoder/nli-deberta-v3-base) | Written, **not run** where this was built, because the workspace could not reach Hugging Face. Run the first-use check below before trusting it |
| `alpha` (fusion weight) and the two NLI thresholds | **Placeholders** (0.5, 0.80, 0.80). The page warns until `"tuned": true` is set after validation selects them |
| Labels from the stub backend | Wiring checks only. Never report them |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm                # or en_core_web_md, see Limitations
```

Run every command from this folder, with `python -m verifier`.

### Try it without the textbook or any model download (stub)

```bash
python -m verifier build-index --config config.stub.json --textbook samples/mini_textbook.txt
VERIFIER_CONFIG=config.stub.json streamlit run app.py      # Windows PowerShell: $env:VERIFIER_CONFIG="config.stub.json"
python -m verifier verify --config config.stub.json --file samples/sample_ai_text.txt
```

The page shows a red STUB MODE banner for as long as this configuration is in use.

### Real run

1. Put your licensed copy of *Database System Concepts*, 7th edition, in a folder that is **not** tracked by git (the textbook, `data/` and `config.json` are already in `.gitignore`).
2. `cp config.example.json config.json`, then set `first_pdf_page` and `page_offset` so that printed page numbers match the book (printed page = PDF page + `page_offset`).
3. Build the index. It prints the number of passages per chapter and the share inside 100 to 150 words. Compare the chapter table with the printed contents page before going further.
   ```bash
   python -m verifier build-index --config config.json --textbook path/to/textbook.pdf
   ```
4. Start the page: `streamlit run app.py`. The first run downloads both models (several hundred MB) from Hugging Face.

**First-use check for the real backend** (this could not be done where the POC was built). Before indexing the full book, try the real models on the small sample book. Copy `config.example.json` to `config.mini.json`, set `index_dir` to `data/index_mini` and `log_dir` to `data/logs_mini`, then:

```bash
python -m verifier build-index --config config.mini.json --textbook samples/mini_textbook.txt
python -m verifier verify --config config.mini.json --file samples/sample_ai_text.txt
```

Check that the labels are not all the same, that the sentence about isolation in the sample text retrieves the isolation passage at rank 1 or 2, and that per-claim latency is plausible. An index built with the stub encoder is refused by the real backend (and the reverse), with a message telling you to rebuild.

## Commands

| Command | Purpose |
|---|---|
| `build-index --textbook FILE [--out DIR]` | Segment the textbook (PDF, or .txt with form feeds between pages) and embed the passages |
| `verify --file FILE` or `--text "..."` (`--json` for machine output) | One verification run |
| `show-log [--run-id ID]` | List runs, or show one run with its evidence |
| `purge-logs --days N` | Delete runs older than N days from both logs |

## Configuration

All settings are in one JSON file (`Config` in `verifier/config.py`); unknown keys are rejected. The page does not let users change them, so every run is reproducible from the logged configuration.

| Key | Default | Note |
|---|---|---|
| `passage_min_words`, `passage_max_words` | 100, 150 | Passage length target |
| `top_k` | 5 | Passages per claim |
| `alpha` | 0.5 | **Placeholder.** F = alpha x BM25 + (1 - alpha) x dense, each min-max normalised per claim |
| `bm25_k1`, `bm25_b`, `bm25_epsilon` | 1.5, 0.75, 0.25 | `rank_bm25` 0.2.2 BM25Okapi defaults |
| `tau_entail`, `tau_contra` | 0.80, 0.80 | **Placeholders.** Eq. 3.11 thresholds |
| `dense_model`, `nli_model` | all-MiniLM-L6-v2, nli-deberta-v3-base | Used as published, without fine-tuning |
| `spacy_model` | en_core_web_sm | Claim extraction |
| `backend` | `real` | `stub` for wiring tests only |
| `tuned` | false | Set to true only after validation selects `alpha` and the thresholds |

## Decision rule (Eq. 3.11)

E and C are the highest entailment and contradiction probabilities over the five retrieved passages.
Supported if E >= tau_E and E > C. Contradicted if C >= tau_C and C > E. Otherwise Insufficient Evidence.
Insufficient Evidence means the textbook does not settle the claim; it does not mean the claim is false.

## Audit logs (Table 3.8)

Two SQLite files in `data/logs/`, kept separate on purpose and joined only by `verification_run_id` and `claim_id`:

- `content_examination.sqlite` holds the submitted text (once per run, in `runs`) and one row per claim in `claims`: extracted claim, predicted label, review flag, latency, pipeline version.
- `source_evidence.sqlite` holds one row per retrieved passage: textbook and edition, passage ID, chapter, section, pages, rank, BM25, dense and fused scores, the three NLI probabilities, evidence status and reason. It stores **no textbook text**; the passage text is looked up from the index by passage ID.

The submitted text is stored in the log, so treat `data/` as confidential and set a retention period with `purge-logs`. Files are created with owner-only permissions where the operating system supports it.

## Reuse in the experiments

`VerificationPipeline`, `Retriever`, `decide` and `ClaimExtractor` are importable, so the benchmark scripts can run the same code path as the page. `verify` records per-claim latency (retrieval, NLI and decision; extraction time is reported separately per run). Record the hardware, Python and library versions next to any latency you report; the configuration hash is already in `pipeline_version`.

## Limitations

- **Claim extraction uses a small English parser and rules.** It splits only clear "and" / "but" coordination and semicolons, and it keeps negation intact. It will miss some compound claims and mis-tag some domain words (for example it mislabels "refers" and "guarantees" as nouns in a few test sentences). Those claims are kept and carry a visible parser warning rather than being dropped. `en_core_web_md` handled the foreign-key test sentence that `en_core_web_sm` mis-parsed; compare models on the annotated claims in Experiment A before choosing one.
- **Pronouns.** A pronoun in the same sentence is resolved to that sentence's subject. A pronoun that starts a sentence is resolved to the previous sentence's subject, which can be wrong ("A relation has a primary key. It must be unique." becomes a claim about the relation), so those claims carry a confirm-the-reference warning. Pronouns that cannot be resolved are left as written and warned.
- **One textbook, one edition.** A claim that is true in general but not in this book is Insufficient Evidence.
- **Retrieval is the ceiling.** If the right passage is not in the top five, the label cannot be Supported or Contradicted. Measure retrieval recall separately (Experiment A).
- **No rewriting, no correction.** The tool reports and flags only.

## Tests

```bash
python -m pytest -q
```

Covers the Eq. 3.11 truth table, passage segmentation, retrieval fusion and tie-breaking, claim extraction cases, log separation (no passage text and no submitted text in the evidence log), the run-to-run join, an end-to-end CLI run, and the Streamlit page driven headlessly.
