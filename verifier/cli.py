"""Command line interface: build the index, verify text, inspect and purge logs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audit import AuditLogs
from .config import Config
from .retrieve import make_encoder
from .textbook import build_manifest, build_passages, chapter_summary, length_stats, save_passages


def _load_config(args) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    overrides = {}
    for name in ("backend", "textbook_id", "edition", "first_pdf_page", "page_offset"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return replace(cfg, **overrides) if overrides else cfg


def cmd_build_index(args) -> int:
    cfg = _load_config(args)
    out = Path(args.out or cfg.index_dir)
    out.mkdir(parents=True, exist_ok=True)

    passages, info = build_passages(args.textbook, cfg)
    encoder = make_encoder(cfg)
    embeddings = encoder.encode([p.text for p in passages])

    save_passages(passages, out / "passages.jsonl")
    np.save(out / "embeddings.npy", embeddings)
    manifest = build_manifest(args.textbook, cfg, passages, info, encoder.name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = length_stats(passages, cfg)
    print(f"Index written to {out}")
    print(f"  labelling mode: {info['labeling_mode']}  pages read: {info['pages_read']}  "
          f"dropped fragments: {info['dropped_fragments']}")
    print(f"  passages: {stats['passages']}  words min/median/max: "
          f"{stats['min_words']}/{stats['median_words']}/{stats['max_words']}  "
          f"within {cfg.passage_min_words}-{cfg.passage_max_words} words: {stats['share_within_target']:.0%}")
    print("\nPassages per chapter (check these against the printed book):")
    for row in chapter_summary(passages):
        print(f"  {row['passages']:>5}  pp. {row['first_page']}-{row['last_page']}  {row['chapter'] or '(no chapter)'}")
    return 0


def cmd_verify(args) -> int:
    from .pipeline import VerificationPipeline

    cfg = _load_config(args)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        print("give --file or --text", file=sys.stderr)
        return 2
    pipeline = VerificationPipeline.from_config(cfg)
    result = pipeline.verify(text)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if cfg.backend == "stub":
        print("*** STUB BACKEND: labels check the wiring only and are not results ***\n")
    elif not cfg.tuned:
        print("*** Default alpha and thresholds are placeholders; replace them with validated values ***\n")
    print(f"Run {result.run_id}  ({result.pipeline_version})")
    for claim in result.claims:
        flag = "  [REVIEW]" if claim.review_flag else ""
        print(f"\n{claim.claim_id}  {claim.label}{flag}  ({claim.latency_ms:.0f} ms)")
        print(f"  {claim.claim}")
        for warning in claim.warnings:
            print(f"  ! {warning}")
        for item in claim.evidence[:2]:
            print(f"  - {item.passage_id} [{item.locator}] E={item.entailment:.2f} C={item.contradiction:.2f}")
    print(f"\nCounts: {result.counts()}")
    return 0


def cmd_show_log(args) -> int:
    logs = AuditLogs(_load_config(args).log_dir)
    if not args.run_id:
        for row in logs.list_runs(args.limit):
            print(f"{row['verification_run_id']}  {row['timestamp']}  {row['n_claims']} claims  {row['pipeline_version']}")
        return 0
    print(json.dumps(logs.joined(args.run_id), indent=2, ensure_ascii=False))
    return 0


def cmd_purge_logs(args) -> int:
    logs = AuditLogs(_load_config(args).log_dir)
    print(f"removed {logs.purge_older_than(args.days)} run(s) older than {args.days} days")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verifier", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="JSON file with configuration overrides")
        p.add_argument("--backend", choices=["real", "stub"], help="override the backend")

    p = sub.add_parser("build-index", help="segment the textbook and build the retrieval index")
    common(p)
    p.add_argument("--textbook", required=True, help="textbook .pdf, or .txt with form feeds between pages")
    p.add_argument("--out", help="index directory (default from config)")
    p.add_argument("--textbook-id", dest="textbook_id")
    p.add_argument("--edition")
    p.add_argument("--first-pdf-page", dest="first_pdf_page", type=int)
    p.add_argument("--page-offset", dest="page_offset", type=int)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("verify", help="verify the claims in a text")
    common(p)
    p.add_argument("--file")
    p.add_argument("--text")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("show-log", help="list runs, or show one run joined across both logs")
    common(p)
    p.add_argument("--run-id")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_show_log)

    p = sub.add_parser("purge-logs", help="delete runs older than N days from both logs")
    common(p)
    p.add_argument("--days", type=int, required=True)
    p.set_defaults(func=cmd_purge_logs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
