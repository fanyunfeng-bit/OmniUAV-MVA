"""`python -m mva ...` entry point. Also exposed as the `mva` console script
via [project.scripts] in pyproject.toml.

Subcommands (M2.8 mainline):
    ingest     🆕 Unified L0 → Segmenter → detect + embed → DuckDB + ChromaDB
    query      Mode A REPL (NL ↔ Qwen2.5-VL ↔ tools)
    ask        Single-shot NL question, with optional --image / --video
    eval       Batch QA accuracy on benchmark datasets (MVU-Eval)
    ui         🆕 M5.4 Gradio main page (NL chat + segment playback)

Legacy (frozen at M2.7, kept for back-reference; new work uses `ingest`):
    perceive   L0+L1+L2 → DuckDB (old per-view detections + cross-view links)
    index      Qwen3-VL-Embedding → ChromaDB (old per-segment-only path)
"""
from __future__ import annotations

import argparse
import sys

from mva.cli import ask, eval as eval_mod, index, ingest, perceive, query, ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mva",
        description="Multi-Video-Analysis: multi-drone multi-view + 7B VLM "
                    "situational understanding engine. "
                    "See PLAN.md / TECHNIC_REPORT.md for design.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    # M2.8 mainline
    ingest.add_subparser(sub)
    query.add_subparser(sub)
    ask.add_subparser(sub)
    eval_mod.add_subparser(sub)
    ui.add_subparser(sub)
    # M2.7 legacy (frozen)
    perceive.add_subparser(sub)
    index.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
