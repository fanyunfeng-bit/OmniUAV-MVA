"""`mva ask` — single-shot Mode A with optional attachments.

Same QueryService as `mva query`, but consumes a single question +
optional --image / --video flags and exits. Useful for scripting and as
the reference shape for the future HTTP / UI route.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mva.cli._common import (
    add_cross_view_arg,
    add_embedder_args,
    add_llm_args,
    add_store_args,
)
from mva.cli.query import (
    QueryService,
    _check_db_populated,
    _print_result,
    _resolve_quantize,
    _warn_hidden_cross_view,
)
from mva.contracts import Attachment, RichQuery


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "ask",
        help="Single-shot NL question with optional attachments",
    )
    p.add_argument("question", help="Natural-language question text")
    p.add_argument("--image", action="append", default=[], type=Path,
                   help="Path to an image attachment (repeatable)")
    p.add_argument("--video", action="append", default=[], type=Path,
                   help="Path to a video attachment (repeatable)")
    add_store_args(p, db_required=True)
    add_llm_args(p, llm_required=True)
    add_embedder_args(p)
    add_cross_view_arg(p)
    p.set_defaults(func=cmd_ask)


def cmd_ask(args: argparse.Namespace) -> int:
    attachments: list[Attachment] = []
    for path in args.image:
        if not path.is_file():
            print(f"[fatal] image not found: {path}")
            return 1
        attachments.append(Attachment(kind="image", path=path, label=path.name))
    for path in args.video:
        if not path.is_file():
            print(f"[fatal] video not found: {path}")
            return 1
        attachments.append(Attachment(kind="video", path=path, label=path.name))

    if not _check_db_populated(args.db_path):
        return 1
    _warn_hidden_cross_view(args.db_path, args.cross_view)

    effective_quantize = _resolve_quantize(args)

    rich = RichQuery(text=args.question, attachments=attachments)

    with QueryService(
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        llm_model=args.llm,
        embedder_model=args.embedder_model if args.chroma_dir else None,
        embed_dim=args.embed_dim,
        quantization=effective_quantize,
        enable_cross_view=(args.cross_view == "auto"),
    ) as service:
        _print_result(service.answer(rich))
    return 0
