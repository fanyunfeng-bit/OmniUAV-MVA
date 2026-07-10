"""Shared argparse helpers for all `mva ...` subcommands."""
from __future__ import annotations

import argparse
from pathlib import Path

from mva.datasets import list_known


def add_dataset_args(p: argparse.ArgumentParser, scene_required: bool = False) -> None:
    """Standard --dataset / --dataset-root / --scene block."""
    p.add_argument(
        "--dataset", required=True,
        choices=list_known(),
        help="Dataset name (registered in mva.datasets.registry)",
    )
    p.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Override the dataset's default root directory",
    )
    p.add_argument(
        "--scene", required=scene_required, default=None,
        help="Scene id within the dataset (e.g. 'MATRIX_30x30' or 'qa-0'). "
             "Required for perceive/index; not required for eval (iterates QAs).",
    )


def add_store_args(p: argparse.ArgumentParser, db_required: bool = True) -> None:
    """Standard --db-path / --chroma-dir block."""
    p.add_argument(
        "--db-path", required=db_required,
        help="DuckDB path (WorldStateStore). ':memory:' = in-process only.",
    )
    p.add_argument(
        "--chroma-dir", default=None,
        help="ChromaDB persist directory (multimodal embedding index).",
    )


def add_llm_args(p: argparse.ArgumentParser, llm_required: bool = True) -> None:
    """Standard --llm / --quantize block."""
    p.add_argument(
        "--llm", required=llm_required,
        help="Generative LLM model id (e.g. Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    p.add_argument(
        "--quantize", choices=["int4", "int8"], default=None,
        help="Quantize the gen LLM via bitsandbytes. int4 (~5GB) lets it "
             "coexist with Qwen3-VL-Embedding-8B on a 24GB GPU. When unset, "
             "query/ask auto-picks INT4 if the GPU is ≤30GB AND --chroma-dir "
             "is on (embedder needs to coexist); >30GB GPUs keep FP16.",
    )
    p.add_argument(
        "--no-auto-quantize", dest="no_auto_quantize",
        action="store_true", default=False,
        help="Disable the chroma+VRAM auto-quantize heuristic. With this, "
             "--quantize is the sole truth — unset means FP16 even if "
             "embedder also needs to load.",
    )


def add_embedder_args(p: argparse.ArgumentParser) -> None:
    """Standard --embedder-model / --embed-dim block."""
    from mva.l5_state import DEFAULT_DIM, DEFAULT_MODEL
    p.add_argument(
        "--embedder-model", default=DEFAULT_MODEL,
        help=f"Embedder model id (default {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--embed-dim", type=int, default=DEFAULT_DIM,
        help=f"MRL output dim, 64-4096 (default {DEFAULT_DIM} = sentrysearch)",
    )


def add_cross_view_arg(p: argparse.ArgumentParser) -> None:
    """Standard --cross-view {auto,off} flag for ask/query/eval.

    auto = expose cross-view tools (get_cross_view_links, count_cross_view_links,
           find_across_views) to the LLM.
    off  = hide those tools; the DuckDB rows are NOT deleted, only made invisible
           to the planner. Used for ablation runs.
    """
    p.add_argument(
        "--cross-view", choices=["auto", "off"], default="auto",
        help="auto (default): expose cross-view tools to the LLM. "
             "off: hide get_cross_view_links / count_cross_view_links / "
             "find_across_views (ablation; DB rows untouched).",
    )


def resolve_dataset(args) -> object:
    """Construct adapter from --dataset / --dataset-root."""
    from mva.datasets import get_adapter
    return get_adapter(args.dataset, root=args.dataset_root)
