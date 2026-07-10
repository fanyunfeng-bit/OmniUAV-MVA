"""`mva eval` — batch QA accuracy on benchmark datasets (currently MVU-Eval).

Iterates `adapter.load_qa_pairs()`, builds an MVU-Eval-style multi-video
prompt for each (per DATASETS/MVU-Eval/main_all_MVU_Eval_llama3.py
reference), runs `LLMClient.complete_messages`, parses the answer letter,
and accumulates per-task accuracy.

JSONL output (one line per question):
    {"qa_id": "...", "task": "...", "model_output": "...",
     "predicted": "A", "ground_truth": "A", "correct": true,
     "latency_ms": 12345}
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from mva.cli._common import add_dataset_args, add_llm_args, resolve_dataset
from mva.datasets import MVUEvalDataset
from mva.datasets.mvu_eval import (
    MVU_DEFAULT_MAX_PIXELS,
    MVU_DEFAULT_NFRAMES,
)
from mva.l4_llm import LLMClient


# Hoisted from cmd_eval so OOM-retry helper can also call empty_cache().
# If torch isn't installed, the helpers degrade to no-op.
try:
    import torch as _torch
    _HAS_CUDA = _torch.cuda.is_available()
except ImportError:
    _torch = None  # type: ignore[assignment]
    _HAS_CUDA = False


# M3.x: MVU-Eval's 8 task types — used by --per-task stratified sampling.
# Order is the same as the paper's reference script.
MVU_TASK_TYPES = (
    "Counting", "KIR", "TR", "OR", "SU", "ICL", "Comparison", "RAG",
)


# P3-16: multi-video QAs (≥ this many videos) drop nframes to keep peak
# activations under the 24 GB GPU budget. Defaults calibrated from
# M5.2-pilot OOM autopsy (TR task: 6-8 videos × 32 frames → 22 GiB).
MULTI_VIDEO_THRESHOLD_DEFAULT = 4
MULTI_VIDEO_NFRAMES_DEFAULT = 16


_LETTER_RE = re.compile(r"\b([A-Z])\b")


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "eval",
        help="Run a benchmark dataset's QA pairs and compute accuracy "
             "(currently only MVU-Eval)",
    )
    add_dataset_args(p, scene_required=False)
    add_llm_args(p, llm_required=True)
    p.add_argument("--max-questions", type=int, default=10,
                   help="Cap number of QA pairs (default 10; 0 = all). "
                        "Mutually exclusive with --per-task.")
    p.add_argument("--per-task", type=int, default=None,
                   help="M5.2-pilot stratified sampling: take N QA pairs from "
                        "EACH MVU task type (Counting / KIR / TR / OR / SU / "
                        "ICL / Comparison / RAG). Total = N * 8. Overrides "
                        "--max-questions when set.")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task filter "
                        "(e.g. 'Counting,Spatial Understanding')")
    p.add_argument("--per-question-timeout", type=int, default=300,
                   help="M3.9 (PROBLEMS P2-09): per-question wall-clock "
                        "timeout in seconds. On timeout the model output "
                        "becomes '[TIMEOUT]' and eval continues. Default 300 "
                        "(Qwen2.5-VL on 8-min videos can take ~3-5 min so 5 "
                        "min is comfortably generous; lower it for shorter "
                        "videos)")
    p.add_argument("--nframes", type=int, default=MVU_DEFAULT_NFRAMES,
                   help=f"Frames per video sent to the VLM "
                        f"(default {MVU_DEFAULT_NFRAMES}, MVU-Eval reference)")
    p.add_argument("--max-pixels", type=int, default=MVU_DEFAULT_MAX_PIXELS,
                   help=f"Max pixel dim per frame "
                        f"(default {MVU_DEFAULT_MAX_PIXELS}, MVU-Eval reference)")
    p.add_argument("--multi-video-threshold", type=int,
                   default=MULTI_VIDEO_THRESHOLD_DEFAULT,
                   help=f"M5.2-prep (PROBLEMS P3-16): when a question references "
                        f">= this many videos, drop --nframes to "
                        f"--multi-video-nframes to keep peak activations under "
                        f"the 24 GB budget (default {MULTI_VIDEO_THRESHOLD_DEFAULT}, "
                        f"calibrated from TR task OOM autopsy)")
    p.add_argument("--multi-video-nframes", type=int,
                   default=MULTI_VIDEO_NFRAMES_DEFAULT,
                   help=f"M5.2-prep (PROBLEMS P3-16): reduced nframes for "
                        f"multi-video QAs at/above --multi-video-threshold "
                        f"(default {MULTI_VIDEO_NFRAMES_DEFAULT}). Capped at "
                        f"--nframes so an explicit low --nframes is never "
                        f"overridden upward.")
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL output file (one record per question)")
    p.set_defaults(func=cmd_eval)


def cmd_eval(args: argparse.Namespace) -> int:
    adapter = resolve_dataset(args)
    if not adapter.supports_qa_eval:
        print(f"[fatal] dataset {adapter.name!r} does not support QA eval")
        return 1
    if not isinstance(adapter, MVUEvalDataset):
        # Generic QA pipeline could come later; current prompt format is
        # specifically tuned for MVU-Eval per its reference script.
        print(f"[warn] mva eval prompt format is MVU-Eval-specific; "
              f"adapter {adapter.name!r} may produce wrong results")

    # M3.9 (PROBLEMS P2-10): fail fast on decord missing rather than 5
    # layers deep into qwen-vl-utils.process_vision_info.
    import importlib.util
    if importlib.util.find_spec("decord") is None:
        print("[fatal] decord required for MVU-Eval video reading. "
              "Install with: pip install decord  (or `pip install -e .[llm]`)")
        return 1

    # Build the QA iterable. --per-task wins over --max-questions when set.
    if args.per_task is not None and args.per_task > 0:
        if args.tasks:
            print("[fatal] --per-task and --tasks are mutually exclusive")
            return 1
        qa_iter = _stratified_qa(adapter, args.per_task)
        print(f"[eval] stratified: {args.per_task} per task × "
              f"{len(MVU_TASK_TYPES)} tasks = {args.per_task * len(MVU_TASK_TYPES)} total")
    else:
        tasks_filter = (
            [t.strip() for t in args.tasks.split(",")] if args.tasks else None
        )
        limit = args.max_questions if args.max_questions > 0 else None
        qa_iter = adapter.load_qa_pairs(tasks=tasks_filter, limit=limit)

    # P3-16 startup hint: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # reduces fragmentation OOM for variable-shape inputs like multi-video
    # QA. Has to be set BEFORE the process starts (torch reads it at
    # import) — we can only nudge the user, not set it ourselves.
    if _HAS_CUDA and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        print("[eval] HINT (P3-16): for multi-video runs, launch with "
              "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce "
              "fragmentation OOM. Example:\n"
              "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True mva eval ...")

    print(f"[L4] Loading {args.llm} (quantization={args.quantize})")
    llm = LLMClient(model_path=args.llm, quantization=args.quantize)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[eval] output → {args.output}")
    print(f"[eval] per-question timeout: {args.per_question_timeout}s")
    print(f"[eval] multi-video heuristic: >= {args.multi_video_threshold} "
          f"videos → nframes={min(args.nframes, args.multi_video_nframes)} "
          f"(base nframes={args.nframes})")

    n_correct = 0
    n_total = 0
    per_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    # M3.9 (PROBLEMS P3-16): per-question torch.cuda.empty_cache() to
    # release KV cache + decoded video tensors. M5.2-pilot caught a 23%
    # OOM rate on TR (Temporal Reasoning) which sends 6-8 videos × 32
    # frames each; without empty_cache between questions, KV cache
    # accumulates until OOM hits on the next big input.
    with args.output.open("w", encoding="utf-8") as out_f:
        for qa in qa_iter:
            if _HAS_CUDA:
                _torch.cuda.empty_cache()
            effective_nframes = _resolve_nframes(
                qa,
                base_nframes=args.nframes,
                threshold=args.multi_video_threshold,
                reduced=args.multi_video_nframes,
            )
            t0 = time.time()
            try:
                model_output = _run_qa_with_oom_retry(
                    adapter, llm, qa, effective_nframes, args.max_pixels,
                    timeout_sec=args.per_question_timeout,
                )
            except Exception as exc:
                model_output = f"[ERROR] {type(exc).__name__}: {exc}"
            latency_ms = int((time.time() - t0) * 1000)

            predicted = _parse_letter(model_output) or "?"
            correct = (
                qa.ground_truth is not None
                and predicted == qa.ground_truth.upper()
            )
            n_total += 1
            n_correct += int(correct)
            task = qa.task or "unknown"
            per_task[task]["total"] += 1
            per_task[task]["correct"] += int(correct)

            record = {
                "qa_id": qa.qa_id,
                "task": task,
                "question": qa.question,
                "options": qa.options,
                "ground_truth": qa.ground_truth,
                "predicted": predicted,
                "model_output": model_output,
                "correct": correct,
                "latency_ms": latency_ms,
                "nframes_used": effective_nframes,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            mark = "✓" if correct else "✗"
            print(f"  qa-{qa.qa_id} [{task}] {mark}  pred={predicted}  "
                  f"gt={qa.ground_truth}  ({latency_ms} ms, "
                  f"nframes={effective_nframes})")

    print("\n========== eval summary ==========")
    for task in sorted(per_task):
        d = per_task[task]
        pct = 100.0 * d["correct"] / max(1, d["total"])
        print(f"  {task}:  {d['correct']}/{d['total']}  ({pct:.1f}%)")
    overall_pct = 100.0 * n_correct / max(1, n_total)
    print(f"\nOverall: {n_correct}/{n_total} ({overall_pct:.1f}%)")
    print(f"[eval] wrote {n_total} records to {args.output}")
    llm.unload()
    return 0


def _stratified_qa(adapter: MVUEvalDataset, per_task: int):
    """M5.2-pilot: yield `per_task` questions from each of the 8 MVU
    task types in turn. Total = per_task * 8. Quietly skips task types
    with fewer than `per_task` questions (e.g. OR has 126 but Counting
    has 227)."""
    for task in MVU_TASK_TYPES:
        n = 0
        for qa in adapter.load_qa_pairs(tasks=[task], limit=per_task):
            yield qa
            n += 1
        if n == 0:
            print(f"[eval] WARN: task {task!r} matched 0 QA — skipping")


def _run_qa_with_timeout(
    adapter, llm: LLMClient, qa, nframes: int, max_pixels: int,
    *, timeout_sec: int,
) -> str:
    """M3.9 (PROBLEMS P2-09): wrap `_run_qa` in a thread + future so a
    single hung video decode (e.g. corrupt mp4) doesn't sink the whole
    eval run.

    We use a single-thread executor per call rather than a global pool
    because each call holds at most one in-flight future. Timeout
    returns sentinel '[TIMEOUT]' so the parser doesn't crash; the eval
    loop continues with the next question."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run_qa, adapter, llm, qa, nframes, max_pixels)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            future.cancel()   # best-effort; CUDA work may continue briefly
            print(f"  ⏱ TIMEOUT qa-{qa.qa_id} after {timeout_sec}s — skipping")
            return "[TIMEOUT]"


def _resolve_nframes(
    qa, *, base_nframes: int, threshold: int, reduced: int,
) -> int:
    """P3-16: pick effective nframes for this QA.

    Multi-video QAs (>= `threshold` videos) drop to `reduced` frames to
    keep peak activations under the 24 GB GPU budget. Single-video QAs
    keep the full `base_nframes`. The reduced value is capped at
    `base_nframes` so an explicit low --nframes (e.g. for a tiny GPU)
    is never increased.
    """
    video_paths = (qa.metadata or {}).get("video_paths") if hasattr(qa, "metadata") else None
    n_videos = len(video_paths) if video_paths else 0
    if n_videos >= threshold:
        return min(base_nframes, reduced)
    return base_nframes


def _is_oom(exc: BaseException) -> bool:
    """Detect CUDA OOM across torch versions.

    Modern torch (>=1.13) raises `torch.cuda.OutOfMemoryError`. Older
    versions raise plain RuntimeError with "out of memory" in the
    message. We match by class name (so we don't need a torch import
    to recognize it) AND by message substring."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _run_qa_with_oom_retry(
    adapter, llm: LLMClient, qa, nframes: int, max_pixels: int,
    *, timeout_sec: int,
) -> str:
    """P3-16: wrap `_run_qa_with_timeout` with one OOM retry at halved
    nframes.

    On CUDA OOM the wrapper empty_caches, halves nframes (floor 1), and
    retries once. A second OOM returns an error sentinel so the eval
    loop can continue with the next question. Non-OOM exceptions
    propagate to cmd_eval's general handler."""
    try:
        return _run_qa_with_timeout(
            adapter, llm, qa, nframes, max_pixels, timeout_sec=timeout_sec,
        )
    except Exception as exc:
        if not _is_oom(exc):
            raise
        retry_nframes = max(1, nframes // 2)
        print(f"  ⚠ OOM qa-{qa.qa_id} with nframes={nframes} — "
              f"retrying with nframes={retry_nframes}")
        if _HAS_CUDA:
            _torch.cuda.empty_cache()
        try:
            return _run_qa_with_timeout(
                adapter, llm, qa, retry_nframes, max_pixels,
                timeout_sec=timeout_sec,
            )
        except Exception as exc2:
            return (
                f"[ERROR] OOM retry failed (nframes "
                f"{nframes}→{retry_nframes}): "
                f"{type(exc2).__name__}: {exc2}"
            )


def _run_qa(adapter, llm: LLMClient, qa, nframes: int, max_pixels: int) -> str:
    """Build MVU-Eval-format messages + invoke complete_messages."""
    video_paths = qa.metadata.get("video_paths", [])
    if not video_paths:
        return "[ERROR] qa.metadata.video_paths is empty"

    content: list[dict[str, Any]] = []
    for idx, fname in enumerate(video_paths):
        try:
            resolved = adapter._resolve_video(fname)
        except FileNotFoundError as exc:
            return f"[ERROR] {exc}"
        content.append({"type": "text", "text": f"The following is the Video {idx+1}"})
        content.append({
            "type": "video",
            "video": str(resolved),
            "max_pixels": max_pixels * max_pixels,
            "nframes": nframes,
        })
    content.append({"type": "text", "text": qa.question})
    if qa.options:
        content.append({"type": "text", "text": "\n".join(qa.options)})
    content.append({"type": "text",
                    "text": "Please select the correct answer from the options. "
                            "Answer with the option's letter directly."})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    return llm.complete_messages(messages, max_new_tokens=128)


def _parse_letter(text: str) -> Optional[str]:
    """Extract first standalone A-Z letter from the model output."""
    if not text:
        return None
    m = _LETTER_RE.search(text)
    return m.group(1) if m else None
