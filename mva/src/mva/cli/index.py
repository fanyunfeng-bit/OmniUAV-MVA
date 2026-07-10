"""`mva index` — encode IndexUnits with Qwen3-VL-Embedding → ChromaDB.

Dataset-agnostic: drives `DatasetAdapter.iter_indexable_units(scene_id)`
and pushes each unit through the embedder + into the vector store.

MATRIX yields one ReID-typed unit per ROI crop (requires a populated
WorldStateStore from a prior `mva perceive`). MVU-Eval yields one
frame-typed unit per **video segment** (sentrysearch-style, default
~10 s sliding window). Default `--max-units 100` to prevent accidental
9-hour eval-set encodes.

Optional `--detect-per-segment` runs YOLO over `--detect-sample-k` frames
of every image_seq unit and stuffs `detected_classes` / `detected_counts`
into ChromaDB metadata, so the LLM can ground answers like "this segment
contains 3 persons + 1 car" without re-running detection at query time.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Optional

from mva.cli._common import (
    add_dataset_args,
    add_embedder_args,
    add_store_args,
    resolve_dataset,
)
from mva.datasets import MVUEvalDataset
from mva.datasets.mvu_eval import (
    DEFAULT_NFRAMES_PER_SEGMENT,
    DEFAULT_STRIDE_SEC,
    DEFAULT_WINDOW_SEC,
)
from mva.l5_state import MultimodalEmbedder, VectorStore, WorldStateStore


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "index",
        help="Encode IndexUnits via Qwen3-VL-Embedding → ChromaDB",
    )
    add_dataset_args(p, scene_required=True)
    add_store_args(p, db_required=False)         # db only needed for some adapters (MATRIX)
    add_embedder_args(p)
    p.add_argument("--views", nargs="+", default=None,
                   help="View ids to index (default: all in the scene)")
    p.add_argument("--max-units", type=int, default=100,
                   help="Cap units indexed across all views. Default 100 — "
                        "raise explicitly for large runs (MVU-Eval has "
                        "thousands of videos × multiple segments each).")
    p.add_argument("--device", default=None,
                   help="Torch device for the embedder (default auto)")

    # ---- video-slicing knobs (MVU-Eval-style adapters only) -------------
    p.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                   help=f"Sliding-window length in seconds for video adapters "
                        f"(default {DEFAULT_WINDOW_SEC}). Ignored for MATRIX.")
    p.add_argument("--stride-sec", type=float, default=DEFAULT_STRIDE_SEC,
                   help=f"Stride between segments in seconds (default "
                        f"{DEFAULT_STRIDE_SEC}; equal to --window-sec = "
                        f"non-overlapping).")
    p.add_argument("--nframes-per-segment", type=int,
                   default=DEFAULT_NFRAMES_PER_SEGMENT,
                   help=f"Frames sampled per segment, mean-pooled into one "
                        f"embedding (default {DEFAULT_NFRAMES_PER_SEGMENT}).")

    # ---- lightweight in-line detection ----------------------------------
    p.add_argument("--detect-per-segment", action="store_true",
                   help="Run YOLO on each image_seq unit's frames; store "
                        "aggregated class counts in ChromaDB metadata "
                        "(detected_classes + detected_counts).")
    p.add_argument("--detect-model", default="yolo11n.pt",
                   help="ultralytics model id (default yolo11n.pt). Only "
                        "loaded when --detect-per-segment is set.")
    p.add_argument("--detect-conf", type=float, default=0.25,
                   help="Detection confidence threshold (default 0.25).")
    p.add_argument("--detect-device", default=None,
                   help="Torch device for the detector "
                        "(default = embedder device or auto).")
    p.set_defaults(func=cmd_index)


def cmd_index(args: argparse.Namespace) -> int:
    if not args.chroma_dir:
        print("[fatal] --chroma-dir is required for index")
        return 1

    adapter = resolve_dataset(args)
    scene = adapter.get_scene(args.scene)
    view_ids = args.views or scene.view_ids
    print(f"[index] dataset={adapter.name} scene={args.scene} "
          f"views={view_ids}  max_units={args.max_units}")
    if isinstance(adapter, MVUEvalDataset):
        print(f"[index] segmenting: window={args.window_sec}s "
              f"stride={args.stride_sec}s "
              f"nframes/segment={args.nframes_per_segment}")
    if args.detect_per_segment:
        print(f"[index] lightweight detection ON: model={args.detect_model} "
              f"conf={args.detect_conf}")

    store: Optional[WorldStateStore] = None
    if args.db_path:
        print(f"[L5] WorldStateStore(db_path={args.db_path})")
        store = WorldStateStore(db_path=args.db_path)

    print(f"[L4] Loading {args.embedder_model} (dim={args.embed_dim}) ...")
    embedder = MultimodalEmbedder(
        model_path=args.embedder_model, dim=args.embed_dim, device=args.device,
    )
    embedder._ensure_loaded()

    detector = None
    if args.detect_per_segment:
        from mva.l1_perception import Detector
        det_device = args.detect_device or args.device
        detector = Detector(
            model_name=args.detect_model,
            conf=args.detect_conf,
            device=det_device,
        )

    print(f"[L5] VectorStore(persist_dir={args.chroma_dir})")
    vstore = VectorStore(
        persist_dir=args.chroma_dir,
        embedding_function=embedder.as_chromadb_embedding_function(),
    )
    initial_count = vstore.collection.count()

    total = 0
    for vid in view_ids:
        if total >= args.max_units:
            break
        try:
            units = _iter_units(adapter, args, vid, store, total)
        except ValueError as exc:
            print(f"[fatal] {exc}")
            embedder.unload()
            if store:
                store.close()
            return 1
        for unit in units:
            if total >= args.max_units:
                break
            if unit.kind == "image":
                vec = embedder.encode_image(unit.data)
            elif unit.kind == "image_seq":
                vec = embedder.encode_images(unit.data)
            else:
                continue

            extra_meta = _build_metadata(unit, detector)

            vstore.add(
                vec, unit.vector_type, unit.scene_id + "::" + unit.view_id,
                unit.unit_id,
                extra_metadata=extra_meta,
                document=unit.document,
            )
            total += 1
            if total % 25 == 0:
                print(f"  encoded {total} units...")

    final_count = vstore.collection.count()
    print(f"\n[index] {total} units encoded this run; collection now has "
          f"{final_count} entries (was {initial_count})")
    print("[L4] Unloading embedder to free GPU memory")
    embedder.unload()
    if store:
        store.close()
    return 0


def _iter_units(adapter, args, view_id, store, already_done: int):
    """Call the adapter's `iter_indexable_units` with whichever kwargs it
    accepts. MVU-Eval needs the video-segment knobs; MATRIX doesn't."""
    common_kwargs: dict[str, Any] = {
        "view_id": view_id,
        "max_frames": args.max_units - already_done,
        "store": store,
    }
    if isinstance(adapter, MVUEvalDataset):
        common_kwargs.update({
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "nframes_per_segment": args.nframes_per_segment,
        })
    return adapter.iter_indexable_units(args.scene, **common_kwargs)


def _build_metadata(unit, detector) -> dict[str, Any]:
    """Merge segment timing + (optional) detection summary into the
    metadata that goes to ChromaDB. Stays primitive-only (str/int/float/
    bool) — ChromaDB doesn't accept list/dict values."""
    meta: dict[str, Any] = dict(unit.metadata or {})
    if unit.start_sec is not None:
        meta["start_sec"] = float(unit.start_sec)
    if unit.end_sec is not None:
        meta["end_sec"] = float(unit.end_sec)
    if unit.segment_idx is not None:
        meta["segment_idx"] = int(unit.segment_idx)
        # `chunk_id` is the ChromaDB id suffix knob — keeps per-segment ids
        # unique even when the rest of (view, tracklet, vector_type) collide.
        meta.setdefault("chunk_id", int(unit.segment_idx))

    if detector is not None and unit.kind == "image_seq":
        counts: Counter[str] = Counter()
        for frame in unit.data:
            for det in detector.detect(frame):
                counts[det.class_name] += 1
        if counts:
            # Per-segment count = max(per-frame count) is a reasonable proxy
            # for "how many of X are visible at once" given the segment.
            # We just store sum across sampled frames; downstream can divide
            # by nframes if it wants the average. Storing sum keeps the
            # metadata schema simple.
            classes_sorted = sorted(counts.keys())
            meta["detected_classes"] = ",".join(classes_sorted)
            meta["detected_counts_json"] = json.dumps(
                {k: counts[k] for k in classes_sorted}, ensure_ascii=False,
            )
    return meta
