"""`mva ingest` — M2.8 unified perception + indexing in one pass.

Replaces the M2.7 split of `mva perceive` (DuckDB) + `mva index` (ChromaDB).
A single dataset traversal produces both stores:

    L0 source ─→ Segmenter (10s windows × K frames) ─┬─→ segment embedding
                                                     │      ↓
                                                     │  ChromaDB (vector_kind="segment")
                                                     │      ↓
                                                     │  DuckDB segments(view_id, segment_idx, ...,
                                                     │                  embed_chroma_id)
                                                     │
                                                     ├─→ YOLO detect + ByteTrack (M3.1)
                                                     │   on each sampled frame, aggregating
                                                     │   per-track within a segment
                                                     │      ↓
                                                     │  DuckDB tracklets_<view> (one row per
                                                     │                            (segment, track))
                                                     │
                                                     └─→ (optional) bbox crop → ReID embedding,
                                                         mean-pooled across the track's frames
                                                            ↓
                                                        ChromaDB (vector_kind="bbox") and
                                                        embedding_ref column in tracklets_<view>

Defaults (per user decision, 2026-05-22):
  --detect             ON   (off via --no-detect)
  --embed-bboxes       ON   (off via --no-embed-bboxes; bbox encode is ~12 GB-hours on full MVU-Eval)
  --track              ON   (off via --no-track; M3.0 baseline / ablation)
  --segments-per-view  30   (per-view cap; ensures equal temporal coverage; pass 0 for no cap)

Legacy `mva perceive` and `mva index` stay frozen at M2.7 — they write
the old DB shape and don't know about segments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Optional

from mva.cli._common import (
    add_dataset_args,
    add_embedder_args,
    add_store_args,
    resolve_dataset,
)
from mva.contracts import ViewObservation
from mva.l1_perception import ByteTracker
from mva.l2_crossview import (
    AppearanceCrossViewLinker,
    GeometricCrossViewLinker,
)
from mva.l5_state import MultimodalEmbedder, VectorStore, WorldStateStore
from mva.segmentation import SegmenterConfig


# M3.0 default cross-view thresholds.
#
# Empirical observation 2026-05-22: Qwen3-VL-Embedding-8B's cosine
# between same-class crops across two video clips (MVU-Eval "add"
# editing variants of the same scene) tops out at ~0.56, mean ~0.50.
# The original "conservative" defaults (0.6 / 0.75) reject everything,
# producing 0 links across legitimately-overlapping content. We lower
# defaults to land just under the typical positive range and expose
# via CLI flag so users can sweep:
#
#   --appearance-threshold 0.6  → strict, few links, high precision
#   --appearance-threshold 0.45 → permissive, more recall, some FP
#
# Future fix (M5.5 already SKIP per user decision): swap in OSNet /
# TransReID for ReID-specific cosines. Until then, lowered defaults
# acknowledge Qwen-VL-Emb's general-purpose ceiling on cross-video
# object identity.
_GEOMETRIC_APPEARANCE_THRESHOLD = 0.5    # MATRIX synchronized secondary filter
_APPEARANCE_ONLY_THRESHOLD = 0.45        # MVU-Eval-style pure-cosine


# Sentinel value so users can pass `--segments-per-view 0` to disable the cap.
_NO_CAP_SENTINEL = 0


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "ingest",
        help="Unified ingest: L0 → Segmenter → detect + embed → DuckDB + ChromaDB",
    )
    add_dataset_args(p, scene_required=True)
    add_store_args(p, db_required=True)
    add_embedder_args(p)
    p.add_argument("--views", nargs="+", default=None,
                   help="View ids to ingest (default: all in the scene)")
    p.add_argument("--device", default=None,
                   help="Torch device for the embedder (default auto)")

    # ---- segmentation knobs --------------------------------------------
    p.add_argument("--window-sec", type=float, default=10.0,
                   help="Sliding-window length (default 10.0)")
    p.add_argument("--stride-sec", type=float, default=10.0,
                   help="Stride between segment starts (default 10.0 = "
                        "non-overlapping)")
    p.add_argument("--nframes-per-segment", type=int, default=4,
                   help="Frames sampled per segment, mean-pooled into one "
                        "segment embedding (default 4)")

    # ---- per-view cap -----------------------------------------------------
    p.add_argument("--segments-per-view", type=int, default=30,
                   help="Max segments ingested PER VIEW (default 30). "
                        "Ensures every view gets equal temporal coverage "
                        "regardless of source video length. Pass 0 for no cap.")

    # ---- detection (default ON) ----------------------------------------
    p.add_argument("--detect", dest="detect", action="store_true",
                   default=True, help="Run YOLO on each segment's sampled "
                                      "frames (default ON)")
    p.add_argument("--no-detect", dest="detect", action="store_false",
                   help="Skip detection (segment embedding only)")
    p.add_argument("--detect-model", default="yolo11n.pt",
                   help="ultralytics model id. yolo11n.pt (closed 80-class, "
                        "default), yoloe-11l-seg.pt (open-vocab+seg), "
                        "yolov8x-worldv2.pt (open-vocab)")
    p.add_argument("--detect-conf", type=float, default=0.25,
                   help="Detection confidence threshold (default 0.25)")
    p.add_argument("--detect-classes", default=None,
                   help="Comma-separated natural-language target classes, e.g. "
                        "'car,person,三轮车'. For YOLOE/YOLO-World these become "
                        "the open-vocab vocabulary; for closed YOLO they filter "
                        "the output by name. Default: model's native classes.")
    p.add_argument("--detect-imgsz", type=int, default=None,
                   help="Inference image size. Open-vocab models on small aerial "
                        "objects want 1280/1920 (default: ultralytics 640).")
    p.add_argument("--detect-device", default=None,
                   help="Torch device for YOLO (default = --device)")

    # ---- bbox embedding (default ON) -----------------------------------
    p.add_argument("--embed-bboxes", dest="embed_bboxes", action="store_true",
                   default=True,
                   help="Embed every detected bbox crop into ChromaDB as "
                        "vector_kind='bbox' (default ON; off via "
                        "--no-embed-bboxes)")
    p.add_argument("--no-embed-bboxes", dest="embed_bboxes",
                   action="store_false", help="Skip per-bbox embedding")

    # ---- M3.1 tracker (default ON) -------------------------------------
    p.add_argument("--track", dest="track", action="store_true",
                   default=True,
                   help="Run ByteTrack to assign track IDs across frames "
                        "within each segment (default ON). Tracklets are "
                        "aggregated per (segment, track): bboxes JSON has "
                        "K rows, bbox embedding is mean-pooled over K crops.")
    p.add_argument("--no-track", dest="track", action="store_false",
                   help="Skip tracking — every detection writes an "
                        "independent tracklet row (M3.0 baseline / "
                        "ablation).")
    p.add_argument("--track-iou", type=float, default=0.5,
                   help="IoU bar for the tracker to associate a bbox with "
                        "an existing track (default 0.5).")
    p.add_argument("--tracker", default="iou_greedy",
                   choices=["iou_greedy", "bytetrack", "botsort"],
                   help="Tracking algorithm. iou_greedy (default, deterministic, "
                        "best for sparse K<=4); bytetrack (boxmot, faster); "
                        "botsort (boxmot, CMC camera-motion-comp — better on "
                        "moving-drone footage, ~2.5x slower than bytetrack).")

    # ---- M3.0 cross-view linking threshold ------------------------------
    p.add_argument("--appearance-threshold", type=float, default=None,
                   help="Cosine threshold for cross-view appearance "
                        "matching. None (default) uses built-in: 0.45 "
                        "for 'appearance' mode (MVU-Eval), 0.5 for "
                        "'synchronized' mode secondary filter (MATRIX). "
                        "Higher = stricter, fewer links, higher precision. "
                        "Qwen3-VL-Emb empirically tops out near 0.55-0.6 "
                        "on cross-video same-object pairs; values > 0.7 "
                        "typically yield 0 links.")

    # ---- M4.1 ROI cache (opt-in, default OFF) --------------------------
    p.add_argument(
        "--cache-rois", dest="cache_rois", action="store_true", default=False,
        help="Write each tracked bbox crop as a JPEG to <chroma-dir>-rois/ "
             "and populate ViewObservation.roi_uri. Required for fast "
             "LLMCrossViewLinker fallback in M4 (cache-hit ~5-20ms vs "
             "delayed decode 50-200ms on mp4). Default OFF — ~10-50MB/scene "
             "extra disk + ~10ms/track ingest overhead when ON.")
    p.add_argument(
        "--rois-dir", default=None,
        help="Where to write cached ROIs when --cache-rois is ON. "
             "Default <chroma-dir>-rois (i.e. peer dir to --chroma-dir).")

    # ---- M4.3 LLM fallback (opt-in, default OFF) ------------------------
    p.add_argument(
        "--enable-llm-fallback", dest="enable_llm_fallback",
        action="store_true", default=False,
        help="Trigger LLMCrossViewLinker (Qwen2.5-VL) as a second opinion "
             "on low-confidence (<0.5) cross-view links. Throttled to 1 "
             "LLM call per view across the run. Requires --fallback-llm. "
             "Default OFF — adds ~1-2s per fallback pair on a 3090.")
    p.add_argument(
        "--fallback-llm", default=None,
        help="HuggingFace model id or local path for the cross-view LLM "
             "judge (e.g. 'Qwen/Qwen2.5-VL-7B-Instruct'). Required when "
             "--enable-llm-fallback is set. Loaded with INT4 quantization "
             "to coexist with the embedder on a 24GB GPU.")
    p.add_argument(
        "--fallback-confidence-threshold", type=float, default=0.5,
        help="Cross-view link confidence below this triggers LLM fallback "
             "(default 0.5). Has no effect unless --enable-llm-fallback "
             "is set.")

    p.set_defaults(func=cmd_ingest)


def cmd_ingest(args: argparse.Namespace) -> int:
    if not args.chroma_dir:
        print("[fatal] --chroma-dir is required for ingest")
        return 1

    adapter = resolve_dataset(args)
    scene = adapter.get_scene(args.scene)
    view_ids = args.views or scene.view_ids

    # Use adapter's recommended defaults when user didn't override
    meta = scene.metadata or {}
    window = args.window_sec if args.window_sec != 10.0 else meta.get("default_window_sec", 10.0)
    stride = args.stride_sec if args.stride_sec != 10.0 else meta.get("default_stride_sec", 10.0)
    config = SegmenterConfig(
        window_sec=window,
        stride_sec=stride,
        nframes_per_segment=args.nframes_per_segment,
    )

    cap_disabled = args.segments_per_view == _NO_CAP_SENTINEL
    per_view_cap = float("inf") if cap_disabled else args.segments_per_view
    cap_str = "no cap" if cap_disabled else f"≤ {args.segments_per_view}/view"

    print(f"[ingest] dataset={adapter.name} scene={args.scene} views={view_ids}")
    print(f"[ingest] segmenter: window={config.window_sec}s "
          f"stride={config.stride_sec}s K={config.nframes_per_segment} "
          f"({cap_str})")
    print(f"[ingest] detect={'ON' if args.detect else 'OFF'} "
          f"track={'ON' if args.track else 'OFF'} "
          f"embed_bboxes={'ON' if args.embed_bboxes else 'OFF'} "
          f"cache_rois={'ON' if args.cache_rois else 'OFF'}")

    # ---- M4.1: ROI cache directory -------------------------------------
    rois_dir: Optional[str] = None
    if args.cache_rois:
        from pathlib import Path
        rois_dir = args.rois_dir or f"{args.chroma_dir}-rois"
        Path(rois_dir).mkdir(parents=True, exist_ok=True)
        print(f"[L5] ROI cache dir: {rois_dir} (M4.1 — required for fast "
              f"LLMCrossViewLinker fallback)")
    cv_mode = getattr(adapter, "cross_view_linking_mode", "none")
    if cv_mode != "none":
        effective = (args.appearance_threshold
                     if args.appearance_threshold is not None
                     else (_APPEARANCE_ONLY_THRESHOLD if cv_mode == "appearance"
                           else _GEOMETRIC_APPEARANCE_THRESHOLD))
        print(f"[ingest] cross-view mode={cv_mode} "
              f"appearance_threshold={effective}")
    if args.detect and args.embed_bboxes and cap_disabled:
        print("[ingest] WARNING: bbox embedding without --segments-per-view cap "
              "can take hours on large datasets")

    # ---- L5 stores -----------------------------------------------------
    print(f"[L5] WorldStateStore(db_path={args.db_path})")
    store = WorldStateStore(db_path=args.db_path)

    print(f"[L4] Loading embedder {args.embedder_model} (dim={args.embed_dim}) ...")
    embedder = MultimodalEmbedder(
        model_path=args.embedder_model, dim=args.embed_dim, device=args.device,
    )
    embedder._ensure_loaded()

    print(f"[L5] VectorStore(persist_dir={args.chroma_dir})")
    vstore = VectorStore(
        persist_dir=args.chroma_dir,
        embedding_function=embedder.as_chromadb_embedding_function(),
    )
    initial_chroma = vstore.collection.count()

    detector = None
    if args.detect:
        from mva.l1_perception import Detector
        det_device = args.detect_device or args.device
        detect_classes = (
            [c.strip() for c in args.detect_classes.split(",") if c.strip()]
            if args.detect_classes else None
        )
        detector = Detector(
            model_name=args.detect_model,
            conf=args.detect_conf,
            device=det_device,
            classes=detect_classes,
            imgsz=args.detect_imgsz,
        )
        if detector.open_vocab:
            print(f"[L1] open-vocab detector {args.detect_model} "
                  f"classes={detect_classes} imgsz={args.detect_imgsz}")

    # ---- M4.3: lazy-load fallback LLM only when --enable-llm-fallback --
    fallback_llm_client = None
    if args.enable_llm_fallback:
        if not args.fallback_llm:
            print("[fatal] --enable-llm-fallback requires --fallback-llm "
                  "<model_path> (e.g. 'Qwen/Qwen2.5-VL-7B-Instruct')")
            store.close()
            embedder.unload()
            return 1
        from mva.l4_llm import LLMClient
        print(f"[L4] Loading fallback LLM {args.fallback_llm} (int4 "
              f"quantized to coexist with embedder)")
        fallback_llm_client = LLMClient(
            model_path=args.fallback_llm, quantization="int4",
        )

    # ---- main loop -----------------------------------------------------
    try:
        stats = ingest_scene(
            adapter=adapter, scene_id=args.scene, view_ids=view_ids,
            config=config,
            embedder=embedder, detector=detector,
            embed_bboxes=args.embed_bboxes,
            store=store, vstore=vstore,
            segments_per_view=per_view_cap,
            appearance_threshold=args.appearance_threshold,
            track=args.track,
            track_iou=args.track_iou,
            track_conf_threshold=args.detect_conf,
            tracker_algorithm=args.tracker,
            rois_dir=rois_dir,
            enable_llm_fallback=args.enable_llm_fallback,
            fallback_llm_client=fallback_llm_client,
            fallback_confidence_threshold=args.fallback_confidence_threshold,
        )
    finally:
        print("[L4] Unloading embedder to free GPU memory")
        embedder.unload()
        if fallback_llm_client is not None:
            print("[L4] Unloading fallback LLM to free GPU memory")
            fallback_llm_client.unload()
        store.close()

    final_chroma = vstore.collection.count()
    cv_mode = getattr(adapter, "cross_view_linking_mode", "none")
    print(f"\n[ingest] {stats['segments']} segments / "
          f"{stats['tracklets']} tracklets / "
          f"{stats['detections']} detections / "
          f"{stats['bbox_embeddings']} bbox embeddings / "
          f"{stats['cross_view_links']} cross-view links ({cv_mode}) written")
    print(f"[ingest] ChromaDB: {initial_chroma} → {final_chroma} entries")
    return 0


# --------------------------------------------------------------------------
# Core loop — exposed so tests can drive it with mocks
# --------------------------------------------------------------------------


def ingest_scene(
    *,
    adapter: Any,
    scene_id: str,
    view_ids: list[str],
    config: SegmenterConfig,
    embedder: Any,
    detector: Optional[Any],
    embed_bboxes: bool,
    store: WorldStateStore,
    vstore: VectorStore,
    segments_per_view: float = float("inf"),
    appearance_threshold: Optional[float] = None,
    track: bool = True,
    track_iou: float = 0.5,
    track_conf_threshold: float = 0.25,
    tracker_algorithm: str = "iou_greedy",
    rois_dir: Optional[str] = None,
    enable_llm_fallback: bool = False,
    fallback_llm_client: Any = None,
    fallback_confidence_threshold: float = 0.5,
    fallback_max_per_view: int = 1,
) -> dict[str, int]:
    """Run the unified ingest loop on one scene. Returns counters.

    `detector` is a `mva.l1_perception.Detector`-like object exposing
    `.detect(frame) -> list[Detection]`. `None` skips detection entirely.

    `embedder` only needs `.encode_images(list[np.ndarray])` and
    `.encode_image(np.ndarray)`.

    `segments_per_view` caps how many segments each view ingests
    independently, ensuring equal temporal coverage across views.
    `float('inf')` means no cap (CLI `--segments-per-view 0`).

    `appearance_threshold` (M3.0): cosine bar for L2 cross-view linker.
    None → use built-in defaults (0.45 for appearance mode, 0.5 for
    synchronized mode secondary filter). Pass explicit value to sweep.

    `track` (M3.1): when True (default), per-view ByteTracker assigns
    track IDs across the K sampled frames of a segment. Detections that
    the tracker associates to the same identity get one DuckDB row
    (multi-frame bboxes JSON) and one mean-pooled bbox embedding.
    When False, every detection writes an independent row (M3.0 baseline).
    Returns `stats["tracklets"]` separately so callers can distinguish
    "raw detections" from "consolidated tracklet rows".

    `rois_dir` (M4.1): when set, the rep-frame bbox crop of each track is
    written as a JPEG to this directory and its path attached to the
    ViewObservation as `roi_uri`. None (default) → no caching, LLM
    cross-view fallback uses delayed-decode through `source_uri` +
    `frame_idx` instead. Cheap (~10ms/track) but adds disk
    (~10-50MB/scene).
    """
    from pathlib import Path as _Path
    total_segments = 0
    total_tracklets = 0
    total_detections = 0
    total_bbox_embeddings = 0
    observations_for_l2: list[ViewObservation] = []

    # M3.1: per-view tracker so IDs don't bleed across views.
    # Reset per segment — segment-local IDs only (cross-segment identity
    # merging is M5 stretch work).
    view_trackers: dict[str, Optional[ByteTracker]] = {
        vid: (ByteTracker(
            conf_threshold=track_conf_threshold, iou_threshold=track_iou,
            algorithm=tracker_algorithm,
        ) if track and detector is not None else None)
        for vid in view_ids
    }

    # Per-view iteration with independent caps. Each view ingests up to
    # `segments_per_view` segments, guaranteeing equal temporal coverage
    # across views regardless of source video length.
    view_iters = {
        vid: iter(adapter.iter_segments(scene_id, vid, config))
        for vid in view_ids
    }
    view_counts: dict[str, int] = {vid: 0 for vid in view_ids}
    remaining_views = list(view_ids)
    while remaining_views:
        for view_id in list(remaining_views):
            if view_counts[view_id] >= segments_per_view:
                remaining_views.remove(view_id)
                continue
            try:
                seg = next(view_iters[view_id])
            except StopIteration:
                remaining_views.remove(view_id)
                continue

            tracker = view_trackers.get(view_id)
            if tracker is not None:
                tracker.reset()

            # 1. Segment-level embedding ----------------------------------
            seg_vec = embedder.encode_images(seg.frames)
            seg_chroma_id = _add_segment_vector(
                vstore, seg_vec, scene_id, seg,
            )

            # 2. Detection → per-track aggregation within the segment ----
            #
            # `per_track_obs` is keyed by either:
            #   - "track{N}" when tracking is ON (tracker maps multiple
            #     frame observations of the same identity to one key)
            #   - "f{sample_idx}-d{det_idx}" when tracking is OFF
            #     (every detection gets its own key → preserves M3.0
            #     baseline of one tracklet row per detection)
            per_track_obs: dict[
                str,
                list[tuple[int, Any, Any, int]],
            ] = defaultdict(list)

            if detector is not None:
                for sample_idx, (frame, src_fidx) in enumerate(
                    zip(seg.frames, seg.frame_indices)
                ):
                    dets = detector.detect(frame)
                    if tracker is not None:
                        raw = tracker.update(
                            dets, frame.shape[0], frame.shape[1], frame=frame,
                        )
                        labeled = [(det, f"track{tid}") for det, tid in raw]
                    else:
                        labeled = [
                            (det, f"f{sample_idx}-d{i}")
                            for i, det in enumerate(dets)
                        ]
                    for det, label in labeled:
                        per_track_obs[label].append(
                            (sample_idx, det, frame, src_fidx)
                        )

            # 3. Write per-track to DuckDB + ChromaDB --------------------
            counts: Counter[str] = Counter()
            for track_label, obs_list in per_track_obs.items():
                tracklet_id = (
                    f"{view_id}-seg{seg.segment_idx:04d}-{track_label}"
                )

                # Pre-compute rep frame indices: always-defined so the L2
                # observation step at the bottom can rely on them whether
                # or not we ran bbox embedding for this track.
                rep_idx_in_list = len(obs_list) // 2
                rep_sample_idx, rep_det, rep_frame, rep_src_fidx = (
                    obs_list[rep_idx_in_list]
                )

                # Track-level bbox embedding (mean-pool over all crops in
                # this track within this segment). One ChromaDB entry per
                # track instead of K — eliminates the M2.8 near-duplicate
                # problem (PROBLEMS P3-11).
                bbox_chroma_id: Optional[str] = None
                track_vec: Optional[list[float]] = None
                track_roi_uri: Optional[str] = None
                if embed_bboxes:
                    crops = []
                    for _, det, f, _ in obs_list:
                        c = _crop_bbox(f, det.bbox)
                        if c is not None:
                            crops.append(c)
                    if crops:
                        track_vec = embedder.encode_images(crops)
                        # M3.6.D: capture every distinct class seen in this
                        # class-agnostic IoU track so retrieval can fall
                        # back to the multiset when rep_det.class_name is
                        # the minority label (PROBLEMS P3-12).
                        classes_csv = ",".join(sorted({
                            det.class_name for _, det, _, _ in obs_list
                        }))
                        bbox_chroma_id = _add_bbox_vector(
                            vstore, track_vec, scene_id, seg,
                            track_label, rep_sample_idx, rep_src_fidx, rep_det,
                            n_frames=len(crops),
                            classes_in_track=classes_csv,
                        )
                        # M4.1: write rep-frame crop to ROI cache (opt-in).
                        # `rep_idx_in_list` indexes obs_list; we need the
                        # crop at the corresponding position inside `crops`.
                        # Crops can be shorter than obs_list when some
                        # frames produced empty crops, so clamp.
                        if rois_dir is not None:
                            import cv2  # type: ignore
                            crop_idx = min(rep_idx_in_list, len(crops) - 1)
                            roi_path = _Path(rois_dir) / f"{bbox_chroma_id}.jpg"
                            cv2.imwrite(str(roi_path), crops[crop_idx])
                            track_roi_uri = str(roi_path)
                        total_bbox_embeddings += 1

                # DuckDB row: one per track. The bboxes JSON now holds the
                # bbox at every sampled frame the track was seen in.
                bboxes_json = []
                span = seg.end_t - seg.start_t
                K = max(1, len(seg.frames))
                for sample_idx, det, _, _ in obs_list:
                    frame_t = seg.start_t + (sample_idx + 0.5) * (span / K)
                    bboxes_json.append([
                        float(frame_t),
                        float(det.bbox[0]), float(det.bbox[1]),
                        float(det.bbox[2]), float(det.bbox[3]),
                        det.class_name, float(det.confidence),
                    ])
                    counts[det.class_name] += 1
                store.insert_tracklet(
                    view_id=view_id,
                    tracklet_id=tracklet_id,
                    t_start=seg.start_t,
                    t_end=seg.end_t,
                    bboxes=bboxes_json,
                    embedding_ref=bbox_chroma_id,
                    segment_idx=seg.segment_idx,
                )
                total_tracklets += 1
                total_detections += len(obs_list)

                # One L2 observation per track (representative frame's bbox
                # + track-level mean-pooled embedding). Compared to M3.0
                # (one observation per detection), Hungarian's input matrix
                # is much smaller and rows correspond to stable identities
                # instead of arbitrary per-frame samples. M4.1: also carry
                # roi_uri / frame_idx / source_uri so LLMCrossViewLinker
                # fallback can hybrid-load the ROI without re-decoding.
                h, w = rep_frame.shape[:2]
                observations_for_l2.append(_build_observation(
                    view_id, tracklet_id, seg, rep_sample_idx,
                    rep_det, track_vec, w, h,
                    roi_uri=track_roi_uri,
                    frame_idx=int(rep_src_fidx),
                    source_uri=seg.source_uri,
                ))

            # 4. Segment row in DuckDB ------------------------------------
            detected_classes_str: Optional[str] = None
            detected_counts: Optional[dict[str, int]] = None
            if counts:
                detected_classes_str = ",".join(sorted(counts))
                detected_counts = dict(counts)
            store.insert_segment(
                view_id=view_id,
                segment_idx=seg.segment_idx,
                start_t=seg.start_t,
                end_t=seg.end_t,
                source_uri=seg.source_uri,
                embed_chroma_id=seg_chroma_id,
                nframes_sampled=len(seg.frames),
                detected_classes=detected_classes_str,
                detected_counts=detected_counts,
            )

            total_segments += 1
            view_counts[view_id] += 1

    # Warn when per-view cap clipped work — probe one more segment from
    # each capped view to confirm there was more to ingest.
    cap_finite = segments_per_view != float("inf")
    if cap_finite:
        capped_views: list[str] = []
        for vid in view_ids:
            if view_counts[vid] >= segments_per_view:
                try:
                    next(view_iters[vid])
                    capped_views.append(vid)
                except StopIteration:
                    pass
        if capped_views:
            print(
                f"[ingest] hit --segments-per-view="
                f"{int(segments_per_view)} cap on {len(capped_views)} "
                f"view(s) ({capped_views}). Pass --segments-per-view 0 "
                f"to ingest everything, or raise the cap."
            )

    # 5. M3.0 — L2 cross-view link post-pass over all collected observations.
    # M4.3: optional LLM second-opinion on low-confidence links.
    total_cross_view_links = _link_cross_views(
        adapter, observations_for_l2, store,
        appearance_threshold=appearance_threshold,
        enable_llm_fallback=enable_llm_fallback,
        llm_client=fallback_llm_client,
        llm_confidence_threshold=fallback_confidence_threshold,
        llm_fallback_max_per_view=fallback_max_per_view,
    )

    return {
        "segments": total_segments,
        "tracklets": total_tracklets,
        "detections": total_detections,
        "bbox_embeddings": total_bbox_embeddings,
        "cross_view_links": total_cross_view_links,
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _add_segment_vector(vstore, seg_vec, scene_id, seg) -> str:
    """Insert segment-level vector. `vector_type="frame"` so it sorts with
    other frame-like embeddings; `vector_kind="segment"` in metadata lets
    L6 tools filter segment-vs-bbox at query time."""
    return vstore.add(
        seg_vec,
        vector_type="frame",
        view_id=f"{scene_id}::{seg.view_id}",
        tracklet_id=f"seg{seg.segment_idx:04d}",
        extra_metadata={
            "vector_kind": "segment",
            "scene_id": scene_id,
            "view_id_raw": seg.view_id,
            "segment_idx": seg.segment_idx,
            "start_t": float(seg.start_t),
            "end_t": float(seg.end_t),
            "source_uri": seg.source_uri,
            "nframes_sampled": len(seg.frames),
            # chunk_id mirrors segment_idx so ChromaDB ids stay unique
            # per-segment (P2-04 carryover from M2.7).
            "chunk_id": int(seg.segment_idx),
        },
        document=f"{seg.view_id} [{seg.start_t:.1f}-{seg.end_t:.1f}s]",
    )


def _add_bbox_vector(
    vstore, bbox_vec, scene_id, seg,
    track_label: str, rep_sample_idx: int, rep_src_fidx: int, rep_det,
    *, n_frames: int = 1, classes_in_track: Optional[str] = None,
) -> str:
    """Insert track-level bbox vector with metadata that lets retrieval map
    back to (segment, representative frame, bbox, class). `n_frames` is
    how many frames mean-pooled into this vector (1 in --no-track mode,
    K in tracked mode).

    `classes_in_track` is a comma-separated sorted CSV of every distinct
    `class_name` seen across the K frames merged into this track. M3.1's
    class-agnostic IoU tracker can merge cat/dog YOLO mis-labels into one
    track; without this field, the minority class is lost in the
    rep-frame-only `class_name` metadata (PROBLEMS P3-12). When omitted
    we default to `rep_det.class_name` so single-class tracks still
    populate the field (downstream consumers don't need to handle missing).
    """
    x1, y1, x2, y2 = rep_det.bbox
    return vstore.add(
        bbox_vec,
        vector_type="reid",
        view_id=f"{scene_id}::{seg.view_id}",
        tracklet_id=f"seg{seg.segment_idx:04d}-{track_label}",
        extra_metadata={
            "vector_kind": "bbox",
            "scene_id": scene_id,
            "view_id_raw": seg.view_id,
            "segment_idx": seg.segment_idx,
            "track_label": track_label,
            "n_frames_in_track": int(n_frames),
            "sample_frame_idx": int(rep_sample_idx),
            "source_frame_idx": int(rep_src_fidx),
            "bbox_x1": float(x1),
            "bbox_y1": float(y1),
            "bbox_x2": float(x2),
            "bbox_y2": float(y2),
            "class_name": rep_det.class_name,
            "classes_in_track": classes_in_track or rep_det.class_name,
            "confidence": float(rep_det.confidence),
            "source_uri": seg.source_uri,
            "start_t": float(seg.start_t),
            "end_t": float(seg.end_t),
        },
        document=f"{rep_det.class_name} @ {seg.view_id} seg{seg.segment_idx} "
                 f"track{track_label} (conf={rep_det.confidence:.2f}, "
                 f"frames={n_frames})",
    )


def _build_observation(
    view_id: str,
    tracklet_id: str,
    seg,
    sample_idx: int,
    det,
    bbox_vec: Optional[list[float]],
    frame_w: int,
    frame_h: int,
    *,
    roi_uri: Optional[str] = None,
    frame_idx: Optional[int] = None,
    source_uri: Optional[str] = None,
) -> ViewObservation:
    """Construct the L2 ViewObservation for one track.

    Normalizes the pixel bbox to [0, 1] so the geometric linker (MATRIX)
    can compare across views with different image resolutions. The
    appearance linker (MVU-Eval) ignores the bbox and uses
    `appearance_embedding` only.

    `roi_uri` / `frame_idx` / `source_uri` (M4.1) attach the ROI
    pointers `LLMCrossViewLinker` needs for its hybrid loader. Default
    None preserves the M3.x calling convention; ingest sets them when
    `--cache-rois` is on or when the source path is known regardless.
    """
    x1, y1, x2, y2 = det.bbox
    w_safe = max(1, frame_w)
    h_safe = max(1, frame_h)
    norm_bbox = (
        float(x1) / w_safe, float(y1) / h_safe,
        float(x2) / w_safe, float(y2) / h_safe,
    )
    span = seg.end_t - seg.start_t
    K = max(1, len(seg.frames))
    t = seg.start_t + (sample_idx + 0.5) * (span / K)
    return ViewObservation(
        view_id=view_id,
        tracklet_id=tracklet_id,
        t=float(t),
        bbox=norm_bbox,
        class_name=det.class_name,
        appearance_embedding=bbox_vec,
        segment_idx=seg.segment_idx,
        roi_uri=roi_uri,
        frame_idx=frame_idx,
        source_uri=source_uri,
    )


def _link_cross_views(
    adapter: Any,
    observations: list[ViewObservation],
    store: WorldStateStore,
    appearance_threshold: Optional[float] = None,
    *,
    enable_llm_fallback: bool = False,
    llm_client: Any = None,
    llm_confidence_threshold: float = 0.5,
    llm_fallback_max_per_view: int = 1,
) -> int:
    """Dispatch M3.0 L2 cross-view linking based on the adapter's
    declared mode. Returns the number of links written.

    M4.3 LLM fallback (PLAN §6.2 M4.3, 2026-05-23 design lock):
    when `enable_llm_fallback=True` and `llm_client` is set, every link
    with confidence < `llm_confidence_threshold` (default 0.5) is sent
    to `LLMCrossViewLinker` for a second opinion. The LLM either
    confirms (the link is replaced with `created_by="llm"` + the LLM's
    confidence) or rejects (the link is dropped). High-confidence links
    pass through untouched. Throttled to `llm_fallback_max_per_view`
    LLM calls per view across the whole call — coarser than the literal
    "per view per segment" rule for safety; relax if needed.

    `synchronized` mode: GeometricCrossViewLinker (Hungarian on bbox
    geometry + optional appearance secondary filter).
    `appearance` mode: AppearanceCrossViewLinker (pure cosine on
    `appearance_embedding`, requires bbox embeddings to be populated).
    `none` / no observations: no-op.

    `appearance_threshold=None` uses the mode-appropriate default
    (`_APPEARANCE_ONLY_THRESHOLD` for appearance,
    `_GEOMETRIC_APPEARANCE_THRESHOLD` for synchronized)."""
    mode = getattr(adapter, "cross_view_linking_mode", "none")
    if mode == "none" or not observations:
        return 0

    if mode == "synchronized":
        thresh = (appearance_threshold if appearance_threshold is not None
                  else _GEOMETRIC_APPEARANCE_THRESHOLD)
        linker = GeometricCrossViewLinker(appearance_threshold=thresh)
    elif mode == "appearance":
        # Pure appearance requires bbox embeddings on both sides.
        with_embed = [o for o in observations if o.appearance_embedding]
        if not with_embed:
            print("[ingest] WARN: cross_view_linking_mode='appearance' "
                  "but no bbox embeddings present (--no-embed-bboxes?); "
                  "skipping L2.")
            return 0
        thresh = (appearance_threshold if appearance_threshold is not None
                  else _APPEARANCE_ONLY_THRESHOLD)
        linker = AppearanceCrossViewLinker(appearance_threshold=thresh)
        observations = with_embed
    else:
        print(f"[ingest] WARN: unknown cross_view_linking_mode={mode!r}; "
              "skipping L2.")
        return 0

    links = linker.link(observations)

    if enable_llm_fallback and llm_client is not None and links:
        links = _llm_fallback_upgrade_links(
            links, observations, llm_client,
            threshold=llm_confidence_threshold,
            max_per_view=llm_fallback_max_per_view,
        )

    for link in links:
        store.insert_cross_view_link(link)
    return len(links)


def _llm_fallback_upgrade_links(
    links: list,
    observations: list[ViewObservation],
    llm_client: Any,
    *,
    threshold: float,
    max_per_view: int,
) -> list:
    """M4.3 (PLAN §6.2): per-link upgrade of low-confidence cross-view
    links via LLMCrossViewLinker.

    For each link with `confidence < threshold`, ask the LLM (one
    pairwise call) whether the two observations are truly the same
    physical target. The LLM either:

    - **confirms** (same_object=true, confidence >= threshold) → the
      original link is replaced with the LLM's version (created_by="llm",
      LLM-reported confidence). `link_id` stays identical because
      `make_link_id` is deterministic on `(view_id, tracklet_id)` pairs,
      so DuckDB's INSERT OR REPLACE simply updates the row.
    - **rejects** (LLM returns no link for that pair) → the original
      low-confidence link is dropped entirely.

    Throttling: a per-view counter caps LLM invocations at `max_per_view`
    PER view_id across the whole linker call. Coarser than the literal
    "per view per segment" PLAN spec (cheap to relax later when we have
    per-segment cost data). High-confidence links never trigger the LLM
    and don't count against the throttle.
    """
    from collections import defaultdict
    from mva.l2_crossview import LLMCrossViewLinker

    obs_by_key = {(o.view_id, o.tracklet_id): o for o in observations}
    invocation_counts: dict[str, int] = defaultdict(int)
    llm_linker = LLMCrossViewLinker(
        llm_client=llm_client,
        confidence_threshold=threshold,
    )

    upgraded: list = []
    for link in links:
        if link.confidence >= threshold:
            upgraded.append(link)
            continue

        view_ids_in_link = {v for v, _ in link.view_observations}
        # Throttle BEFORE incrementing — once any participating view has
        # used up its quota, all subsequent low-conf links touching that
        # view stay at their original (geometric/appearance) score.
        if any(invocation_counts[v] >= max_per_view for v in view_ids_in_link):
            upgraded.append(link)
            continue

        obs_for_pair = [
            obs_by_key.get(key) for key in link.view_observations
        ]
        if not all(o is not None for o in obs_for_pair):
            # Lost observation → can't rebuild the prompt; keep original
            upgraded.append(link)
            continue

        for v in view_ids_in_link:
            invocation_counts[v] += 1
        llm_links = llm_linker.link([o for o in obs_for_pair if o is not None])

        if llm_links:
            # LLM confirmed same object → replace with LLM-judged link.
            # link_id matches via make_link_id determinism, so this is a
            # clean upgrade (no duplicate row).
            upgraded.append(llm_links[0])
            print(
                f"[ingest] M4.3 LLM fallback: upgraded link "
                f"{link.link_id} ({link.created_by} conf={link.confidence:.2f}) "
                f"→ created_by=llm conf={llm_links[0].confidence:.2f}"
            )
        else:
            # LLM rejected (same_object=false OR below LLM threshold) →
            # drop the low-conf link entirely. The algorithmic linker
            # was uncertain; the LLM second opinion says "no" → trust it.
            print(
                f"[ingest] M4.3 LLM fallback: dropped low-conf link "
                f"{link.link_id} ({link.created_by} conf={link.confidence:.2f}) "
                f"— LLM judged not the same object"
            )

    # LLM-upgraded confidences may shift the DESC ordering
    upgraded.sort(key=lambda link: link.confidence, reverse=True)
    return upgraded


def _crop_bbox(frame, bbox):
    """Integer-clamp the bbox + crop. Returns None if empty crop."""
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(round(bbox[0]))))
    y1 = max(0, min(h, int(round(bbox[1]))))
    x2 = max(0, min(w, int(round(bbox[2]))))
    y2 = max(0, min(h, int(round(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop
