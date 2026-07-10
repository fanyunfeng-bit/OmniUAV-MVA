"""`mva perceive` — L0 → L1 → L2 → L5 DuckDB.

⚠ Frozen at M2.7 (kept for back-reference; M2.8 mainline is `mva ingest`).
Writes the M2.7 schema (no `segments` table, no `tracklets.segment_idx`
column); new analysis tools won't read its output.

Iterates frames from the requested views, runs YOLOv11 detection per frame,
runs L2 GeometricCrossViewLinker across same-t same-class observations,
persists tracklets + cross-view links to DuckDB.

Dataset-agnostic via `DatasetAdapter.open_view`. For datasets that don't
support cross-view linking (e.g. MVU-Eval), L2 is skipped automatically
when `adapter.supports_cross_view_linking == False`.
"""
from __future__ import annotations

import argparse

from mva.cli._common import (
    add_dataset_args,
    add_store_args,
    resolve_dataset,
)
from mva.contracts import ViewObservation
from mva.l2_crossview import GeometricCrossViewLinker
from mva.l5_state import WorldStateStore


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "perceive",
        help="Run detection + cross-view linking and persist to DuckDB",
    )
    add_dataset_args(p, scene_required=True)
    add_store_args(p, db_required=True)
    p.add_argument("--views", nargs="+", default=None,
                   help="View ids to process (default: all views in the scene)")
    p.add_argument("--max-frames", type=int, default=10)
    p.add_argument("--sample-fps", type=float, default=None,
                   help="Downsample stream FPS (default: native)")
    p.add_argument("--link-threshold", type=float, default=0.3,
                   help="L2 GeometricCrossViewLinker distance threshold")
    p.add_argument("--cross-view", choices=["auto", "on", "off"], default="auto",
                   help="L2 cross-view linking: auto=adapter默认, on=强制开, off=强制关")
    p.set_defaults(func=cmd_perceive)


def cmd_perceive(args: argparse.Namespace) -> int:
    try:
        from mva.l1_perception import Detector
    except ImportError:
        print("[fatal] ultralytics not installed; install with: pip install 'mva[detection]'")
        return 1

    adapter = resolve_dataset(args)
    scene = adapter.get_scene(args.scene)
    view_ids = args.views or scene.view_ids
    if not view_ids:
        print(f"[fatal] scene {args.scene} has no views")
        return 1

    print(f"[L0] dataset={adapter.name} scene={args.scene} views={view_ids}")
    sources = [adapter.open_view(args.scene, v, args.sample_fps) for v in view_ids]

    print("[L1] loading YOLOv11n ...")
    detector = Detector()

    if args.cross_view == "off":
        do_cross_view = False
    elif args.cross_view == "on":
        if not adapter.supports_cross_view_linking:
            print(f"[fatal] adapter {adapter.name} 不支持 cross-view linking，无法 --cross-view on")
            return 1
        if len(sources) < 2:
            print(f"[fatal] --cross-view on 需要 ≥2 个 view，当前 {len(sources)}")
            return 1
        do_cross_view = True
    else:  # auto
        do_cross_view = adapter.supports_cross_view_linking and len(sources) >= 2
    linker = GeometricCrossViewLinker(distance_threshold=args.link_threshold) if do_cross_view else None
    print(f"[L2] cross-view linking: {'enabled' if do_cross_view else 'disabled'} (mode={args.cross_view})")

    print(f"[L5] WorldStateStore(db_path={args.db_path})")
    store = WorldStateStore(db_path=args.db_path)

    totals = {v: 0 for v in view_ids}
    link_total = 0
    iters = [iter(s) for s in sources]

    for frame_idx in range(args.max_frames):
        try:
            frames = [next(it) for it in iters]
        except StopIteration:
            print(f"[L0] one stream ended at frame {frame_idx}")
            break

        t_sync = float(frame_idx)
        print(f"\n--- frame pair {frame_idx} (t={t_sync}) ---")

        observations: list[ViewObservation] = []
        for frame in frames:
            h, w = frame.image.shape[:2]
            dets = detector.detect(frame.image)
            print(f"[L1] {frame.view_id}: {len(dets)} detections")
            for det_idx, det in enumerate(dets):
                tracklet_id = f"{frame.view_id}-f{frame_idx}-d{det_idx}"
                x1, y1, x2, y2 = det.bbox
                store.insert_tracklet(
                    view_id=frame.view_id,
                    tracklet_id=tracklet_id,
                    t_start=t_sync, t_end=t_sync,
                    bboxes=[(t_sync, x1, y1, x2, y2,
                             det.class_name, det.confidence)],
                )
                totals[frame.view_id] += 1
                if linker is not None:
                    observations.append(ViewObservation(
                        view_id=frame.view_id,
                        tracklet_id=tracklet_id,
                        t=t_sync,
                        bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                        class_name=det.class_name,
                    ))

        if linker is not None and observations:
            links = linker.link(observations)
            for link in links:
                store.insert_cross_view_link(link)
            link_total += len(links)
            if links:
                print(f"[L2] {len(links)} cross-view link(s) (top conf={links[0].confidence:.3f})")

    print("\n========== L5 Validation ==========")
    for v in view_ids:
        rows = store.query_tracklets(v)
        print(f"  query_tracklets({v}) → {len(rows)} rows")
    all_links = store.query_cross_view_links()
    print(f"  query_cross_view_links() → {len(all_links)} rows")
    print(f"\nTotals: tracklets={totals}, links={link_total}")
    print(f"[perceive] DuckDB written to {args.db_path}")
    store.close()
    return 0
