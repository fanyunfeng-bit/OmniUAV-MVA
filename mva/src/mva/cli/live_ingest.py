"""Phase 2 — in-process incremental ("live") ingest for the streaming UI.

A background worker periodically ingests the next window of the looping view
videos into the SAME DuckDB + ChromaDB the chat queries, **reusing the
QueryService's already-resident embedder** — a subprocess `mva ingest` would
load a second ~16 GB embedder and OOM the 24 GB card. YOLOE is loaded per cycle
and dropped after (it only adds ~0.5 GB, but the per-cycle unload keeps headroom
on the budget: embedder 16 + LLM int4 5 ≈ 22 GB resident, measured peak with
YOLOE ≈ 23.2 GB / 24). A shared GPU lock makes an ingest cycle and a chat
generation mutually exclusive so their activations never stack into an OOM
(the price is a chat-latency blip during a cycle — the accepted tradeoff).

History is a stack-bounded FIFO: each cycle appends one segment per view with a
monotonically-increasing `segment_idx`, then evicts the oldest beyond `keep_n`
from BOTH stores so the rolling window stays bounded.

We drive the real `ingest_scene` via a one-window adapter rather than
re-implementing detect→track→embed→write, so live rows are byte-identical to
`mva ingest` rows and the G-1/G-2 query tools read them with no special-casing.
"""
from __future__ import annotations

import gc
import threading
from typing import Optional

from mva.segmentation import Segment, SegmenterConfig


def _decode_window(path: str, w_start: float, w_end: float, n: int):
    """Decode `n` evenly-spaced (frames, frame_indices) from [w_start, w_end]."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        span = max(0.1, w_end - w_start)
        frames, idxs = [], []
        for i in range(n):
            t = w_start + (i + 0.5) * span / n
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if ok and img is not None:
                frames.append(img)
                idxs.append(int(t * fps) if fps > 0 else i)
        return frames, idxs
    finally:
        cap.release()


class _OneWindowAdapter:
    """Yields exactly one Segment per view for the current cycle's window, so
    we can drive the real `ingest_scene` (same write format as `mva ingest`)."""

    name = "live"
    cross_view_linking_mode = "appearance"
    supports_cross_view_linking = True
    supports_qa_eval = False

    def __init__(self, sources, seg_idx, w_start, w_end, nframes):
        self._sources = sources
        self._seg_idx = seg_idx
        self._w_start = w_start
        self._w_end = w_end
        self._nframes = nframes

    def iter_segments(self, scene_id, view_id, config):
        path = self._sources.get(view_id)
        if not path:
            return
        frames, idxs = _decode_window(
            path, self._w_start, self._w_end, self._nframes)
        if not frames:
            return
        yield Segment(
            view_id=view_id, segment_idx=self._seg_idx,
            start_t=float(self._w_start), end_t=float(self._w_end),
            frames=frames, frame_indices=idxs, source_uri=str(path),
            metadata={"live": True},
        )


class LiveIngestor:
    """Background worker: roll a bounded window of fresh ingest into the live DB."""

    def __init__(
        self, *, embedder, store, vstore, sources: dict,
        loop_duration: float, gpu_lock,
        detect_model: str = "yoloe-11l-seg.pt",
        detect_classes=("boat", "ship", "drone", "uav"),
        detect_imgsz: int = 1280, detect_conf: float = 0.25,
        window_sec: float = 5.0, nframes: int = 4, keep_n: int = 30,
        cycle_sec: float = 5.0, rois_dir: Optional[str] = None,
        embed_bboxes: bool = False,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.vstore = vstore
        self.sources = dict(sources)
        self.views = list(self.sources)
        self.loop = loop_duration if loop_duration > 0 else window_sec
        self.gpu_lock = gpu_lock
        self.detect_model = detect_model
        self.detect_classes = list(detect_classes)
        self.detect_imgsz = detect_imgsz
        self.detect_conf = detect_conf
        self.window_sec = window_sec
        self.nframes = nframes
        self.keep_n = keep_n
        self.cycle_sec = cycle_sec
        # embed_bboxes is OFF by default: encoding every track's crops on top of
        # the resident embedder(16G)+LLM(5G) OOMs the 24G card. Counts/stats read
        # DuckDB (unaffected); only live bbox-level look_at degrades to scene mode.
        self.embed_bboxes = embed_bboxes
        self.rois_dir = rois_dir if embed_bboxes else None
        # Drop the segments secondary index: the live re-ingest creates
        # duplicate (view_id, start_t) keys whose ART-index delete fatally
        # crashes DuckDB during FIFO eviction. PK delete is unaffected.
        self.store.drop_secondary_indexes()
        self._seg_idx = self._next_seg_idx()
        self._video_cursor = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cycles = 0
        self.errors = 0

    def _next_seg_idx(self) -> int:
        """Continue the monotonic counter above any existing segments so a
        restarted UI keeps appending rather than colliding/overwriting."""
        mx = -1
        for v in self.views:
            idxs = self.store.list_segment_indices(v)
            if idxs:
                mx = max(mx, max(idxs))
        return mx + 1

    # ---- thread lifecycle -------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="live-ingest")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_cycle()
            except Exception as exc:                       # noqa: BLE001
                self.errors += 1
                msg = str(exc)
                print(f"[live-ingest] cycle error: "
                      f"{type(exc).__name__}: {msg}", flush=True)
                # A fatally-invalidated DuckDB connection can't recover; stop
                # rather than spin printing the same error every cycle.
                if "invalidated" in msg or "fatal error" in msg.lower():
                    print("[live-ingest] DB connection invalidated — stopping "
                          "worker (chat stays up on last-good data).", flush=True)
                    return
            self._stop.wait(self.cycle_sec)               # interruptible sleep

    # ---- one cycle --------------------------------------------------------

    def run_cycle(self) -> None:
        from mva.cli.ingest import ingest_scene
        from mva.l1_perception import Detector

        w_start = self._video_cursor
        w_end = min(self.loop, w_start + self.window_sec)
        adapter = _OneWindowAdapter(
            self.sources, self._seg_idx, w_start, w_end, self.nframes)
        cfg = SegmenterConfig(
            window_sec=self.window_sec, stride_sec=self.window_sec,
            nframes_per_segment=self.nframes)

        detector = Detector(
            model_name=self.detect_model, conf=self.detect_conf,
            classes=self.detect_classes, imgsz=self.detect_imgsz)
        try:
            # ingest ⊥ chat: hold the shared GPU lock for the whole cycle so an
            # ingest's activations never stack on top of a chat generation's.
            with self.gpu_lock:
                ingest_scene(
                    adapter=adapter, scene_id="live", view_ids=self.views,
                    config=cfg, embedder=self.embedder, detector=detector,
                    embed_bboxes=self.embed_bboxes,
                    store=self.store, vstore=self.vstore,
                    segments_per_view=float("inf"), appearance_threshold=None,
                    track=True, track_iou=0.5,
                    track_conf_threshold=self.detect_conf,
                    tracker_algorithm="iou_greedy", rois_dir=self.rois_dir,
                    enable_llm_fallback=False, fallback_llm_client=None,
                    fallback_confidence_threshold=0.5,
                )
        finally:
            detector = None                                # per-cycle YOLOE unload
            _free_cuda()

        self._prune()
        self.cycles += 1
        print(f"[live-ingest] cycle {self.cycles} seg={self._seg_idx} "
              f"window=[{w_start:.0f},{w_end:.0f}]s "
              f"kept≤{self.keep_n}/view", flush=True)
        self._seg_idx += 1
        self._video_cursor = w_end if w_end < self.loop else 0.0   # loop

    def _prune(self) -> None:
        """Evict the oldest segments beyond keep_n from DuckDB + ChromaDB."""
        for v in self.views:
            idxs = self.store.list_segment_indices(v)
            if len(idxs) <= self.keep_n:
                continue
            for old in idxs[: len(idxs) - self.keep_n]:
                chroma_ids = self.store.delete_segment(v, old)
                self.vstore.delete(chroma_ids)


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def rois_dir_for(chroma_dir: Optional[str]) -> Optional[str]:
    """Match `mva ingest`'s ROI cache location so live look_at (G-2.1) works."""
    return f"{chroma_dir}-rois" if chroma_dir else None
