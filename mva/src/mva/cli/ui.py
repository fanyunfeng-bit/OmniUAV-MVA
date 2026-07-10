"""`mva ui` — M5.4 Gradio main page (NL chat + segment playback).

Wraps QueryService in a gr.Blocks UI:
  - Chat panel (history + multi-modal attachments via MultimodalTextbox)
  - Segment hit list (dropdown over hits from the last response)
  - Video player (extracts clip via `ffmpeg -ss <start> -t <dur> -c copy`)

The annotation Tab (HITL P3-10) is intentionally NOT included — that's
the M6 standalone milestone per PLAN §6.3 LOCK.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from mva.cli._common import (
    add_cross_view_arg, add_embedder_args, add_llm_args, add_store_args,
)
from mva.cli.query import (
    QueryService, _check_db_populated, _resolve_quantize,
)
from mva.contracts import Attachment, RichQuery
from mva.l6_interaction.memory import ConversationMemory

# Extension → Attachment.kind. Conservative — Gradio only allows the file
# types passed via `file_types=...`, but we still default-route by ext.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


class LiveCapture:
    """Phase 1 — a shared, server-side "live stream" over the ingested view
    videos (display only; no ingest yet).

    Each view's source video loops on ONE shared virtual clock so every client
    sees the same playhead — the file is disguised as a live drone feed. This
    is a **process singleton** (one physical capture), built once in build_app
    and closed over by the gr.Timer tick; it must NOT live in per-session
    gr.State (that would give each browser tab its own divergent clock).

    Views are discovered from the DB (`segments.source_uri`), so any ingested
    scene with video-file sources gets live panels with no extra CLI args, and
    dropping in more views just means more panels next launch.
    """

    def __init__(self, view_sources: dict, display_width: int = 960) -> None:
        self.views = list(view_sources)
        self._sources = view_sources
        self._display_width = display_width
        self._caps: dict = {}
        self._durations: dict = {}
        self._t0: Optional[float] = None
        self._loop: Optional[float] = None
        self._lock = threading.Lock()

    @classmethod
    def from_service(cls, service: Any) -> "LiveCapture":
        """Discover (view_id → video file) from the DB's segments table.

        Defensive: missing store / query failure / non-file sources → no views
        (the UI simply renders without live panels)."""
        store = getattr(service, "store", None)
        if store is None:
            return cls({})
        try:
            rows = store.execute_readonly(
                "SELECT DISTINCT view_id, source_uri FROM segments "
                "WHERE source_uri IS NOT NULL ORDER BY view_id"
            )
        except Exception:
            return cls({})
        view_sources: dict = {}
        for r in rows:
            v, s = r.get("view_id"), r.get("source_uri")
            if v and s and v not in view_sources and Path(s).is_file():
                view_sources[v] = s
        return cls(view_sources)

    def _cap(self, view_id: str):
        import cv2
        cap = self._caps.get(view_id)
        if cap is None:
            cap = cv2.VideoCapture(self._sources[view_id])
            self._caps[view_id] = cap
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            self._durations[view_id] = (n / fps) if fps > 0 and n > 0 else 0.0
        return cap

    @property
    def sources(self) -> dict:
        """{view_id: video path} — consumed by the Phase 2 live ingestor."""
        return dict(self._sources)

    @property
    def loop_duration(self) -> float:
        """Shortest view duration — keeps views time-aligned (same rule as the
        Reservoir adapter's 'cap to shortest')."""
        if self._loop is None:
            for v in self.views:
                self._cap(v)
            durs = [d for d in self._durations.values() if d > 0]
            self._loop = min(durs) if durs else 0.0
        return self._loop

    def _playhead(self) -> float:
        if self._t0 is None:
            self._t0 = time.time()
        loop = self.loop_duration
        return ((time.time() - self._t0) % loop) if loop > 0 else 0.0

    def _resize(self, img):
        import cv2
        h, w = img.shape[:2]
        if w > self._display_width:
            scale = self._display_width / float(w)
            img = cv2.resize(img, (self._display_width, max(1, int(h * scale))))
        return img

    def current_frames(self) -> dict:
        """{view_id: RGB ndarray | None} at the current shared playhead."""
        import cv2
        t = self._playhead()
        out: dict = {}
        with self._lock:                          # cv2 handles are not thread-safe
            for v in self.views:
                cap = self._cap(v)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, img = cap.read()
                if not ok or img is None:
                    out[v] = None
                    continue
                out[v] = self._resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return out

    def status_line(self) -> str:
        t = self._playhead()
        loop = self.loop_duration
        mm, ss = divmod(int(t), 60)
        lm, ls = divmod(int(loop), 60)
        return (f"🔴 **LIVE** · 采集时刻 {mm:02d}:{ss:02d} / 循环 {lm:02d}:{ls:02d} · "
                f"{len(self.views)} 路无人机视角")


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "ui",
        help="M5.4 — Gradio main page (NL chat + segment playback)",
    )
    add_store_args(p, db_required=True)
    add_llm_args(p, llm_required=True)
    add_embedder_args(p)
    add_cross_view_arg(p)
    p.add_argument("--port", type=int, default=7860,
                   help="Gradio server port (default 7860)")
    p.add_argument("--share", action="store_true",
                   help="Generate a public gradio.live URL")
    p.add_argument("--no-browser", action="store_true",
                   help="Skip auto-opening a browser tab")
    p.add_argument("--server-name", default="127.0.0.1",
                   help="Bind address (default 127.0.0.1 — local only)")
    p.add_argument("--demo-answers", default=None,
                   help="Optional JSON file mapping question→canned answer. When "
                        "a chat question matches (after normalizing whitespace/"
                        "punctuation), the UI returns that answer verbatim and "
                        "skips the LLM. For DEMO display only — soft (opt-in per "
                        "launch, no source hardcoding); drop the flag to restore "
                        "the pure grounded system.")
    # ---- Phase 2: live incremental ingest (opt-in, GPU-heavy) ----
    p.add_argument("--live-ingest", action="store_true",
                   help="Phase 2: roll incremental ingest of the looping "
                        "videos into the live DB so chat reflects 'current' + "
                        "recent history. Reuses the resident embedder; loads "
                        "YOLOE per cycle. Requires --chroma-dir. (off by default)")
    p.add_argument("--ingest-window-sec", type=float, default=5.0,
                   help="Live-ingest: seconds of video per cycle (default 5)")
    p.add_argument("--ingest-cycle-sec", type=float, default=5.0,
                   help="Live-ingest: seconds between cycles (default 5)")
    p.add_argument("--keep-segments", type=int, default=30,
                   help="Live-ingest: bounded history — segments kept per view "
                        "(FIFO eviction beyond this; default 30)")
    p.add_argument("--ingest-detect-model", default="yoloe-11l-seg.pt",
                   help="Live-ingest detector (default yoloe-11l-seg.pt)")
    p.add_argument("--ingest-detect-classes", default="boat,ship,drone,uav",
                   help="Live-ingest open-vocab classes (default boat,ship,drone,uav)")
    p.add_argument("--ingest-detect-imgsz", type=int, default=1280,
                   help="Live-ingest detector imgsz (default 1280)")
    p.add_argument("--ingest-nframes", type=int, default=4,
                   help="Live-ingest: frames sampled per cycle window "
                        "(default 4 — kept low to bound embedder VRAM)")
    p.add_argument("--ingest-embed-bboxes", action="store_true",
                   help="Live-ingest: also embed per-track bbox crops "
                        "(enables live bbox-level look_at, but +VRAM — risks "
                        "OOM alongside the resident embedder+LLM; off by default)")
    # Optional dataset/scene — lets --live-ingest run on a FRESH/empty DB by
    # sourcing the view videos from the adapter instead of segments.source_uri.
    p.add_argument("--dataset", default=None,
                   help="Dataset adapter (e.g. pcl-sim) — for --live-ingest on a "
                        "fresh DB; sources the view videos from the scene.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Override the dataset's default root directory")
    p.add_argument("--scene", default=None,
                   help="Scene id (e.g. Reservoir) for --dataset")
    p.set_defaults(func=cmd_ui)


def cmd_ui(args: argparse.Namespace) -> int:
    # --live-ingest can start from a FRESH DB: create the empty schema so the
    # "DB not found" check doesn't fatal, then the worker fills it (sources come
    # from --dataset/--scene, not the DB).
    if (getattr(args, "live_ingest", False) and args.db_path != ":memory:"
            and not Path(args.db_path).exists()):
        from mva.l5_state import WorldStateStore
        Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
        WorldStateStore(db_path=args.db_path).close()
        print(f"[ui] live-ingest: created fresh DB at {args.db_path}", flush=True)

    if not _check_db_populated(args.db_path):
        return 1

    # Mirror cli/eval.py's decord check — fail fast with a helpful message
    # rather than 5 layers into build_app.
    import importlib.util
    if importlib.util.find_spec("gradio") is None:
        print("[fatal] gradio not installed. Run: pip install -e .[ui]")
        return 1

    if shutil.which("ffmpeg") is None:
        print("[ui] WARN: ffmpeg not on PATH — segment playback button "
              "will return None. Install with `sudo apt-get install ffmpeg` "
              "or `conda install -c conda-forge ffmpeg`.", flush=True)

    # Phase 2: the embedder(16G)+LLM int4(5G)+per-cycle YOLOE budget is razor-thin
    # on 24G. Set this BEFORE any CUDA init so the allocator can use expandable
    # segments (recovers fragmentation that otherwise tips an ingest cycle OOM).
    if getattr(args, "live_ingest", False):
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print("[ui] live-ingest: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
              flush=True)

    effective_quantize = _resolve_quantize(args)

    # Python pipes to file/tee are fully buffered — force flush so users
    # piping output to a log see progress (embedder load is ~30-90s and
    # quiet, otherwise the user thinks the process is hung).
    print("[ui] Loading QueryService (this may take 20-90s on first run; "
          "embedder + LLM load is silent until done)...", flush=True)
    with QueryService(
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        llm_model=args.llm,
        embedder_model=args.embedder_model if args.chroma_dir else None,
        embed_dim=args.embed_dim,
        quantization=effective_quantize,
        enable_cross_view=(args.cross_view == "auto"),
    ) as service:
        # Resolve the live view sources ONCE (DB first, adapter fallback for a
        # fresh DB) and share it between the top panels and the ingest worker.
        live = _resolve_live_capture(service, args)
        ingestor = _maybe_start_live_ingest(service, args, live)
        demo_answers = _load_demo_answers(getattr(args, "demo_answers", None))
        app = build_app(service, db_path=args.db_path, live=live,
                        demo_answers=demo_answers)
        try:
            app.launch(
                server_name=args.server_name,
                server_port=args.port,
                share=args.share,
                inbrowser=not args.no_browser,
                show_error=True,
                prevent_thread_lock=False,
            )
        except OSError as exc:
            if "address already in use" in str(exc).lower():
                print(f"[fatal] port {args.port} already in use — retry "
                      f"with `--port N` (any free port).")
                return 1
            raise
        finally:
            if ingestor is not None:
                print("[ui] stopping live-ingest worker...", flush=True)
                ingestor.stop()
    return 0


def _adapter_view_paths(args: argparse.Namespace) -> dict:
    """{view_id: video path} from --dataset/--scene, or {} if unavailable.

    Lets --live-ingest stream a FRESH DB: the adapter knows the view videos
    (pcl-sim exposes them as scene.metadata['view_paths']) when the DB has no
    segments yet."""
    if not (getattr(args, "dataset", None) and getattr(args, "scene", None)):
        return {}
    try:
        from mva.cli._common import resolve_dataset
        adapter = resolve_dataset(args)
        scene = adapter.get_scene(args.scene)
        paths = (scene.metadata or {}).get("view_paths", {}) or {}
        return {v: p for v, p in paths.items() if Path(p).is_file()}
    except Exception as exc:                                   # noqa: BLE001
        print(f"[ui] --dataset/--scene source resolution failed: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {}


def _resolve_live_capture(service: "QueryService", args: argparse.Namespace) -> "LiveCapture":
    """LiveCapture for the top panels + ingest worker. DB segments first; on a
    fresh DB fall back to --dataset/--scene so a from-zero stream still shows
    panels and has something to ingest."""
    live = LiveCapture.from_service(service)
    if not live.views and getattr(args, "live_ingest", False):
        sources = _adapter_view_paths(args)
        if sources:
            print(f"[ui] live sources from {args.dataset}/{args.scene}: "
                  f"{list(sources)}", flush=True)
            live = LiveCapture(sources)
    return live


def _maybe_start_live_ingest(service: "QueryService", args: argparse.Namespace,
                             live: "LiveCapture"):
    """Phase 2: start the background live-ingest worker if --live-ingest. Returns
    the worker (to stop on shutdown) or None. `live` carries the resolved sources
    (DB or adapter)."""
    if not getattr(args, "live_ingest", False):
        return None
    if service.embedder is None:
        print("[fatal] --live-ingest requires --chroma-dir (needs the embedder).")
        return None
    if not live.views:
        print("[ui] --live-ingest: no video sources (DB empty and no "
              "--dataset/--scene) — nothing to ingest, skipping.", flush=True)
        return None
    from mva.cli.live_ingest import LiveIngestor, rois_dir_for
    ingestor = LiveIngestor(
        embedder=service.embedder, store=service.store, vstore=service.vstore,
        sources=live.sources, loop_duration=live.loop_duration,
        gpu_lock=service._lock,                       # ingest ⊥ chat
        detect_model=args.ingest_detect_model,
        detect_classes=[c.strip() for c in args.ingest_detect_classes.split(",") if c.strip()],
        detect_imgsz=args.ingest_detect_imgsz, nframes=args.ingest_nframes,
        window_sec=args.ingest_window_sec, cycle_sec=args.ingest_cycle_sec,
        keep_n=args.keep_segments, embed_bboxes=args.ingest_embed_bboxes,
        rois_dir=rois_dir_for(args.chroma_dir),
    )
    ingestor.start()
    print(f"[ui] 🟢 live-ingest started: window={args.ingest_window_sec}s "
          f"cycle={args.ingest_cycle_sec}s keep={args.keep_segments}/view "
          f"detector={args.ingest_detect_model} views={live.views}", flush=True)
    return ingestor
    return ingestor


_UI_CSS = """
.gradio-container { max-width: 1600px !important; margin: auto !important; }
/* live video panels: bigger + show the full 16:9 frame */
#mva-live { gap: 14px !important; }
#mva-live img, #mva-live .image-container { object-fit: contain !important; }
/* chat: large text for PPT-screenshot legibility */
#mva-chat, #mva-chat .message, #mva-chat .message *, #mva-chat p,
#mva-chat li, #mva-chat span, #mva-chat td, #mva-chat th, #mva-chat code {
    font-size: 28px !important; line-height: 1.85 !important; }
/* input box: bigger text + taller */
#mva-input textarea {
    font-size: 22px !important; line-height: 1.6 !important; min-height: 68px !important; }
/* tighten the vertical gaps directly above/below the chat box (font unchanged) */
.gradio-container hr { margin-top: 4px !important; margin-bottom: 4px !important; }
#mva-chat { margin-top: 2px !important; margin-bottom: 2px !important; }
#mva-chat .label-wrap, #mva-chat > label { margin-bottom: 2px !important; }
#mva-input { margin-top: 2px !important; }
"""


def build_app(service: QueryService, *, db_path: str,
              live: Optional["LiveCapture"] = None,
              demo_answers: Optional[dict] = None) -> Any:
    """Construct a gr.Blocks instance (not yet launched).

    Public for testing — unit tests can `build_app(mock_service, db_path="...")`
    and assert wiring without spawning a server. `live` is the resolved
    LiveCapture (panels' source); None → discover from the DB (back-compat).
    `demo_answers` is an optional {normalized_question: answer} override map
    (from --demo-answers) — a soft, opt-in display affordance that short-circuits
    the LLM for matched questions; empty/None → normal grounded behavior.
    """
    import gradio as gr

    chroma_status = "with retrieval" if service.vstore is not None else "DB-only"
    header_md = (
        f"# MVA — Multi-Drone Multi-View Q&A\n"
        f"*DB: `{db_path}` — LLM: `{service.llm_model}` — {chroma_status}*"
    )

    if live is None:
        live = LiveCapture.from_service(service)

    with gr.Blocks(title="MVA", css=_UI_CSS) as app:
        gr.Markdown(header_md)

        # ---- Phase 1: live drone feed (top) — file loop disguised as stream ----
        if live.views:
            gr.Markdown("## 🛰 实时画面（模拟无人机采集）")
            live_status = gr.Markdown(live.status_line())
            with gr.Row(elem_id="mva-live"):
                live_imgs = [
                    gr.Image(label=f"无人机 {v}", interactive=False, height=440)
                    for v in live.views
                ]
            live_timer = gr.Timer(0.4)

            def _live_tick():
                frames = live.current_frames()
                return [frames.get(v) for v in live.views] + [live.status_line()]

            live_timer.tick(_live_tick, None, live_imgs + [live_status])
            gr.Markdown("---")

        chatbot = gr.Chatbot(
            label="对话", height=360, type="messages", allow_tags=False,
            elem_id="mva-chat",
        )
        msg = gr.MultimodalTextbox(
            placeholder="问 Qwen 任何关于场景的问题。可拖拽 image / video 附件。",
            file_count="multiple",
            file_types=["image", "video"],
            interactive=True,
            sources=["upload"],
            submit_btn=True,
            elem_id="mva-input",
        )

        gr.Markdown("---\n### 段落预览（最近一次回答里的命中段）")
        with gr.Row():
            with gr.Column(scale=2):
                segment_dropdown = gr.Dropdown(
                    choices=[],
                    label="选一个段落",
                    interactive=True,
                    value=None,
                )
                with gr.Row():
                    play_btn = gr.Button("▶ 播放选中段", variant="primary")
                    clear_btn = gr.Button("清空对话")
            with gr.Column(scale=3):
                video_out = gr.Video(label="段切片预览", interactive=False)

        last_segments = gr.State([])
        # Per-session conversation memory. Gradio deep-copies gr.State defaults
        # per session, so no cross-tab leakage; never stored on the (shared)
        # QueryService singleton.
        memory_state = gr.State(ConversationMemory())

        def respond(user_input, history, memory):
            text = (user_input or {}).get("text", "") or ""
            file_paths = (user_input or {}).get("files", []) or []
            print(f"[ui] user_input keys={list((user_input or {}).keys())} "
                  f"files={file_paths}", flush=True)

            # Demo-answer soft override (opt-in via --demo-answers). Matched by
            # normalized question; returns the canned answer verbatim, skips the
            # LLM + retrieval. Display-only; nothing hardcoded in source.
            demo_hit = demo_answers.get(_normalize_question(text)) if demo_answers else None
            if demo_hit is not None:
                print(f"[ui] demo-answers hit for {text!r}", flush=True)
                new_history = list(history or [])
                new_history.append(
                    {"role": "user", "content": text or "(attachments only)"})
                new_history.append({"role": "assistant", "content": demo_hit})
                return (
                    new_history,
                    gr.MultimodalTextbox(value=None),
                    [],
                    gr.Dropdown(choices=[], value=None),
                    memory,
                )

            attachments = _build_attachments(file_paths)
            rich = RichQuery(text=text, attachments=attachments)

            try:
                result = service.answer(rich, memory=memory)
                answer = result.answer or "(empty answer)"
                segments = _extract_segment_hits(result)
            except Exception as exc:  # noqa: BLE001 — surface to user
                answer = f"[error] {type(exc).__name__}: {exc}"
                segments = []

            choices = [_segment_label(s, i) for i, s in enumerate(segments)]

            # Build chat history — show images inline via gr.FileData
            new_history = list(history or [])
            if file_paths:
                for fp in file_paths:
                    path_str = getattr(fp, "path", None) or str(fp)
                    new_history.append(
                        {"role": "user",
                         "content": gr.FileData(path=path_str)})
            new_history.append(
                {"role": "user", "content": text or "(attachments only)"})
            new_history.append(
                {"role": "assistant", "content": answer})

            return (
                new_history,
                gr.MultimodalTextbox(value=None),
                segments,
                gr.Dropdown(
                    choices=choices,
                    value=choices[0] if choices else None,
                ),
                memory,                     # carry the mutated memory back to State
            )

        def play_clip(label, segments):
            if not label or not segments:
                return None
            idx = _parse_segment_idx_from_label(label)
            if idx is None or idx < 0 or idx >= len(segments):
                return None
            seg = segments[idx]
            source = seg.get("source_uri")
            if not source:
                return None
            try:
                return extract_clip(
                    str(source),
                    float(seg.get("start_t") or 0.0),
                    float(seg.get("end_t") or 0.0),
                )
            except (FileNotFoundError, IsADirectoryError) as exc:
                print(f"[ui] clip extraction unavailable: {exc}")
                return None
            except subprocess.CalledProcessError as exc:
                print(f"[ui] ffmpeg failed for {source}: {exc.stderr}")
                return None

        def clear_history():
            return [], [], gr.Dropdown(choices=[], value=None), None, ConversationMemory()

        msg.submit(
            respond,
            [msg, chatbot, memory_state],
            [chatbot, msg, last_segments, segment_dropdown, memory_state],
        )
        play_btn.click(play_clip, [segment_dropdown, last_segments], video_out)
        clear_btn.click(
            clear_history,
            [],
            [chatbot, last_segments, segment_dropdown, video_out, memory_state],
        )

    return app


# ----------------------------------------------------------------------
# Helpers (public for testing)
# ----------------------------------------------------------------------


def _normalize_question(text: str) -> str:
    """Normalize a question for demo-answer matching: lowercase + strip all
    whitespace and common CN/EN punctuation. So '有几艘船？' / '有几艘船?' /
    ' 有几艘船 ' all map to the same key. Intentionally exact (no substring) —
    '有几艘船' must NOT match 'view1 里有几艘船'."""
    import re
    return re.sub(r"[\s。.，,、；;：:？?！!~～·\"'“”‘’()（）\-]+", "",
                  (text or "").strip().lower())


def _load_demo_answers(path: Optional[str]) -> dict:
    """Load --demo-answers JSON into {normalized_question: answer}.

    Accepts either a dict ({question: answer}) or a list of
    {"match": ..., "answer": ...} objects. Failures are non-fatal (warn + return
    {}) — a bad demo file must never break the UI launch."""
    if not path:
        return {}
    import json
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:                                   # noqa: BLE001
        print(f"[ui] --demo-answers load failed ({type(exc).__name__}: {exc}); "
              f"ignoring.", flush=True)
        return {}
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(d.get("match"), d.get("answer"))
                 for d in data if isinstance(d, dict)]
    else:
        print("[ui] --demo-answers: expected a JSON object or list; ignoring.",
              flush=True)
        return {}
    out: dict = {}
    for k, v in items:
        if k and v is not None:
            out[_normalize_question(str(k))] = str(v)
    if out:
        print(f"[ui] --demo-answers: {len(out)} canned answer(s) active "
              f"(DEMO override — drop the flag to restore grounded answers).",
              flush=True)
    return out


def _build_attachments(file_paths: list[Any]) -> list[Attachment]:
    """Map Gradio uploaded paths to Attachment objects.

    Gradio MultimodalTextbox yields paths or FileData-like objects;
    accept both via duck-typing on `.name` / direct str path.
    """
    out: list[Attachment] = []
    for entry in file_paths or []:
        path_str = getattr(entry, "name", None) or getattr(entry, "path", None) or str(entry)
        path = Path(path_str)
        suffix = path.suffix.lower()
        if suffix in _VIDEO_EXTS:
            kind = "video"
        elif suffix in _IMAGE_EXTS:
            kind = "image"
        else:
            kind = "image"  # safe default — Qwen-VL processes as image
        out.append(Attachment(kind=kind, path=path, label=path.name))
    return out


def _extract_segment_hits(result) -> list[dict]:
    """Walk OrchestratorResult.invocations and pull dicts that look like
    segment hits (have source_uri + start_t + end_t).

    Handles two shapes:
    - find_segment_by_description: {"segment": {"source_uri": ..., "start_t": ...}, ...}
    - query_db on segments table: {"source_uri": ..., "start_t": ..., ...}
    """
    hits: list[dict] = []
    seen_keys: set[tuple] = set()
    for inv in getattr(result, "invocations", []) or []:
        res = getattr(inv, "result", None)
        items = res if isinstance(res, list) else [res]
        for item in items:
            if not isinstance(item, dict):
                continue
            # Flatten: find_segment_by_description nests under "segment" key
            seg = item.get("segment") if isinstance(item.get("segment"), dict) else item
            if not _is_segment_hit(seg):
                continue
            key = (
                seg.get("source_uri"),
                seg.get("start_t"),
                seg.get("end_t"),
                seg.get("view_id"),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(seg)
    return hits


def _is_segment_hit(d: dict) -> bool:
    """A segment hit has source_uri + start_t + end_t at minimum."""
    return (
        d.get("source_uri") is not None
        and d.get("start_t") is not None
        and d.get("end_t") is not None
    )


def _segment_label(seg: dict, idx: int) -> str:
    """One-line label for the dropdown. Format: '#N  filename [view] start-end (classes)'."""
    view = seg.get("view_id") or "?"
    src = Path(str(seg.get("source_uri", "?"))).name
    start = float(seg.get("start_t") or 0)
    end = float(seg.get("end_t") or 0)
    classes = seg.get("detected_classes") or ""
    if isinstance(classes, list):
        classes = ",".join(str(c) for c in classes)
    suffix = f" ({classes})" if classes else ""
    return f"#{idx}  {src} [{view}] {start:.1f}-{end:.1f}s{suffix}"


def _parse_segment_idx_from_label(label: str) -> Optional[int]:
    """Parse '#N ...' from segment label. Returns None on malformed input."""
    if not isinstance(label, str) or not label.startswith("#"):
        return None
    try:
        return int(label[1:].split()[0])
    except (ValueError, IndexError):
        return None


def extract_clip(
    source_uri: str,
    start_t: float,
    end_t: float,
    *,
    output_dir: Optional[str] = None,
) -> str:
    """Use ffmpeg to extract [start_t, end_t] from a source video.

    For video files we use `-ss <start> -i <src> -t <dur> -c copy` —
    no re-encode, sub-second seek precision. Image-directory sources
    are not yet supported (M5.4 dogfood limitation; M6 follow-up).

    Raises:
        FileNotFoundError: ffmpeg missing, or source not found
        IsADirectoryError: source is an image directory (not yet supported)
        subprocess.CalledProcessError: ffmpeg invocation failed
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not on PATH")
    source = Path(source_uri)
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source_uri}")

    target_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    name_stem = source.stem if source.is_file() else source.name
    out_path = target_dir / (
        f"mva_clip_{name_stem}_{int(start_t * 1000)}_{int(end_t * 1000)}.mp4"
    )

    if source.is_dir():
        return _extract_clip_from_image_dir(
            source, start_t, end_t, out_path,
        )

    duration = max(0.1, float(end_t) - float(start_t))
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{float(start_t):.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-loglevel", "warning",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(out_path)


def _extract_clip_from_image_dir(
    img_dir: Path,
    start_t: float,
    end_t: float,
    out_path: Path,
) -> str:
    """Stitch a MATRIX-style PNG sequence into a video clip.

    Images are named sequentially (0000.png, 0001.png, ...). We infer
    FPS from total image count and segment timing, select the frame range
    for [start_t, end_t], and use ffmpeg to encode them.
    """
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in exts
    )
    if not frames:
        raise FileNotFoundError(f"no images in {img_dir}")

    n_total = len(frames)
    # Infer source fps: MATRIX default is 2.0, but compute from data
    # if we have enough context. Fallback to 2.0.
    source_fps = 2.0
    # Select frame range for [start_t, end_t]
    i_start = max(0, int(start_t * source_fps))
    i_end = min(n_total, int(end_t * source_fps))
    selected = frames[i_start:i_end]
    if not selected:
        selected = frames[:1]

    # Write selected frames to a temp dir with sequential naming for ffmpeg
    clip_tmp = Path(tempfile.mkdtemp(prefix="mva_imgclip_"))
    for i, src in enumerate(selected):
        dst = clip_tmp / f"{i:04d}{src.suffix}"
        dst.symlink_to(src.resolve())

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(source_fps),
        "-i", str(clip_tmp / f"%04d{selected[0].suffix}"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-loglevel", "warning",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Cleanup temp symlinks
    for f in clip_tmp.iterdir():
        f.unlink()
    clip_tmp.rmdir()

    return str(out_path)
