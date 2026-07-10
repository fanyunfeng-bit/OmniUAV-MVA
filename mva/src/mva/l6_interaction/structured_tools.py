"""L6 typed structured tools (G-1) — grounded answers without LLM-authored SQL.

Why this module exists: the 7B planner kept writing SQL against columns that
don't exist (e.g. `tracklets_*.detected_classes` — class lives inside the
`bboxes` JSON, not a column), the query errored, and the answer LLM then
*fabricated* counts. These tools move the correct, JSON-aware query INTO code
the model never sees. The planner only picks a tool by name and fills typed
slots (class / view / time); it cannot get a column wrong because it writes no
SQL.

Each tool ships a `render(result, args) -> str | None` that templates the
simple-case answer so the orchestrator can skip the answer LLM entirely
(faster + un-hallucinatable). `render` returns None when the result needs LLM
composition.

**Counting semantic** (load-bearing — tracking is segment-local, so the same
physical object across N segments is N tracklet rows; cross-segment identity
merging is M5 stretch). For each class we report two honest numbers:
  - `peak_per_segment` — max simultaneous tracks of the class in any one
    segment. This is the headline "how many X are there" (least inflated).
  - `distinct_tracks`   — total tracklet rows of the class (sightings; inflated
    by persistence across segments).
Never invent a single precise count we cannot support.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Optional

from mva.l6_interaction.tools import ToolRegistry, ToolSpec


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _list_views(store: Any) -> list[str]:
    """Distinct original view_ids, in stable order."""
    rows = store.execute_readonly("SELECT DISTINCT view_id FROM segments")
    return sorted(r["view_id"] for r in rows if r.get("view_id"))


def _track_class(bboxes: list) -> Optional[str]:
    """Dominant (majority) class across a track's bbox entries.

    bbox entry shape: [t, x1, y1, x2, y2, class_name, confidence].
    """
    cc: Counter = Counter()
    for b in bboxes or []:
        if isinstance(b, (list, tuple)) and len(b) >= 6 and b[5] is not None:
            cc[str(b[5])] += 1
    return cc.most_common(1)[0][0] if cc else None


def _collect(
    store: Any,
    view_ids: list[str],
    t_start: Optional[float],
    t_end: Optional[float],
) -> dict[str, dict]:
    """Per view: distinct tracks + per-frame detection counts.

    Returns {view_id: {
        "tracks":  [(segment_idx, dominant_class), ...],   # track-level
        "frames":  {frame_t: Counter(class -> n)},         # per-frame detections
    }}.

    Per-frame is the honest basis for "how many X at once": within a segment
    the segment-local IoU tracker fragments one moving object into several
    tracks across the K sampled frames, so a track count over-states presence.
    A single frame's detections do not (YOLO emits one box per visible object),
    so `max over frames` of a class's per-frame count = peak simultaneous.
    """
    out: dict[str, dict] = {}
    for v in view_ids:
        tracks: list[tuple] = []
        frames: dict[float, Counter] = defaultdict(Counter)
        for tk in store.query_tracklets(v, t_start=t_start, t_end=t_end):
            cls = _track_class(tk.get("bboxes"))
            if cls is not None:
                tracks.append((tk.get("segment_idx"), cls))
            for b in tk.get("bboxes") or []:
                if isinstance(b, (list, tuple)) and len(b) >= 6 and b[5] is not None:
                    ft = round(float(b[0]), 2)
                    if t_start is not None and ft < t_start:
                        continue
                    if t_end is not None and ft > t_end:
                        continue
                    frames[ft][str(b[5])] += 1
        out[v] = {"tracks": tracks, "frames": frames}
    return out


def _peak_and_distinct(view_data: dict, cls: str) -> tuple[int, int]:
    """(peak_per_frame, distinct_tracks) for `cls` in one view."""
    peak = max((fr.get(cls, 0) for fr in view_data["frames"].values()), default=0)
    distinct = sum(1 for _, c in view_data["tracks"] if c == cls)
    return peak, distinct


def _norm_views(store: Any, view_id: Optional[str]) -> list[str]:
    return [view_id] if view_id else _list_views(store)


# ----------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------


def _count_objects(store, class_name, view_id=None, t_start=None, t_end=None, **_):
    cls = str(class_name).strip().lower()
    views = _norm_views(store, view_id)
    collected = _collect(store, views, t_start, t_end)
    per_view = {}
    for v in views:
        peak, distinct = _peak_and_distinct(collected[v], cls)
        per_view[v] = {"peak_per_frame": peak, "distinct_tracks": distinct}
    return {
        "class": cls,
        "time_range": [t_start, t_end] if (t_start or t_end) else None,
        "per_view": per_view,
        "total_distinct_tracks": sum(d["distinct_tracks"] for d in per_view.values()),
        "max_peak_per_frame": max((d["peak_per_frame"] for d in per_view.values()), default=0),
    }


def _render_count(result: dict, args: dict) -> Optional[str]:
    cls = result["class"]
    per_view = result["per_view"]
    tr = result.get("time_range")
    span = f"（{tr[0]}–{tr[1]}s）" if tr else ""
    seen = {v: d for v, d in per_view.items() if d["distinct_tracks"] > 0}
    if not seen:
        return f"没有检测到 {cls}{span}。（注：检测召回有限，可能存在漏检）"
    parts = [
        f"{v} 最多同时 {d['peak_per_frame']} 个（累计 {d['distinct_tracks']} 条轨迹）"
        for v, d in per_view.items() if d["distinct_tracks"] > 0
    ]
    return f"{cls}{span}：" + "；".join(parts) + "。"


def _list_objects(store, view_id=None, t_start=None, t_end=None, **_):
    views = _norm_views(store, view_id)
    collected = _collect(store, views, t_start, t_end)
    per_view = {}
    for v in views:
        classes: set = set(c for _, c in collected[v]["tracks"])
        for fr in collected[v]["frames"].values():
            classes.update(fr.keys())
        peaks = {
            c: max((fr.get(c, 0) for fr in collected[v]["frames"].values()), default=0)
            for c in classes
        }
        per_view[v] = dict(sorted(peaks.items(), key=lambda kv: -kv[1]))
    return {"per_view": per_view, "time_range": [t_start, t_end] if (t_start or t_end) else None}


def _render_list(result: dict, args: dict) -> Optional[str]:
    per_view = result["per_view"]
    nonempty = {v: c for v, c in per_view.items() if c}
    if not nonempty:
        return "未检测到任何目标。"
    lines = []
    for v, classes in per_view.items():
        if classes:
            inv = "、".join(f"{cls}×{n}" for cls, n in classes.items())
            lines.append(f"{v}：{inv}")
        else:
            lines.append(f"{v}：无目标")
    return "各视角目标（数字=最多同时出现数）——" + "；".join(lines) + "。"


def _which_views(store, class_name, **_):
    cls = str(class_name).strip().lower()
    views = _list_views(store)
    collected = _collect(store, views, None, None)
    hit = [v for v in views if any(c == cls for _, c in collected[v]["tracks"])]
    return {"class": cls, "views": hit}


def _render_which_views(result: dict, args: dict) -> Optional[str]:
    cls, views = result["class"], result["views"]
    if not views:
        return f"没有任何视角检测到 {cls}。"
    return f"{cls} 出现在：{'、'.join(views)}。"


def _when_seen(store, class_name, view_id=None, **_):
    cls = str(class_name).strip().lower()
    views = _norm_views(store, view_id)
    spans = []
    for v in views:
        for tk in store.query_tracklets(v):
            if _track_class(tk.get("bboxes")) == cls:
                spans.append({
                    "view_id": v,
                    "segment_idx": tk.get("segment_idx"),
                    "t_start": tk.get("t_start"),
                    "t_end": tk.get("t_end"),
                })
    spans.sort(key=lambda s: (s["view_id"], s["t_start"] if s["t_start"] is not None else 0))
    return {"class": cls, "spans": spans}


def _render_when_seen(result: dict, args: dict) -> Optional[str]:
    cls, spans = result["class"], result["spans"]
    if not spans:
        return f"没有检测到 {cls}。"
    # Group by view; list up to 6 time windows per view, summarize the rest.
    by_view: dict[str, list] = defaultdict(list)
    for s in spans:
        by_view[s["view_id"]].append(s)
    lines = []
    for v, ss in by_view.items():
        uniq = list(dict.fromkeys(f"{s['t_start']:.0f}–{s['t_end']:.0f}s" for s in ss))
        wins = uniq[:6]
        more = f" 等共 {len(uniq)} 个时间段" if len(uniq) > 6 else ""
        lines.append(f"{v}：{'、'.join(wins)}{more}")
    return f"{cls} 出现时间——" + "；".join(lines) + "。"


def _objects_at_time(store, t, **_):
    t = float(t)
    views = _list_views(store)
    collected = _collect(store, views, None, None)
    per_view = {}
    for v in views:
        frames = collected[v]["frames"]
        if frames:
            nearest = min(frames.keys(), key=lambda ft: abs(ft - t))
            per_view[v] = dict(frames[nearest].most_common()) if abs(nearest - t) <= 1.5 else {}
        else:
            per_view[v] = {}
    return {"t": t, "per_view": per_view}


def _render_objects_at_time(result: dict, args: dict) -> Optional[str]:
    t, per_view = result["t"], result["per_view"]
    parts = []
    for v, classes in per_view.items():
        if classes:
            parts.append(f"{v} 有 " + "、".join(f"{c}×{n}" for c, n in classes.items()))
        else:
            parts.append(f"{v} 无目标")
    return f"第 {t:.0f} 秒：" + "；".join(parts) + "。"


def _cross_view_matches(store, class_name=None, **_):
    # Build tracklet_id -> dominant class across all views.
    tid2cls: dict[str, str] = {}
    for v in _list_views(store):
        for tk in store.query_tracklets(v):
            c = _track_class(tk.get("bboxes"))
            if c is not None:
                tid2cls[tk.get("tracklet_id")] = c
    target = str(class_name).strip().lower() if class_name else None
    links = []
    for link in store.query_cross_view_links():
        obs = getattr(link, "view_observations", None) or []
        pairs = []
        for o in obs:
            # ViewObservation pydantic OR ["view","tracklet"] list
            vid = getattr(o, "view_id", None)
            tid = getattr(o, "tracklet_id", None) or getattr(o, "observation_id", None)
            if vid is None and isinstance(o, (list, tuple)) and len(o) >= 2:
                vid, tid = o[0], o[1]
            pairs.append((vid, tid2cls.get(tid)))
        classes = [c for _, c in pairs if c]
        consistent = len(set(classes)) == 1 if classes else None
        link_cls = classes[0] if (consistent and classes) else None
        if target and link_cls != target:
            continue
        links.append({
            "confidence": getattr(link, "confidence", None),
            "views": [vid for vid, _ in pairs],
            "classes": classes,
            "class_consistent": consistent,
        })
    return {"class": target, "n_links": len(links), "links": links}


def _render_cross_view(result: dict, args: dict) -> Optional[str]:
    n, links = result["n_links"], result["links"]
    cls = result.get("class")
    if n == 0:
        if cls:
            return f"没有找到两个视角共享的 {cls}。"
        return "没有找到跨视角的同一目标。"
    consistent = sum(1 for l in links if l.get("class_consistent"))
    by_cls = Counter(l["classes"][0] for l in links if l.get("class_consistent") and l["classes"])
    head = f"共 {n} 对跨视角同一目标" + (f"（{cls}）" if cls else "")
    if by_cls and not cls:
        head += "，其中 " + "、".join(f"{c} {k} 对" for c, k in by_cls.most_common())
    head += f"（类别一致 {consistent}/{n}）。"
    return head


def _scene_stats(store, **_):
    views = _list_views(store)
    segs = store.query_segments()
    per_view_segs = Counter(s["view_id"] for s in segs)
    times = [s["end_t"] for s in segs if s.get("end_t") is not None]
    n_links = len(store.query_cross_view_links())
    return {
        "views": views,
        "segments_per_view": dict(per_view_segs),
        "coverage_sec": max(times) if times else 0.0,
        "cross_view_links": n_links,
    }


def _render_scene_stats(result: dict, args: dict) -> Optional[str]:
    views = result["views"]
    spv = result["segments_per_view"]
    seg_desc = "、".join(f"{v} {spv.get(v, 0)} 段" for v in views)
    return (
        f"{len(views)} 个视角（{'、'.join(views)}），{seg_desc}，"
        f"覆盖约 0–{result['coverage_sec']:.0f}s，跨视角链路 {result['cross_view_links']} 条。"
    )


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

_TOOLS = [
    ToolSpec(
        name="count_objects",
        description=(
            "数某一类目标有多少。问『有几X / 多少X / 有没有X』用这个。"
            "args: class_name (str, 必填, 用检测类名如 boat/ship/uav/drone); "
            "view_id (str, 可选, 省略=所有视角); t_start/t_end (float, 可选时间窗)。"
        ),
        fn=None,  # bound below
        render=_render_count,
    ),
    ToolSpec(
        name="list_objects",
        description=(
            "列出某视角/时间窗里有哪些类别的目标及各自数量。问『有哪些目标 / 都看到了什么』用这个。"
            "args: view_id (str, 可选); t_start/t_end (float, 可选)。"
        ),
        fn=None, render=_render_list,
    ),
    ToolSpec(
        name="which_views",
        description=(
            "某类目标出现在哪些视角。问『X在哪个视角 / 哪架无人机看到了X』用这个。"
            "args: class_name (str, 必填)。"
        ),
        fn=None, render=_render_which_views,
    ),
    ToolSpec(
        name="when_seen",
        description=(
            "某类目标在什么时间段出现。问『X什么时候出现 / X出现在哪些时间』用这个。"
            "args: class_name (str, 必填); view_id (str, 可选)。"
        ),
        fn=None, render=_render_when_seen,
    ),
    ToolSpec(
        name="objects_at_time",
        description=(
            "查某个时间点各视角有哪些目标。问『第N秒有什么』用这个。"
            "args: t (float, 必填, 秒)。"
        ),
        fn=None, render=_render_objects_at_time,
    ),
    ToolSpec(
        name="cross_view_matches",
        description=(
            "查两个视角看到的是不是同一目标（跨视角关联）。问『两个视角是同一个X吗 / "
            "哪些目标被两机同时看到』用这个。args: class_name (str, 可选, 只看某类)。"
        ),
        fn=None, render=_render_cross_view,
    ),
    ToolSpec(
        name="scene_stats",
        description=(
            "场景总体统计：几个视角、各多少段、覆盖时长、跨视角链路数。"
            "问『有几个视角 / 覆盖多长时间 / 总体情况(统计)』用这个。args: 无。"
        ),
        fn=None, render=_render_scene_stats,
    ),
]

_IMPLS = {
    "count_objects": _count_objects,
    "list_objects": _list_objects,
    "which_views": _which_views,
    "when_seen": _when_seen,
    "objects_at_time": _objects_at_time,
    "cross_view_matches": _cross_view_matches,
    "scene_stats": _scene_stats,
}


def register_structured_tools(reg: ToolRegistry, store: Any) -> None:
    """Register the G-1 typed structured tools, each bound to `store`."""
    for spec in _TOOLS:
        impl = _IMPLS[spec.name]
        reg.register(ToolSpec(
            name=spec.name,
            description=spec.description,
            fn=(lambda _impl: (lambda **kw: _impl(store, **kw)))(impl),
            render=spec.render,
        ))
