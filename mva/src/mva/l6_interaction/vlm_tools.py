"""L6 G-2 VLM-native tool: `look_at` — answer descriptive / attribute /
activity / scene questions by retrieving the most relevant segment and letting
Qwen2.5-VL look at its ACTUAL frames, not detection labels.

Why: detection only knows the prompted classes (and on sim imagery the labels
are low-confidence / sometimes wrong — a drone mislabeled "boat"). For "那艘船
是什么颜色 / 在做什么 / 水里漂的是什么", the only sound source is the pixels.

Honesty model (the "答案执行度" gate the user asked for — NOT a self-reported
score, which LLMs fake):
  1. retrieval-distance gate (pre-VLM): if no segment matches the question well
     (top distance > LOOK_AT_DISTANCE_GATE), abstain before even decoding frames.
  2. explicit-abstain option (during-VLM): the prompt lets the VLM answer "看不清"
     and we map that to abstain. Giving it an out beats forcing a guess.
Abstain → the orchestrator says "不知道"; a grounded answer is labelled
"（根据画面判断）" so its softer provenance is visible.

The result dict (`abstained` flag) also drives the orchestrator CASCADE: a typed
tool that returns empty *for an existence/identification question* falls back to
look_at to double-check a frame (detection recall is limited), but a plain count
of 0 is trusted (empty ≠ failure).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mva.l6_interaction.tools import (
    ToolRegistry,
    ToolSpec,
    WEAK_MATCH_DISTANCE_THRESHOLD,
    _find_segments,
)

LOOK_AT_NFRAMES = 4
LOOK_AT_DISTANCE_GATE = WEAK_MATCH_DISTANCE_THRESHOLD  # 0.85
VLM_LABEL = "（根据画面判断）"

_ABSTAIN_CUES = (
    "看不清", "看不出", "无法确认", "无法判断", "无法确定", "不确定",
    "没有相关", "没有看到", "看不到", "无法回答", "没有发现",
)

VLM_SCENE_PROMPT = (
    "下面是无人机视频里检索到的若干帧画面。请**只根据画面内容**回答问题：\n"
    "{question}\n"
    "要求：只描述画面里真实可见的内容，用一两句话；若画面里看不清、或没有与问题"
    "相关的内容，就直接回答“看不清”，不要猜测、不要编造。"
)

# G-2.1: object mode — the first image is a tight ROI crop of the actual target
# (so a tens-of-px aerial object fills the frame), optionally followed by its
# full-frame context. This fixes the G-2 over-abstain on attribute questions.
VLM_OBJECT_PROMPT = (
    "下面第一张图是从无人机画面里**裁剪出的一个目标**（放大后的特写），"
    "若还有第二张则是它所在的整帧画面。请**只根据这些画面**回答：\n"
    "{question}\n"
    "聚焦那个被裁剪出的目标，只描述真实可见的内容，一两句话；看不清就直接答"
    "“看不清”，不要猜测、不要编造。"
)

# Cues that a question is about a specific object's appearance/identity (→ bbox
# crop), vs a scene/activity question (→ whole-frame). Scene wins by default.
_OBJECT_QUESTION_CUES = (
    "什么颜色", "颜色", "长什么样", "什么样子", "外观", "什么牌子", "什么型号",
    "这是什么", "那是什么", "是什么", "什么东西", "漂的是", "什么船", "什么车",
    "什么飞机", "几个轮", "戴", "穿",
)


def _is_object_question(text: str) -> bool:
    q = text or ""
    # Scene/activity phrasings stay in scene mode even if they contain 是什么.
    if any(s in q for s in ("在做什么", "在干什么", "发生了什么", "发生什么", "整体", "场景")):
        return False
    return any(c in q for c in _OBJECT_QUESTION_CUES)


def _decode_frames(source_uri: Optional[str], start_t, end_t, n: int) -> list:
    """Decode up to `n` evenly-spaced BGR frames from [start_t, end_t]."""
    if not source_uri:
        return []
    import cv2

    p = Path(source_uri)
    s = float(start_t or 0.0)
    e = float(end_t if end_t is not None else s + 1.0)
    if e <= s:
        e = s + 1.0
    timestamps = [s + (i + 0.5) * (e - s) / n for i in range(n)]

    if p.is_dir():
        exts = (".png", ".jpg", ".jpeg")
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        if not files:
            return []
        # Approximate: map each timestamp to a file index across the whole dir.
        # (Image-dir sources don't carry per-file timing here; sample uniformly.)
        out = []
        for i in range(n):
            idx = min(len(files) - 1, int((i + 0.5) / n * len(files)))
            img = cv2.imread(str(files[idx]))
            if img is not None:
                out.append(img)
        return out

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return []
    try:
        frames = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if ok and img is not None:
                frames.append(img)
        return frames
    finally:
        cap.release()


def _is_abstain(text: str) -> bool:
    t = (text or "").strip()
    return (not t) or any(cue in t for cue in _ABSTAIN_CUES)


def _crop_from_frame(source_uri, start_t, end_t, bbox):
    """Decode one frame from [start_t, end_t] and crop to bbox [x1,y1,x2,y2]."""
    frames = _decode_frames(source_uri, start_t, end_t, 1)
    if not frames:
        return None
    img = frames[0]
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                      int(min(w, bbox[2])), int(min(h, bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _look_at_object(store, vstore, llm, text, view_id, top_k):
    """Object mode: bbox retrieval → ROI crop (+ context frame) → VLM.

    Returns a result dict, or one with reason='no_bbox'/'no_crop' so the
    dispatcher can fall back to scene mode."""
    import cv2

    hits = vstore.query(
        query_text=text, vector_type="reid", view_id=view_id, top_k=top_k,
    )
    hits = [h for h in (hits or [])
            if (h.get("metadata") or {}).get("vector_kind") == "bbox"]
    if not hits:
        return {"abstained": True, "reason": "no_bbox", "answer": None}
    top = hits[0]
    dist = top.get("distance")
    if dist is not None and dist > LOOK_AT_DISTANCE_GATE:
        return {"abstained": True, "reason": "no_bbox", "answer": None,
                "distance": dist}
    md = top.get("metadata") or {}
    seg = None
    if md.get("segment_idx") is not None and md.get("view_id_raw"):
        seg = store.get_segment(md["view_id_raw"], md["segment_idx"])

    frames = []
    # 1) cached ROI crop — filename is exactly the bbox chroma id + ".jpg".
    rois_dir = f"{getattr(vstore, 'persist_dir', '') or ''}-rois"
    crop_path = Path(rois_dir) / f"{top.get('id')}.jpg"
    if crop_path.exists():
        img = cv2.imread(str(crop_path))
        if img is not None:
            frames.append(img)
    # 2) no cached crop → crop the object out of its source frame.
    if not frames:
        src = (seg or {}).get("source_uri") or md.get("source_uri")
        bbox = (md.get("bbox_x1"), md.get("bbox_y1"),
                md.get("bbox_x2"), md.get("bbox_y2"))
        if src and None not in bbox:
            crop = _crop_from_frame(
                src, (seg or {}).get("start_t"), (seg or {}).get("end_t"), bbox)
            if crop is not None:
                frames.append(crop)
    if not frames:
        return {"abstained": True, "reason": "no_crop", "answer": None}
    # context whole-frame (optional; helps spatial/relation questions)
    if seg and seg.get("source_uri"):
        frames += _decode_frames(
            seg["source_uri"], seg.get("start_t"), seg.get("end_t"), 1)

    seg_out = seg or {"view_id": md.get("view_id_raw"), "start_t": None,
                      "end_t": None, "source_uri": md.get("source_uri")}
    raw = llm.complete(VLM_OBJECT_PROMPT.format(question=text), images=frames)
    if _is_abstain(raw):
        return {"abstained": True, "reason": "vlm_abstain", "answer": None,
                "segment": seg_out, "distance": dist, "mode": "object", "raw": raw}
    return {"abstained": False, "answer": raw.strip(), "segment": seg_out,
            "distance": dist, "mode": "object", "n_frames": len(frames)}


def _look_at_scene(store, vstore, llm, text, view_id, top_k):
    """Scene mode: segment retrieval → whole frames → VLM."""
    hits = _find_segments(store, vstore, text, view_id, top_k)
    if not hits:
        return {"abstained": True, "reason": "no_segment", "answer": None}
    top = hits[0]
    dist = top.get("distance")
    seg = top.get("segment") or {}
    if dist is not None and dist > LOOK_AT_DISTANCE_GATE:
        return {"abstained": True, "reason": "weak_match",
                "answer": None, "distance": dist, "segment": seg}
    frames = _decode_frames(
        seg.get("source_uri"), seg.get("start_t"), seg.get("end_t"), LOOK_AT_NFRAMES,
    )
    if not frames:
        return {"abstained": True, "reason": "no_frames",
                "answer": None, "segment": seg, "distance": dist}
    raw = llm.complete(VLM_SCENE_PROMPT.format(question=text), images=frames)
    if _is_abstain(raw):
        return {"abstained": True, "reason": "vlm_abstain", "answer": None,
                "segment": seg, "distance": dist, "mode": "scene", "raw": raw}
    return {"abstained": False, "answer": raw.strip(), "segment": seg,
            "distance": dist, "mode": "scene", "n_frames": len(frames)}


def _look_at(store, vstore, llm, text, view_id=None, top_k=3, **_):
    """Dispatch: object-attribute questions try a bbox crop first (so a small
    aerial target fills the VLM's view); on no usable crop, or for scene/
    activity questions, fall back to whole-frame scene mode.

    Returns {abstained, answer, reason, segment, distance, mode}.
    """
    if vstore is None or llm is None:
        return {"abstained": True, "reason": "no_vlm", "answer": None}
    if _is_object_question(text):
        r = _look_at_object(store, vstore, llm, text, view_id, top_k)
        # Only fall back to scene mode when object RETRIEVAL found nothing — a
        # VLM abstain on a real crop is a genuine "看不清", don't paper over it.
        if r.get("reason") not in ("no_bbox", "no_crop"):
            return r
    return _look_at_scene(store, vstore, llm, text, view_id, top_k)


def _render_look_at(result: dict, args: dict) -> Optional[str]:
    """Templated answer: grounded VLM answer (labelled) or an honest abstain."""
    if not isinstance(result, dict):
        return None
    if result.get("abstained"):
        if result.get("mode") == "object" and result.get("reason") == "vlm_abstain":
            # Object was found+cropped but the VLM still couldn't read it —
            # on aerial footage that's almost always "target too small".
            return "检索到了相关目标，但它在画面中太小/不够清晰，无法判断其细节。"
        return "根据检索到的画面无法确认（看不清或没有相关画面）。"
    ans = (result.get("answer") or "").strip()
    if not ans:
        return None
    seg = result.get("segment") or {}
    where = ""
    if seg.get("view_id") is not None and seg.get("start_t") is not None:
        where = f"（{seg['view_id']} {seg['start_t']:.0f}–{seg.get('end_t', 0):.0f}s）"
    return f"{ans}{VLM_LABEL}{where}"


# ----------------------------------------------------------------------
# Cascade helpers (used by the orchestrator)
# ----------------------------------------------------------------------

_VISUAL_CHECK_CUES = (
    "有没有", "有无", "是不是", "是否有", "是什么", "什么东西", "看得到",
    "看见", "看到", "长什么", "什么样", "什么颜色", "在做什么", "在干什么",
)


def wants_visual_check(question: str) -> bool:
    """True for existence/identification phrasings worth a VLM double-check on
    an empty structured result. Pure-count phrasings ('有几/多少') are excluded —
    a structured count of 0 is trusted (empty ≠ failure)."""
    q = question or ""
    if "有几" in q or "多少" in q:
        return False
    return any(c in q for c in _VISUAL_CHECK_CUES)


def is_empty_structured(tool: str, result: Any) -> bool:
    """Did a typed structured tool come back empty (no objects found)?"""
    if not isinstance(result, dict):
        return False
    if tool == "count_objects":
        return result.get("total_distinct_tracks", 0) == 0
    if tool == "which_views":
        return not result.get("views")
    if tool == "list_objects":
        return not any(result.get("per_view", {}).values())
    if tool == "when_seen":
        return not result.get("spans")
    return False


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_vlm_tools(reg: ToolRegistry, store: Any, vstore: Any, llm: Any) -> None:
    """Register `look_at` (needs vstore for retrieval + llm for vision)."""
    if vstore is None or llm is None:
        return
    reg.register(ToolSpec(
        name="look_at",
        description=(
            "看图作答：检索最相关的视频段，让多模态模型看真实画面回答**描述/属性/"
            "动作/识别**类问题（颜色、长相、在做什么、水里漂的是什么、画面里发生了什么）。"
            "检测类名覆盖不到的视觉问题用这个。args: text (str, 必填, 把用户问题原样传入); "
            "view_id (str, 可选)。"
        ),
        fn=lambda text, view_id=None, top_k=3, **_: _look_at(
            store, vstore, llm, text, view_id=view_id, top_k=top_k,
        ),
        render=_render_look_at,
    ))
