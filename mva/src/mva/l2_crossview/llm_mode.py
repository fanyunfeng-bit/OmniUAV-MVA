"""LLMCrossViewLinker — §3.4 #2 promoted to real implementation in M4.1.

Triggered when geometric / appearance modes return low-confidence
(< 0.5) candidates, or when L7 / Mode A user confirmation requires
multimodal disambiguation. Asks Qwen2.5-VL "are these two crops the
same physical object?" and returns a CrossViewLink with
`created_by="llm"`.

**ROI hybrid loader (M4.1 design lock 2026-05-23)** — see PLAN §6.2 M4.1:

    def _load_roi(obs):
        if obs.roi_uri and Path(obs.roi_uri).exists():
            return cv2.imread(obs.roi_uri)        # (b) cache hit
        return _decode_and_crop(obs.source_uri, obs.frame_idx, obs.bbox)  # (a) fallback

ingest's `--cache-rois` flag (M4.1) populates `obs.roi_uri`; runs without
it (or chroma DBs imported from elsewhere) fall through to delayed
decode automatically. Both paths feed the same crop pixels to the LLM —
the trade-off is only WHEN they get materialized.

Construct with `llm_client` (LLMClient) to actually call the model. The
zero-arg ctor is preserved for the contract test fixture — no client =
returns [] from every `link()` call (degenerate path, never raises).

Candidate generation: pairwise across distinct views within each
`(class_name, segment_idx)` bucket — only "plausible" pairs (same class)
get asked. Per-pair LLM cost is ~1-2s on Qwen2.5-VL-7B FP16, so this is
expected to be a fallback path gated by M4.3, not a default for every
ingest run.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from mva.contracts import CrossViewLink, ViewObservation, make_link_id


_LINK_PROMPT_TEMPLATE = """你是跨视角目标关联助手。下面给出两张 ROI 图像，分别来自两个不同的视角：

view_A: {view_a} (class={class_a}, tracklet={tk_a})
view_B: {view_b} (class={class_b}, tracklet={tk_b})

判断这两张图里是否是**同一物理目标**（不是"同类的两个不同个体"）。
严格只返回一行 JSON，键固定，不要任何其他文字：
{{"same_object": true|false, "confidence": <0.0-1.0>, "reasoning": "<一句话>"}}
"""


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


# ----------------------------------------------------------------------
# Default ROI loader — implements the M4.1 (b)+(a) hybrid
# ----------------------------------------------------------------------


def default_roi_loader(obs: ViewObservation) -> Optional[np.ndarray]:
    """Hybrid ROI loader per M4.1 design lock.

    Cache hit path (b): `obs.roi_uri` exists → cv2.imread (~5-20ms).
    Delayed decode (a): `obs.source_uri` + `obs.frame_idx` present →
    cv2.VideoCapture seek + decode + crop (50-200ms mp4 / 5-20ms PNG).
    Neither → return None so the caller can skip the LLM call.

    Returns the crop as a BGR numpy array (OpenCV convention), or None
    if no source is reachable / decode fails.
    """
    try:
        import cv2  # type: ignore
    except ImportError:                            # pragma: no cover
        return None

    if obs.roi_uri:
        p = Path(obs.roi_uri)
        if p.exists():
            crop = cv2.imread(str(p))
            if crop is not None and crop.size > 0:
                return crop
        # Fall through to delayed decode if cache file missing / unreadable

    if obs.source_uri is None or obs.frame_idx is None:
        return None

    source = Path(obs.source_uri)
    frame: Optional[np.ndarray] = None

    if source.is_dir():
        # MATRIX-style PNG sequence — sorted filename indexing
        candidates = sorted(source.iterdir())
        if 0 <= obs.frame_idx < len(candidates):
            frame = cv2.imread(str(candidates[obs.frame_idx]))
    elif source.is_file():
        # mp4 / video file — seek by frame index
        cap = cv2.VideoCapture(str(source))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(obs.frame_idx))
            ok, frame = cap.read()
            if not ok:
                frame = None
        finally:
            cap.release()

    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = obs.bbox
    # bbox is normalized [0,1]; map back to pixel coords
    ix1 = max(0, min(w, int(round(x1 * w))))
    iy1 = max(0, min(h, int(round(y1 * h))))
    ix2 = max(0, min(w, int(round(x2 * w))))
    iy2 = max(0, min(h, int(round(y2 * h))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = frame[iy1:iy2, ix1:ix2]
    return crop if crop.size > 0 else None


# ----------------------------------------------------------------------
# Linker
# ----------------------------------------------------------------------


class LLMCrossViewLinker:
    """LLM-mode cross-view association — Qwen2.5-VL same-object judge.

    Parameters
    ----------
    llm_client : LLMClient | None
        Qwen2.5-VL wrapper. When None, every `link()` returns `[]` so the
        zero-arg contract fixture passes (M0 stub behavior preserved).
    confidence_threshold : float
        Minimum LLM-reported confidence to emit a CrossViewLink. Below
        this we drop the candidate (treat as "LLM judged not same").
        Default 0.5 — matches PLAN §3.5 confidence gate semantics.
    max_retries : int
        Retries on malformed JSON. PLAN §3.5 L4: one retry with stricter
        prompt, then degrade to "no link" (do NOT hallucinate a link).
    roi_loader : Callable[[ViewObservation], Optional[np.ndarray]] | None
        Strategy for loading ROI pixels. Default = `default_roi_loader`
        (the hybrid above). Tests inject a fake loader; future callers
        can pass an alternate scheme (e.g. in-memory frame cache).
    max_new_tokens : int
        Generation budget per pair. Short by design — the response is
        one-line JSON.
    """

    def __init__(
        self,
        llm_client: Any = None,
        *,
        confidence_threshold: float = 0.5,
        max_retries: int = 1,
        roi_loader: Optional[Callable[[ViewObservation], Optional[np.ndarray]]] = None,
        max_new_tokens: int = 96,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        self.llm_client = llm_client
        self.confidence_threshold = float(confidence_threshold)
        self.max_retries = int(max_retries)
        self.roi_loader = roi_loader or default_roi_loader
        self.max_new_tokens = int(max_new_tokens)

    # ------------------------------------------------------------------
    # Public link()
    # ------------------------------------------------------------------

    def link(
        self, observations: Iterable[ViewObservation]
    ) -> list[CrossViewLink]:
        observations = list(observations)
        if not observations or self.llm_client is None:
            # Contract: empty/no-client → []. Never raise.
            return []

        # Bucket candidates by (class_name, segment_idx). Same-class only —
        # asking the LLM "is this person the same as that car?" wastes
        # tokens. segment_idx defaults to None (synchronized data) so the
        # bucketing degrades to class-only when segments aren't tracked.
        buckets: dict[tuple[str, Optional[int]], list[ViewObservation]] = defaultdict(list)
        for o in observations:
            buckets[(o.class_name, o.segment_idx)].append(o)

        links: list[CrossViewLink] = []
        now = time.time()
        for bucket in buckets.values():
            links.extend(self._link_bucket(bucket, now))

        links.sort(key=lambda link: link.confidence, reverse=True)
        return links

    # ------------------------------------------------------------------
    # Per-bucket pairing + LLM call
    # ------------------------------------------------------------------

    def _link_bucket(
        self, bucket: list[ViewObservation], created_at: float,
    ) -> list[CrossViewLink]:
        by_view: dict[str, list[ViewObservation]] = defaultdict(list)
        for o in bucket:
            by_view[o.view_id].append(o)
        view_ids = list(by_view.keys())
        if len(view_ids) < 2:
            return []

        out: list[CrossViewLink] = []
        # Pairwise: every observation from view i paired with every
        # observation from view j (j > i). Each pair → one LLM call.
        # Cost is O(N_pairs); the caller is expected to keep N small via
        # M4.3 fallback gating.
        for i in range(len(view_ids)):
            for j in range(i + 1, len(view_ids)):
                va, vb = view_ids[i], view_ids[j]
                for a in by_view[va]:
                    for b in by_view[vb]:
                        link = self._judge_pair(a, b, created_at)
                        if link is not None:
                            out.append(link)
        return out

    def _judge_pair(
        self, a: ViewObservation, b: ViewObservation, created_at: float,
    ) -> Optional[CrossViewLink]:
        roi_a = self.roi_loader(a)
        roi_b = self.roi_loader(b)
        if roi_a is None or roi_b is None:
            # Can't show the LLM anything → skip rather than ask blind
            return None

        prompt = _LINK_PROMPT_TEMPLATE.format(
            view_a=a.view_id, view_b=b.view_id,
            class_a=a.class_name, class_b=b.class_name,
            tk_a=a.tracklet_id, tk_b=b.tracklet_id,
        )
        parsed = self._call_and_parse(prompt, [roi_a, roi_b])
        retries_left = self.max_retries
        while parsed is None and retries_left > 0:
            strict_prompt = (
                prompt
                + "\n上一次回复无法解析为 JSON。请只输出一行 JSON，不要任何其他字符。"
            )
            parsed = self._call_and_parse(strict_prompt, [roi_a, roi_b])
            retries_left -= 1
        if parsed is None:
            return None

        same_object = parsed.get("same_object")
        if same_object is not True:
            return None
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            return None
        # Clamp + threshold
        confidence = max(0.0, min(1.0, confidence))
        if confidence < self.confidence_threshold:
            return None

        observations = [
            (a.view_id, a.tracklet_id),
            (b.view_id, b.tracklet_id),
        ]
        return CrossViewLink(
            link_id=make_link_id(observations),
            view_observations=observations,
            confidence=confidence,
            created_by="llm",
            created_at=created_at,
        )

    def _call_and_parse(
        self, prompt: str, images: list[np.ndarray],
    ) -> Optional[dict]:
        response = self.llm_client.complete(
            prompt, images=images, max_new_tokens=self.max_new_tokens,
        )
        return _parse_json_block(response)


def _parse_json_block(response: str) -> Optional[dict]:
    """Extract the first {...} block from an LLM response and parse it.

    Mirrors `mva.l3_events.llm_mode._parse_json_block` — duplicated here
    because both layers parse Qwen JSON outputs and we want them to
    evolve independently if either side's prompt format changes.
    """
    if not response:
        return None
    text = response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
