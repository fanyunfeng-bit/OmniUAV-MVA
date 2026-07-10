"""LLMReasoner — §3.4 #3 promoted to real implementation in M4.2.

Replaces the M0 stub. Real impl as of 2026-05-23 (PLAN §6.2 M4.2):

- `detect_anomaly(view_id, tracklet_id)` — serializes the track's bbox-center
  series into a short text prompt + asks Qwen2.5-VL for `{type, severity,
  explanation}` JSON. Retries once on malformed output with a stricter
  prompt (per PLAN §3.5 L4 retry rule). Second failure → return None.
- `classify_behavior(view_id, tracklet_id, context)` — strict-vocab text
  classifier. The algorithmic baseline returns "unknown" today; LLM mode
  picks from a fixed label set so it can't hallucinate exotic labels.
- `predict_trajectory(view_id, tracklet_id, horizon)` — **deferred to
  M5+** per 2026-05-23 design lock. K=4 sparse sampling (2.5s frame gap)
  can't show the long-range / non-linear motion where LLM would beat
  M3.2's const-vel baseline. Returns None so the parametrized contract
  test treats it as a stub-mode pass.

Construct LLMReasoner with `store` (WorldStateStore) + `llm_client`
(LLMClient). The zero-arg ctor is preserved for the contract-test fixture:
no store / no client → all methods degrade to None / "unknown" cleanly.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from mva.contracts import Anomaly, TrajectoryPrediction
from mva.l3_events.algorithmic import _bboxes_to_centers


_ANOMALY_PROMPT_TEMPLATE = """你是视频分析助手。下面是一个目标在 {view_id} 视角下的轨迹：

tracklet_id: {tracklet_id}
duration: {duration:.1f}s
n_observations: {n_obs}
mean_speed: {mean_speed_str} px/s
max_displacement_from_start: {max_disp:.1f} px
centers (t, cx, cy):
{centers_text}

判断是否存在异常行为。可选类型：
- loitering: 长时间停留在很小的区域内
- speed_spike: 速度明显异常（相对其他物体或自身历史）
- none: 无异常

严格只返回一行 JSON，键固定，不要任何其他文字：
{{"type": "loitering|speed_spike|none", "severity": "low|medium|high", "explanation": "<一句话原因>"}}
"""

_ANOMALY_TYPES = {"loitering", "speed_spike", "none"}
_SEVERITIES = {"low", "medium", "high"}


_BEHAVIOR_PROMPT_TEMPLATE = """你是视频分析助手。下面是一个目标在 {view_id} 视角下的轨迹概要：

tracklet_id: {tracklet_id}
duration: {duration:.1f}s
mean_speed: {mean_speed_str} px/s
max_displacement_from_start: {max_disp:.1f} px
context: {context}

从下面 6 个固定标签中选最贴切的一个。**只返回标签字符串本身**（小写、下划线分隔），不要任何其他文字：
walking, running, stationary, vehicle_moving, vehicle_parked, unknown
"""

_KNOWN_BEHAVIORS = (
    "walking", "running", "stationary",
    "vehicle_moving", "vehicle_parked", "unknown",
)


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class LLMReasoner:
    """LLM-mode L3 reasoner. Dual-mode counterpart to AlgorithmicReasoner;
    both satisfy the `Reasoner` Protocol and share the parametrized
    contract test fixture.

    Parameters
    ----------
    store : WorldStateStore | None
        Tracklet source. When None all methods return the degenerate
        contract value (None / "unknown") — used by the no-arg contract
        fixture in `tests/contracts/test_events.py`.
    llm_client : LLMClient | None
        Qwen2.5-VL wrapper. When None the reasoner has no LLM to call →
        degenerate path same as no-store. When set, all completions use
        text-only `complete()` (M4.2 doesn't fetch ROI images yet — that's
        M4.1's job).
    max_retries : int
        How many extra LLM calls we make when the first response fails to
        parse. M4.2 default 1 = "one retry with stricter prompt" per
        PLAN §3.5 L4 rule. The second failure short-circuits to None.
    n_inline_centers : int
        How many bbox-center rows to dump verbatim into the prompt. Keeps
        prompts under ~500 chars even for long tracklets. Default 10
        balances signal vs. token budget on Qwen2.5-VL-7B.
    """

    def __init__(
        self,
        store: Any = None,
        llm_client: Any = None,
        *,
        max_retries: int = 1,
        n_inline_centers: int = 10,
    ) -> None:
        self.store = store
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.n_inline_centers = n_inline_centers

    # ------------------------------------------------------------------ Protocol

    def predict_trajectory(
        self, view_id: str, tracklet_id: str, horizon: float
    ) -> Optional[TrajectoryPrediction]:
        # 2026-05-23 design lock: LLM trajectory prediction deferred to M5+
        # (PLAN §6.2 M4.2). K=4 sparse sampling can't surface long-range /
        # non-linear motion patterns where LLM would beat const-vel
        # baseline. Returning None keeps the parametrized contract test
        # green (it accepts Optional[TrajectoryPrediction]).
        return None

    def classify_behavior(
        self, view_id: str, tracklet_id: str, context: dict,
    ) -> str:
        if self.store is None or self.llm_client is None:
            return "unknown"
        centers = self._get_centers(view_id, tracklet_id)
        if len(centers) < 2:
            return "unknown"
        summary = self._track_summary(centers)
        prompt = _BEHAVIOR_PROMPT_TEMPLATE.format(
            view_id=view_id,
            tracklet_id=tracklet_id,
            duration=summary["duration"],
            mean_speed_str=_fmt_optional(summary["mean_speed"]),
            max_disp=summary["max_disp"],
            context=_compact_context(context),
        )
        # Behavior labels are short; we keep max_new_tokens tight to
        # discourage the model from rambling past the label.
        response = self.llm_client.complete(prompt, max_new_tokens=16)
        label = _extract_behavior_label(response)
        return label if label in _KNOWN_BEHAVIORS else "unknown"

    def detect_anomaly(
        self, view_id: str, tracklet_id: str,
    ) -> Optional[Anomaly]:
        if self.store is None or self.llm_client is None:
            return None
        centers = self._get_centers(view_id, tracklet_id)
        if len(centers) < 2:
            return None
        summary = self._track_summary(centers)
        prompt = _ANOMALY_PROMPT_TEMPLATE.format(
            view_id=view_id,
            tracklet_id=tracklet_id,
            duration=summary["duration"],
            n_obs=len(centers),
            mean_speed_str=_fmt_optional(summary["mean_speed"]),
            max_disp=summary["max_disp"],
            centers_text=_format_centers(centers, max_rows=self.n_inline_centers),
        )

        parsed = self._call_and_parse_anomaly(prompt)
        # PLAN §3.5 L4: one retry with a stricter prompt, then degrade to
        # natural-language fallback (here: return None — the calling layer
        # never sees a hallucinated Anomaly).
        retries_left = self.max_retries
        while parsed is None and retries_left > 0:
            strict_prompt = (
                prompt
                + "\n上一次回复无法解析为 JSON。请只输出一行 JSON，不要任何其他字符。"
            )
            parsed = self._call_and_parse_anomaly(strict_prompt)
            retries_left -= 1
        if parsed is None:
            return None
        anomaly_type = parsed.get("type")
        if anomaly_type not in _ANOMALY_TYPES or anomaly_type == "none":
            return None
        severity = parsed.get("severity", "medium")
        if severity not in _SEVERITIES:
            severity = "medium"
        return Anomaly(
            event_id=f"llm-{anomaly_type}-{view_id}-{tracklet_id}",
            tracklet_ids=[tracklet_id],
            t=float(centers[0][0]),
            type=anomaly_type,
            severity=severity,
            explanation=parsed.get("explanation"),
        )

    # ------------------------------------------------------------------ Helpers

    def _get_centers(
        self, view_id: str, tracklet_id: str,
    ) -> list[tuple[float, float, float]]:
        for tk in self.store.query_tracklets(view_id):
            if tk["tracklet_id"] != tracklet_id:
                continue
            return _bboxes_to_centers(tk["bboxes"])
        return []

    def _call_and_parse_anomaly(self, prompt: str) -> Optional[dict]:
        response = self.llm_client.complete(prompt, max_new_tokens=128)
        return _parse_json_block(response)

    @staticmethod
    def _track_summary(centers: list[tuple[float, float, float]]) -> dict:
        t_first = centers[0][0]
        t_last = centers[-1][0]
        duration = max(0.0, t_last - t_first)
        x0, y0 = centers[0][1], centers[0][2]
        max_disp = 0.0
        for _, x, y in centers:
            d = math.hypot(x - x0, y - y0)
            if d > max_disp:
                max_disp = d
        total_dist = 0.0
        total_dt = 0.0
        for (a_t, a_x, a_y), (b_t, b_x, b_y) in zip(centers[:-1], centers[1:]):
            dt = b_t - a_t
            if dt <= 0:
                continue
            total_dist += math.hypot(b_x - a_x, b_y - a_y)
            total_dt += dt
        mean_speed: Optional[float] = None
        if total_dt > 0:
            mean_speed = total_dist / total_dt
        return {
            "duration": duration,
            "mean_speed": mean_speed,
            "max_disp": max_disp,
        }


# ----------------------------------------------------------------------
# Module-level helpers (intentionally not on the class so unit tests can
# import + drive them without instantiating an LLMReasoner)
# ----------------------------------------------------------------------


def _format_centers(
    centers: list[tuple[float, float, float]], *, max_rows: int,
) -> str:
    rows = centers[:max_rows]
    lines = [f"  ({t:.1f}, {cx:.0f}, {cy:.0f})" for t, cx, cy in rows]
    if len(centers) > max_rows:
        lines.append(f"  ... ({len(centers) - max_rows} more, total={len(centers)})")
    return "\n".join(lines)


def _fmt_optional(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "n/a"


def _compact_context(context: dict) -> str:
    if not context:
        return "{}"
    # Keep the prompt deterministic — sort keys and stringify values.
    parts = [f"{k}={context[k]!r}" for k in sorted(context)]
    return "{" + ", ".join(parts) + "}"


def _parse_json_block(response: str) -> Optional[dict]:
    """Extract the first {...} block from an LLM response and parse it.

    Qwen sometimes wraps JSON in code fences or prefixes it with a
    sentence ("Here is the result: {...}"). We tolerate both. Returns
    None when no parseable JSON object is found — the caller decides
    whether to retry.
    """
    if not response:
        return None
    text = response.strip()
    # Try a clean parse first (model obeyed the "JSON only" instruction)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Otherwise scan for an embedded {...} block
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_behavior_label(response: str) -> str:
    """Pull a known-vocab label out of a possibly-wordy LLM response.

    Strict-vocab classification: we accept the model only when it
    actually says one of the six fixed labels. Anything else falls back
    to "unknown" so the caller knows the LLM punted.
    """
    if not response:
        return "unknown"
    # Lowercase + strip punctuation around tokens
    text = response.strip().lower()
    # First check: clean exact match (the prompt asks for "label only")
    if text in _KNOWN_BEHAVIORS:
        return text
    # Otherwise scan for the first known label appearing as a substring.
    # Order matters: longer / more-specific labels first so "vehicle_moving"
    # wins over "vehicle".
    for label in sorted(_KNOWN_BEHAVIORS, key=len, reverse=True):
        if label in text:
            return label
    return "unknown"
