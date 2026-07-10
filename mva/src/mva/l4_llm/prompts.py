"""L4 prompt templates.

The `{telemetry_summary}` slot is the §3.4 #4 interface: empty in v0.x,
populated by L0 telemetry in v2+.
"""
from __future__ import annotations

from typing import Optional


SCENE_DESCRIBE_TEMPLATE = """你是一个无人机监控视频分析员，正在观察来自 drone {view_id} 的画面。
{telemetry_summary}

请描述你在画面里看到的内容，重点关注：
- 感兴趣的物体（车辆、行人等）
- 颜色、类型、空间位置关系
- 任何值得注意的活动或行为

请用简洁的自然语言回答。
"""


def _format_telemetry(telemetry: Optional[dict]) -> str:
    """Format telemetry dict into a 1-2 line summary for the prompt.

    Returns empty string when telemetry is None (v0.x default), so prompts
    stay clean. M4+ when telemetry is populated, this renders e.g.:
        "Drone 当前 GPS = (39.9, 116.4), 海拔 100m, 云台朝向 NE"
    """
    if not telemetry:
        return ""

    parts: list[str] = []
    gps_lat = telemetry.get("gps_lat")
    gps_lon = telemetry.get("gps_lon")
    if gps_lat is not None and gps_lon is not None:
        parts.append(f"GPS=({gps_lat:.4f}, {gps_lon:.4f})")
    alt = telemetry.get("alt")
    if alt is not None:
        parts.append(f"altitude={alt}m")
    if not parts:
        return ""
    return "Drone 当前状态: " + ", ".join(parts) + "\n"


def render_describe_prompt(
    view_id: str, telemetry: Optional[dict] = None
) -> str:
    """Render the single-view scene description prompt.

    Empty telemetry → no telemetry block in the prompt. Non-empty telemetry
    → prefix line with drone state. 🔌 §3.4 #4 placeholder.
    """
    return SCENE_DESCRIBE_TEMPLATE.format(
        view_id=view_id,
        telemetry_summary=_format_telemetry(telemetry),
    )
