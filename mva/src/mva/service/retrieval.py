"""检索纯逻辑：解析 ChromaDB 命中 + 从 DuckDB 富化段时间。无 GPU 依赖，可单测。"""
from __future__ import annotations
import re
from typing import Any


def strip_scene(view_id: str) -> str:
    """'Reservoir::view1' -> 'view1'（DuckDB 用无前缀 view_id）。"""
    return view_id.split("::", 1)[-1] if "::" in view_id else view_id


def parse_hits(raw: list[dict]) -> list[dict]:
    hits = []
    for r in raw:
        md = r.get("metadata") or {}
        kind = "segment" if md.get("vector_kind") == "segment" else "bbox"
        dist = r.get("distance")
        score = (1.0 - float(dist)) if dist is not None else 0.0
        hits.append({
            "view_id": strip_scene(md.get("view_id", "")),
            "segment_idx": md.get("segment_idx"),
            "tracklet_id": md.get("tracklet_id"),
            "class_name": md.get("class_name"),
            "kind": kind,
            "score": score,
            "doc": r.get("document"),
        })
    return hits


def enrich_segment_time(hit: dict, store: Any) -> dict:
    """段级命中：查 segments 表补 start_t(=t) 与 source_uri。"""
    if hit.get("kind") != "segment" or hit.get("segment_idx") is None:
        return hit
    sql = (
        "SELECT start_t, source_uri FROM segments "
        f"WHERE view_id = '{hit['view_id']}' AND segment_idx = {int(hit['segment_idx'])} "
        "LIMIT 1"
    )
    try:
        rows = store.execute_readonly(sql)
    except Exception:                       # noqa: BLE001
        rows = []
    if rows:
        row = rows[0]
        hit["t"] = row.get("start_t")
        hit["source_uri"] = row.get("source_uri")
    return hit


def resolve_view_ref(view_ref, views):
    """'1' → `views` 里的具体 raw view_id；无法解析返回 None。

    1) 唯一数字子串匹配('1'↔'view1'/'cam01'/'drone1')；
    2) 否则按名排序取 1-indexed 第 N 个；
    3) 子串多命中时取排序首个；越界/无 view → None。
    """
    if not view_ref or not views:
        return None
    n = str(view_ref).strip()
    if not n.isdigit():
        return None
    num = int(n)
    pat = re.compile(rf"(?<!\d)0*{num}(?!\d)")
    matches = [v for v in views if pat.search(v)]
    if len(matches) == 1:
        return matches[0]
    ordered = sorted(views)
    if 1 <= num <= len(ordered):
        return ordered[num - 1]
    if matches:
        return sorted(matches)[0]
    return None


def resolve_time(c, duration):
    """把 QueryConstraints 的时间解析成绝对 (qs, qe) 秒。

    relative_to_end 时 time_* 是"距末尾偏移量"，用 duration 换算：real = dur - offset。
    duration 缺失(None)时相对时间无法换算 → (None, None)。
    """
    if c.time_start is None and c.time_end is None:
        return None, None
    if not c.relative_to_end:
        return c.time_start, c.time_end
    if duration is None:
        return None, None
    qs = max(0.0, duration - (c.time_start or 0.0))
    qe = max(0.0, duration - (c.time_end or 0.0))
    return (min(qs, qe), max(qs, qe))


def build_metadata_where(view_id_raw, t_start, t_end):
    """chroma `where`(段向量)：view_id_raw 等值 + 时间重叠。

    query [t_start,t_end] 与段 [start_t,end_t] 重叠 ⇔ start_t≤t_end AND end_t≥t_start。
    """
    clauses = []
    if view_id_raw:
        clauses.append({"view_id_raw": view_id_raw})
    if t_end is not None:
        clauses.append({"start_t": {"$lte": float(t_end)}})
    if t_start is not None:
        clauses.append({"end_t": {"$gte": float(t_start)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
