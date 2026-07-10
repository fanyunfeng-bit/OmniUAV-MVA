"""检索纯逻辑：解析 ChromaDB 命中 + 从 DuckDB 富化段时间。无 GPU 依赖，可单测。"""
from __future__ import annotations
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
