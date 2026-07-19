"""L5 structured store (DuckDB).

Per-view tables: tracklets_<v>, events_<v>, captions_<v> — created lazily
on first insert for a given view_id.
Shared tables: cross_view_links, telemetry — created in __init__.

Concurrency: DuckDB allows only a single writer; we wrap all access in an
RLock so multiple L0/L1/L3 threads in the same Python process can safely
hit the store. M2+: spin out a dedicated writer process + asyncio queue
per PLAN.md §3.2 L5.

PLAN.md §3.4 #8 (`human_correction_endpoint`) returns 501 here; the real
write path arrives in M5 with L7.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import duckdb

from mva.contracts import CrossViewLink, Event


_VIEW_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


def _row_to_segment_dict(row: tuple) -> dict:
    """Turn a raw SQL row from `segments` into a friendly dict. Parses
    `detected_counts_json` back into a Python dict when present."""
    counts = None
    if row[8]:
        try:
            counts = json.loads(row[8])
        except (json.JSONDecodeError, TypeError):
            counts = None
    return {
        "view_id": row[0],
        "segment_idx": row[1],
        "start_t": row[2],
        "end_t": row[3],
        "source_uri": row[4],
        "embed_chroma_id": row[5],
        "nframes_sampled": row[6],
        "detected_classes": row[7],
        "detected_counts": counts,
    }


def _sanitize_view_id(view_id: str) -> str:
    """Sanitize view_id for use in a DuckDB identifier.

    Identifiers must start with a letter and contain only [A-Za-z0-9_].
    We replace anything outside that set with `_` and reject empty / digit-leading
    strings. Original view_id stays untouched in the rows themselves; this only
    affects table-name suffixes.
    """
    if not view_id or not _VIEW_ID_RE.match(view_id):
        if not view_id:
            raise ValueError("view_id must be non-empty")
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", view_id)
    if not re.match(r"^[A-Za-z_]", suffix):
        suffix = "v_" + suffix
    return suffix


class WorldStateStore:
    """Per-view + shared structured tables on DuckDB."""

    # Telemetry fields broken out as columns (the rest goes into imu_extra JSON).
    _TELEMETRY_COLUMNS = (
        "gps_lat",
        "gps_lon",
        "alt",
        "gimbal_qx",
        "gimbal_qy",
        "gimbal_qz",
        "gimbal_qw",
    )

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = duckdb.connect(db_path)
        self._known_views: set[str] = set()
        self._create_shared_tables()
        self._discover_existing_views()

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "WorldStateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- schema ----------------------------------------------------------

    def _create_shared_tables(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cross_view_links (
                    link_id            VARCHAR PRIMARY KEY,
                    view_observations  VARCHAR,  -- JSON: [[view_id, tracklet_id], ...]
                    confidence         DOUBLE,
                    created_by         VARCHAR,
                    created_at         DOUBLE
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    t           DOUBLE,
                    view_id     VARCHAR,
                    gps_lat     DOUBLE,
                    gps_lon     DOUBLE,
                    alt         DOUBLE,
                    gimbal_qx   DOUBLE,
                    gimbal_qy   DOUBLE,
                    gimbal_qz   DOUBLE,
                    gimbal_qw   DOUBLE,
                    imu_extra   VARCHAR  -- JSON blob for IMU + future fields
                );
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS telemetry_t_view "
                "ON telemetry(t, view_id);"
            )
            # M2.8 — segments table (per view × per sliding-window segment).
            # PRIMARY KEY (view_id, segment_idx) so re-running ingest with
            # the same config is idempotent (INSERT OR REPLACE).
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    view_id              VARCHAR,
                    segment_idx          INTEGER,
                    start_t              DOUBLE,
                    end_t                DOUBLE,
                    source_uri           VARCHAR,
                    embed_chroma_id      VARCHAR,        -- segment-level vector id in ChromaDB
                    nframes_sampled      INTEGER,
                    detected_classes     VARCHAR,        -- comma-joined sorted class names, NULL if no detection
                    detected_counts_json VARCHAR,        -- JSON {class: total_count_in_segment}
                    PRIMARY KEY (view_id, segment_idx)
                );
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS segments_view_t "
                "ON segments(view_id, start_t);"
            )
            # Phase 0 — global 3D fusion tables (M2 位姿 / M3 全局对象)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS camera_poses (
                    view_id VARCHAR, t DOUBLE,
                    fx DOUBLE, fy DOUBLE, cx DOUBLE, cy DOUBLE,
                    qx DOUBLE, qy DOUBLE, qz DOUBLE, qw DOUBLE,
                    tx DOUBLE, ty DOUBLE, tz DOUBLE,
                    PRIMARY KEY (view_id, t)
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_objects (
                    global_id VARCHAR PRIMARY KEY, class_name VARCHAR,
                    first_t DOUBLE, last_t DOUBLE, n_views INTEGER, confidence DOUBLE
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_observations (
                    global_id VARCHAR, view_id VARCHAR, view_track_id VARCHAR, t DOUBLE,
                    bx1 DOUBLE, by1 DOUBLE, bx2 DOUBLE, by2 DOUBLE,
                    wx DOUBLE, wy DOUBLE, wz DOUBLE
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_trajectory (
                    global_id VARCHAR, t DOUBLE, x DOUBLE, y DOUBLE, z DOUBLE,
                    vx DOUBLE, vy DOUBLE,
                    PRIMARY KEY (global_id, t)
                );
                """
            )
            # Phase 0 — M4 时空推理表（场景图 / 事件 / 预测）
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_graph_edges (
                    t DOUBLE, subj_global_id VARCHAR, rel VARCHAR,
                    obj VARCHAR, confidence DOUBLE
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS situation_events (
                    event_id VARCHAR PRIMARY KEY, kind VARCHAR,
                    t_start DOUBLE, t_end DOUBLE,
                    global_ids VARCHAR,   -- JSON list
                    region VARCHAR, confidence DOUBLE
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_predictions (
                    global_id VARCHAR, t_future DOUBLE, x DOUBLE, y DOUBLE, confidence DOUBLE
                );
                """
            )

    def _discover_existing_views(self) -> None:
        """Seed _known_views from on-disk tables so reads work after reopen."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'tracklets\\_%' ESCAPE '\\'"
            ).fetchall()
        for (name,) in rows:
            self._known_views.add(name[len("tracklets_"):])

    def _ensure_view_tables(self, view_id: str) -> str:
        """Create the per-view triplet on first touch. Returns the suffix."""
        suffix = _sanitize_view_id(view_id)
        if suffix in self._known_views:
            return suffix
        with self._lock:
            # M2.8: segment_idx links a tracklet back to its parent segment
            # row in the shared `segments` table. Nullable for legacy rows
            # written by the frozen `mva perceive` path.
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS tracklets_{suffix} (
                    tracklet_id    VARCHAR PRIMARY KEY,
                    t_start        DOUBLE,
                    t_end          DOUBLE,
                    bboxes         VARCHAR,   -- JSON list of [t, x1, y1, x2, y2, class_name, conf]
                    embedding_ref  VARCHAR,   -- bbox embedding id in ChromaDB (nullable)
                    segment_idx    INTEGER    -- parent segment in segments table (nullable)
                );
                """
            )
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS events_{suffix} (
                    event_id       VARCHAR PRIMARY KEY,
                    tracklet_ids   VARCHAR,   -- JSON list of strings
                    t              DOUBLE,
                    type           VARCHAR,
                    summary_text   VARCHAR
                );
                """
            )
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS captions_{suffix} (
                    caption_id     VARCHAR PRIMARY KEY,
                    frame_idx      INTEGER,
                    t              DOUBLE,
                    caption_text   VARCHAR
                );
                """
            )
            self._known_views.add(suffix)
        return suffix

    # ---- inserts ---------------------------------------------------------

    def insert_tracklet(
        self,
        view_id: str,
        tracklet_id: str,
        t_start: float,
        t_end: float,
        bboxes: Iterable[Any],
        embedding_ref: Optional[str] = None,
        segment_idx: Optional[int] = None,
    ) -> None:
        """Insert or REPLACE a tracklet row by tracklet_id.

        Re-inserting the same tracklet_id overwrites the previous row.
        This makes demo / re-run flows safe (deterministic per-frame
        tracklet ids like `D1-f0-d0` would otherwise crash on second run).

        `segment_idx` is M2.8 — links the tracklet to its parent row in
        the shared `segments` table. None for legacy `mva perceive` rows.
        """
        suffix = self._ensure_view_tables(view_id)
        with self._lock:
            self.conn.execute(
                f"INSERT OR REPLACE INTO tracklets_{suffix} VALUES (?, ?, ?, ?, ?, ?)",
                [
                    tracklet_id,
                    t_start,
                    t_end,
                    json.dumps(list(bboxes)),
                    embedding_ref,
                    segment_idx,
                ],
            )

    def insert_segment(
        self,
        view_id: str,
        segment_idx: int,
        start_t: float,
        end_t: float,
        source_uri: str,
        embed_chroma_id: Optional[str] = None,
        nframes_sampled: Optional[int] = None,
        detected_classes: Optional[str] = None,
        detected_counts: Optional[dict[str, int]] = None,
    ) -> None:
        """🆕 M2.8 — insert or REPLACE a segment row.

        `detected_counts` (dict) is JSON-serialized into
        `detected_counts_json`. Both detection columns are NULL when
        ingest ran without --detect.
        """
        if not isinstance(segment_idx, int):
            raise TypeError(
                f"segment_idx must be int, got {type(segment_idx).__name__}"
            )
        counts_json = (
            json.dumps(detected_counts, ensure_ascii=False, sort_keys=True)
            if detected_counts else None
        )
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO segments VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    view_id,
                    segment_idx,
                    start_t,
                    end_t,
                    source_uri,
                    embed_chroma_id,
                    nframes_sampled,
                    detected_classes,
                    counts_json,
                ],
            )

    def insert_event(self, event: Event) -> None:
        """Insert or REPLACE an event row by event_id."""
        suffix = self._ensure_view_tables(event.view_id)
        with self._lock:
            self.conn.execute(
                f"INSERT OR REPLACE INTO events_{suffix} VALUES (?, ?, ?, ?, ?)",
                [
                    event.event_id,
                    json.dumps(list(event.tracklet_ids)),
                    event.t,
                    event.type,
                    event.summary_text,
                ],
            )

    def insert_caption(
        self,
        view_id: str,
        caption_id: str,
        frame_idx: int,
        t: float,
        caption_text: str,
    ) -> None:
        """Insert or REPLACE a caption row by caption_id."""
        suffix = self._ensure_view_tables(view_id)
        with self._lock:
            self.conn.execute(
                f"INSERT OR REPLACE INTO captions_{suffix} VALUES (?, ?, ?, ?)",
                [caption_id, frame_idx, t, caption_text],
            )

    def insert_cross_view_link(self, link: CrossViewLink) -> None:
        # INSERT OR REPLACE so reruns with deterministic link_ids (see
        # `mva.contracts.make_link_id`) overwrite the prior row instead
        # of stacking duplicates. Original M3.4 fix covered ChromaDB +
        # segments + tracklets but missed this table.
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cross_view_links VALUES (?, ?, ?, ?, ?)",
                [
                    link.link_id,
                    json.dumps([list(obs) for obs in link.view_observations]),
                    link.confidence,
                    link.created_by,
                    link.created_at,
                ],
            )

    def insert_telemetry(self, t: float, view_id: str, telemetry: dict) -> None:
        """🔌 §3.4 #1 — telemetry write path. Splits known columns + JSON blob."""
        extras = {k: v for k, v in telemetry.items() if k not in self._TELEMETRY_COLUMNS}
        with self._lock:
            self.conn.execute(
                "INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    t,
                    view_id,
                    telemetry.get("gps_lat"),
                    telemetry.get("gps_lon"),
                    telemetry.get("alt"),
                    telemetry.get("gimbal_qx"),
                    telemetry.get("gimbal_qy"),
                    telemetry.get("gimbal_qz"),
                    telemetry.get("gimbal_qw"),
                    json.dumps(extras) if extras else None,
                ],
            )

    # ---- queries ---------------------------------------------------------

    def query_tracklets(
        self,
        view_id: str,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        segment_idx: Optional[int] = None,
    ) -> list[dict]:
        """Time-overlap query: returns tracklets whose [t_start, t_end]
        intersects [t_start, t_end]. M2.8: optional `segment_idx` filter
        narrows to one parent segment."""
        suffix = _sanitize_view_id(view_id)
        if suffix not in self._known_views:
            return []
        sql = (
            f"SELECT tracklet_id, t_start, t_end, bboxes, embedding_ref, "
            f"segment_idx FROM tracklets_{suffix}"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if t_end is not None:
            clauses.append("t_start <= ?")
            params.append(t_end)
        if t_start is not None:
            clauses.append("t_end >= ?")
            params.append(t_start)
        if segment_idx is not None:
            clauses.append("segment_idx = ?")
            params.append(segment_idx)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY t_start"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "tracklet_id": r[0],
                "t_start": r[1],
                "t_end": r[2],
                "bboxes": json.loads(r[3]) if r[3] else [],
                "embedding_ref": r[4],
                "segment_idx": r[5],
            }
            for r in rows
        ]

    def query_segments(
        self,
        view_id: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> list[dict]:
        """🆕 M2.8 — query segments by view + time-window overlap.

        Returns dicts with all segment columns + `detected_counts` parsed
        back from JSON for convenience.
        """
        sql = (
            "SELECT view_id, segment_idx, start_t, end_t, source_uri, "
            "embed_chroma_id, nframes_sampled, detected_classes, "
            "detected_counts_json FROM segments"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if view_id is not None:
            clauses.append("view_id = ?")
            params.append(view_id)
        if t_end is not None:
            clauses.append("start_t <= ?")
            params.append(t_end)
        if t_start is not None:
            clauses.append("end_t >= ?")
            params.append(t_start)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY view_id, segment_idx"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_segment_dict(r) for r in rows]

    def get_segment(
        self, view_id: str, segment_idx: int,
    ) -> Optional[dict]:
        """🆕 M2.8 — fetch one segment by (view_id, segment_idx). None if
        not found. Used by retrieval to map a ChromaDB hit back to the
        source clip ((source_uri, start_t, end_t))."""
        with self._lock:
            row = self.conn.execute(
                "SELECT view_id, segment_idx, start_t, end_t, source_uri, "
                "embed_chroma_id, nframes_sampled, detected_classes, "
                "detected_counts_json FROM segments "
                "WHERE view_id = ? AND segment_idx = ?",
                [view_id, segment_idx],
            ).fetchone()
        return _row_to_segment_dict(row) if row else None

    def get_segment_by_chroma_id(self, embed_chroma_id: str) -> Optional[dict]:
        """🆕 M2.8 — reverse lookup: ChromaDB hit → segment row. Used by
        L6 tools that retrieve a segment-level embedding and need
        `(source_uri, start_t, end_t)` to localize the clip."""
        with self._lock:
            row = self.conn.execute(
                "SELECT view_id, segment_idx, start_t, end_t, source_uri, "
                "embed_chroma_id, nframes_sampled, detected_classes, "
                "detected_counts_json FROM segments "
                "WHERE embed_chroma_id = ?",
                [embed_chroma_id],
            ).fetchone()
        return _row_to_segment_dict(row) if row else None

    # ---- global fusion state (Phase 0) ----------------------------------

    def insert_camera_pose(self, pose) -> None:
        qx, qy, qz, qw = pose.quat
        tx, ty, tz = pose.translation
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO camera_poses VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [pose.view_id, pose.t, pose.fx, pose.fy, pose.cx, pose.cy,
                 qx, qy, qz, qw, tx, ty, tz],
            )

    def query_camera_poses(self, view_id=None) -> list[dict]:
        sql = "SELECT * FROM camera_poses"
        if view_id is not None:
            sql += f" WHERE view_id = '{view_id}'"
        return self.execute_readonly(sql + " ORDER BY view_id, t")

    def insert_global_object(self, obj) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO global_objects VALUES (?, ?, ?, ?, ?, ?)",
                [obj.global_id, obj.class_name, obj.first_t, obj.last_t,
                 obj.n_views, obj.confidence],
            )

    def query_global_objects(self) -> list[dict]:
        return self.execute_readonly("SELECT * FROM global_objects ORDER BY global_id")

    def insert_global_observation(self, obs) -> None:
        wx, wy, wz = obs.world_xyz if obs.world_xyz is not None else (None, None, None)
        bx1, by1, bx2, by2 = obs.bbox
        with self._lock:
            self.conn.execute(
                "INSERT INTO global_observations VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [obs.global_id, obs.view_id, obs.view_track_id, obs.t,
                 bx1, by1, bx2, by2, wx, wy, wz],
            )

    def query_global_observations(self, global_id=None) -> list[dict]:
        sql = "SELECT * FROM global_observations"
        if global_id is not None:
            sql += f" WHERE global_id = '{global_id}'"
        return self.execute_readonly(sql + " ORDER BY t")

    def insert_global_trajectory(self, pt) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO global_trajectory VALUES (?, ?, ?, ?, ?, ?, ?)",
                [pt.global_id, pt.t, pt.x, pt.y, pt.z, pt.vx, pt.vy],
            )

    def query_global_trajectory(self, global_id) -> list[dict]:
        return self.execute_readonly(
            f"SELECT * FROM global_trajectory WHERE global_id = '{global_id}' ORDER BY t")

    # ---- M4 reasoning state (Phase 0) -----------------------------------

    def insert_scene_graph_edge(self, e) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO scene_graph_edges VALUES (?, ?, ?, ?, ?)",
                [e.t, e.subj_global_id, e.rel, e.obj, e.confidence],
            )

    def query_scene_graph_edges(self, t=None) -> list[dict]:
        sql = "SELECT * FROM scene_graph_edges"
        if t is not None:
            sql += f" WHERE t = {float(t)}"
        return self.execute_readonly(sql + " ORDER BY t")

    def insert_situation_event(self, ev) -> None:
        gids = json.dumps(ev.global_ids, ensure_ascii=False)
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO situation_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                [ev.event_id, ev.kind, ev.t_start, ev.t_end, gids,
                 ev.region, ev.confidence],
            )

    def query_situation_events(self) -> list[dict]:
        return self.execute_readonly("SELECT * FROM situation_events ORDER BY t_start")

    def insert_global_prediction(self, p) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO global_predictions VALUES (?, ?, ?, ?, ?)",
                [p.global_id, p.t_future, p.x, p.y, p.confidence],
            )

    def query_global_predictions(self, global_id=None) -> list[dict]:
        sql = "SELECT * FROM global_predictions"
        if global_id is not None:
            sql += f" WHERE global_id = '{global_id}'"
        return self.execute_readonly(sql + " ORDER BY t_future")

    def execute_readonly(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query. Returns list of dicts."""
        normalized = sql.strip().rstrip(";").strip()
        if not normalized.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        if ";" in normalized:
            raise ValueError("Multi-statement queries are not allowed")
        with self._lock:
            result = self.conn.execute(normalized)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def drop_secondary_indexes(self) -> None:
        """Drop the optional `segments_view_t` secondary index.

        The live-ingest worker re-ingests the same video windows, so segments
        accumulate DUPLICATE (view_id, start_t) keys in this ART secondary
        index. DuckDB's ART delete on a multi-row-id leaf then fails with
        "Failed to delete all rows from index. Only deleted 0 out of 1 rows"
        and FATALLY invalidates the connection — so FIFO eviction can't work
        while the index exists. The PK index (no duplicates) deletes fine.
        query_segments falls back to a table scan, which is trivial on the
        bounded live table. Static ingest keeps its index (this is opt-in,
        called only by the live path)."""
        with self._lock:
            self.conn.execute("DROP INDEX IF EXISTS segments_view_t")

    def list_segment_indices(self, view_id: str) -> list[int]:
        """All segment_idx for a view, ascending (oldest first). Used by the
        live-ingest worker to pick which segments to evict (stack/FIFO)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT segment_idx FROM segments WHERE view_id = ? "
                "ORDER BY segment_idx",
                [view_id],
            ).fetchall()
        return [r[0] for r in rows]

    def delete_segment(self, view_id: str, segment_idx: int) -> list[str]:
        """Delete a segment row + its tracklets, returning the ChromaDB ids to
        evict (the segment embedding + each tracklet's bbox embedding).

        Used by the live-ingest worker's stack-bounded FIFO eviction so the
        rolling window stays bounded in BOTH DuckDB and ChromaDB. Idempotent —
        deleting a missing segment returns []."""
        suffix = _sanitize_view_id(view_id)
        chroma_ids: list[str] = []
        with self._lock:
            row = self.conn.execute(
                "SELECT embed_chroma_id FROM segments "
                "WHERE view_id = ? AND segment_idx = ?",
                [view_id, segment_idx],
            ).fetchone()
            if row and row[0]:
                chroma_ids.append(row[0])
            if suffix in self._known_views:
                trows = self.conn.execute(
                    f"SELECT embedding_ref FROM tracklets_{suffix} "
                    f"WHERE segment_idx = ?",
                    [segment_idx],
                ).fetchall()
                chroma_ids.extend(r[0] for r in trows if r[0])
                self.conn.execute(
                    f"DELETE FROM tracklets_{suffix} WHERE segment_idx = ?",
                    [segment_idx],
                )
            self.conn.execute(
                "DELETE FROM segments WHERE view_id = ? AND segment_idx = ?",
                [view_id, segment_idx],
            )
        return chroma_ids

    def get_schema_summary(self) -> str:
        """Return schema + data summary for the planner prompt."""
        _TABLE_NOTES = {
            "segments": "每个视角的10s时间窗（时间查询用这个表）",
            "cross_view_links": "跨视角同一目标匹配（判断不同视角是否有相同物体用这个表）",
            "telemetry": "无人机遥测数据",
        }

        with self._lock:
            tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            lines = ["Tables:"]
            for (tname,) in tables:
                cols = self.conn.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = ? ORDER BY ordinal_position", [tname]
                ).fetchall()
                row_count = self.conn.execute(
                    f'SELECT COUNT(*) FROM "{tname}"'
                ).fetchone()[0]
                col_str = ", ".join(c for c, _ in cols)
                note = _TABLE_NOTES.get(tname, "")
                if not note and tname.startswith("tracklets_"):
                    note = f"{tname.replace('tracklets_', '')}视角的检测轨迹"
                if not note and tname.startswith("captions_"):
                    note = "字幕（可能为空）"
                if not note and tname.startswith("events_"):
                    note = "事件（可能为空）"
                ann = f" — {note}" if note else ""
                lines.append(f"  {tname} [{row_count} rows]{ann}")
                lines.append(f"    columns: {col_str}")

            view_rows = self.conn.execute(
                "SELECT view_id, COUNT(*) AS n FROM segments "
                "GROUP BY view_id ORDER BY view_id"
            ).fetchall()
            link_count = self.conn.execute(
                "SELECT COUNT(*) FROM cross_view_links"
            ).fetchone()[0]
        if view_rows:
            views_str = ", ".join(f"{v} ({n} segments)" for v, n in view_rows)
            lines.append(f"\nData: {len(view_rows)} views ({views_str}), "
                         f"{link_count} cross-view links")
        return "\n".join(lines)

    def query_events(
        self,
        view_id: str,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        type: Optional[str] = None,
    ) -> list[Event]:
        suffix = _sanitize_view_id(view_id)
        if suffix not in self._known_views:
            return []
        sql = f"SELECT event_id, tracklet_ids, t, type, summary_text FROM events_{suffix}"
        clauses: list[str] = []
        params: list[Any] = []
        if t_start is not None:
            clauses.append("t >= ?")
            params.append(t_start)
        if t_end is not None:
            clauses.append("t <= ?")
            params.append(t_end)
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY t"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [
            Event(
                event_id=r[0],
                tracklet_ids=json.loads(r[1]) if r[1] else [],
                t=r[2],
                view_id=view_id,
                type=r[3],
                summary_text=r[4] or "",
            )
            for r in rows
        ]

    def query_captions(
        self,
        view_id: str,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> list[dict]:
        suffix = _sanitize_view_id(view_id)
        if suffix not in self._known_views:
            return []
        sql = f"SELECT caption_id, frame_idx, t, caption_text FROM captions_{suffix}"
        clauses: list[str] = []
        params: list[Any] = []
        if t_start is not None:
            clauses.append("t >= ?")
            params.append(t_start)
        if t_end is not None:
            clauses.append("t <= ?")
            params.append(t_end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY t"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [
            {"caption_id": r[0], "frame_idx": r[1], "t": r[2], "caption_text": r[3]}
            for r in rows
        ]

    def query_cross_view_links(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        min_confidence: Optional[float] = None,
        sort_by: Optional[Literal[
            "confidence_desc", "confidence_asc",
            "created_at_desc", "created_at_asc",
        ]] = None,
        limit: Optional[int] = None,
        view_id: Optional[Any] = None,
    ) -> list[CrossViewLink]:
        """Query cross_view_links with optional filters and sort/limit.

        M3.6.A (PROBLEMS P1-04): `sort_by` + `limit` let the LLM ask
        "top-k highest-confidence links" without abusing
        `min_confidence=1.0` (which yields the empty set because no link
        ever lands exactly at 1.0). Default ordering (sort_by=None) is
        kept at "confidence DESC, created_at DESC" for back-compat with
        every M0-M3.5 caller.

        M3.7 (PROBLEMS P1-09): `view_id` filter. Pass a single
        `view_id="D1"` to return only links that include D1 anywhere in
        their `view_observations`; pass a list `view_id=["D1", "D3"]`
        to require the link to span BOTH D1 AND D3 (the link's
        observation set must be a superset of the requested view_id
        set). Filtering happens after the SQL fetch — when view_id is
        set, SQL `LIMIT` is deferred so we don't lose post-filter
        results. Empty list raises ValueError (most likely caller bug).
        """
        sql = (
            "SELECT link_id, view_observations, confidence, created_by, created_at "
            "FROM cross_view_links"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if t_start is not None:
            clauses.append("created_at >= ?")
            params.append(t_start)
        if t_end is not None:
            clauses.append("created_at <= ?")
            params.append(t_end)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        order_by = {
            None: "confidence DESC, created_at DESC",
            "confidence_desc": "confidence DESC, created_at DESC",
            "confidence_asc": "confidence ASC, created_at ASC",
            "created_at_desc": "created_at DESC, confidence DESC",
            "created_at_asc": "created_at ASC, confidence DESC",
        }
        if sort_by not in order_by:
            raise ValueError(
                f"unknown sort_by={sort_by!r}; expected one of "
                f"{[k for k in order_by if k is not None]}"
            )
        sql += " ORDER BY " + order_by[sort_by]

        # Validate view_id arg shape
        required_views: Optional[set[str]] = None
        if view_id is not None:
            if isinstance(view_id, str):
                required_views = {view_id}
            elif isinstance(view_id, (list, tuple, set)):
                if not view_id:
                    raise ValueError(
                        "view_id=[] is ambiguous (no filter? all views? "
                        "none?). Pass view_id=None to disable filtering, "
                        "or a non-empty list to require those views."
                    )
                if not all(isinstance(v, str) for v in view_id):
                    raise TypeError(
                        f"view_id list must contain only str, got {view_id!r}"
                    )
                required_views = set(view_id)
            else:
                raise TypeError(
                    f"view_id must be str / list[str] / None, "
                    f"got {type(view_id).__name__}"
                )

        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError(f"limit must be a non-negative int, got {limit!r}")
            # Apply SQL LIMIT only when view_id is unfiltered — otherwise
            # we'd lose post-filter results.
            if required_views is None:
                sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        result = [
            CrossViewLink(
                link_id=r[0],
                view_observations=[tuple(o) for o in json.loads(r[1])],
                confidence=r[2],
                created_by=r[3],
                created_at=r[4],
            )
            for r in rows
        ]

        if required_views is not None:
            result = [
                link for link in result
                if required_views.issubset(
                    {obs[0] for obs in link.view_observations}
                )
            ]
            if limit is not None:
                result = result[:limit]

        return result

    def query_telemetry(
        self,
        view_id: str,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> list[dict]:
        sql = (
            "SELECT t, view_id, gps_lat, gps_lon, alt, "
            "gimbal_qx, gimbal_qy, gimbal_qz, gimbal_qw, imu_extra "
            "FROM telemetry WHERE view_id = ?"
        )
        params: list[Any] = [view_id]
        if t_start is not None:
            sql += " AND t >= ?"
            params.append(t_start)
        if t_end is not None:
            sql += " AND t <= ?"
            params.append(t_end)
        sql += " ORDER BY t"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        out: list[dict] = []
        for r in rows:
            entry = {
                "t": r[0],
                "view_id": r[1],
                "gps_lat": r[2],
                "gps_lon": r[3],
                "alt": r[4],
                "gimbal_qx": r[5],
                "gimbal_qy": r[6],
                "gimbal_qz": r[7],
                "gimbal_qw": r[8],
            }
            if r[9]:
                entry.update(json.loads(r[9]))
            out.append(entry)
        return out

    # ---- §3.4 #8 ---------------------------------------------------------

    def human_correction_endpoint(
        self, link_id: str, decision: str, user_id: Optional[str] = None
    ) -> int:
        """🔌 §3.4 #8 — RPC endpoint for L7 corrections. M5 implements; M1 stub returns 501."""
        return 501
