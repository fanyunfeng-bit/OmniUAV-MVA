# Phase 0 契约层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地「多视角全局 3D 态势融合」的共享契约层——世界模型数据类型 + 每模块 Protocol 接口 + AirSim GT 适配器——让 6 个模块 owner 能对着契约 + fake 并行开工。

**Architecture:** 只锁契约、不含算法。新增 pydantic 契约类型(几何/全局对象/时空)、三个模块的 Protocol 包(geometry/fusion/reasoning)各带 fake 桩、`WorldStateStore` 新表与读写方法、`datasets/airsim_gt` GT 适配器。复用既有 `l1_perception.Detection`、`perception.Track/Tracker`、`perception.relation.RelationModeler`,不重造。

**Tech Stack:** Python 3.10、pydantic v2(`BaseModel`/`Field`/`field_validator`)、DuckDB、pytest。

## Global Constraints

- MVA 引擎代码在 `mva/src/mva/`,测试在 `mva/tests/`;conda 环境 `mva`:`/home/fyf/miniconda3/envs/mva/bin/python`。
- 测试命令:`cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest <path> -q`;全量:`-m "not gpu"`(现基线 584 passed)。
- 契约用 **pydantic v2** `BaseModel`,放 `mva/src/mva/contracts/`,`contracts/__init__.py` 导出;类型测试放 `tests/contracts/`,Protocol/store 测试放 `tests/unit/`。
- Protocol 用 `@runtime_checkable Protocol`;观测/tracklet 形状用 `Any`(沿用 `l2_crossview/protocol.py` 先例)。
- **复用不重造**:`l1_perception.Detection`(检测)、`perception.Track`/`perception.pipeline.Tracker`(单视角轨迹/跟踪)、`contracts.ViewObservation`、`perception.relation.RelationModeler`。本层不新建 `Detection`/`ViewTracklet` 类型。
- 命名避让既有:已有 `contracts.Event`(dataclass)、`contracts.Anomaly`、`contracts.TrajectoryPrediction`;本层新类型用 `SituationEvent`、`GlobalPrediction`,不叫 `Event`/`TrajectoryPrediction`。
- 本层**只锁契约,不含具体算法**;每个 Protocol 配最小 fake 桩(可跑通、结果平凡)。
- 每次 commit 前 `git grep --cached -nE "sk-[A-Za-z0-9]{20,}"` 确认无密钥;commit 结尾附 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- 设计依据:`docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md`(§5 契约、§11 Phase 0 范围)。

---

### Task 1: 几何契约（M2）

**Files:**
- Create: `mva/src/mva/contracts/geometry.py`
- Modify: `mva/src/mva/contracts/__init__.py`
- Test: `mva/tests/contracts/test_geometry_contracts.py`

**Interfaces:**
- Produces:
  - `WorldPoint(x: float, y: float, z: float=0.0)`
  - `Ray(origin: WorldPoint, direction: tuple[float,float,float])`
  - `CameraPose(view_id, t, fx, fy, cx, cy, quat: tuple[4 floats], translation: tuple[3 floats])`；`quat` 必须长度 4。

- [ ] **Step 1: Write the failing test**

`mva/tests/contracts/test_geometry_contracts.py`:
```python
import pytest
from pydantic import ValidationError
from mva.contracts import WorldPoint, Ray, CameraPose


def test_world_point_defaults_z():
    p = WorldPoint(x=1.0, y=2.0)
    assert (p.x, p.y, p.z) == (1.0, 2.0, 0.0)


def test_ray_holds_origin_and_direction():
    r = Ray(origin=WorldPoint(x=0, y=0, z=10), direction=(0.0, 0.0, -1.0))
    assert r.origin.z == 10
    assert r.direction == (0.0, 0.0, -1.0)


def test_camera_pose_roundtrip():
    c = CameraPose(view_id="cam01", t=1.0, fx=600, fy=600, cx=320, cy=240,
                   quat=(0, 0, 0, 1), translation=(5, 6, 20))
    d = c.model_dump()
    assert d["view_id"] == "cam01" and d["translation"] == (5, 6, 20)


def test_camera_pose_quat_must_be_length_4():
    with pytest.raises(ValidationError):
        CameraPose(view_id="c", t=0, fx=1, fy=1, cx=0, cy=0,
                   quat=(0, 0, 1), translation=(0, 0, 0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/contracts/test_geometry_contracts.py -q`
Expected: FAIL (`ImportError: cannot import name 'WorldPoint'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/contracts/geometry.py`:
```python
"""M2 几何契约：世界点、射线、相机位姿。设计见
docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md §5。"""
from __future__ import annotations

from pydantic import BaseModel, field_validator


class WorldPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class Ray(BaseModel):
    origin: WorldPoint                          # 相机中心，世界系
    direction: tuple[float, float, float]       # 单位方向向量，世界系


class CameraPose(BaseModel):
    view_id: str
    t: float
    fx: float
    fy: float
    cx: float
    cy: float
    quat: tuple[float, float, float, float]     # world<-cam 旋转 (qx,qy,qz,qw)
    translation: tuple[float, float, float]     # 相机中心，世界系

    @field_validator("quat")
    @classmethod
    def _quat_len4(cls, v):
        if len(v) != 4:
            raise ValueError("quat must be length-4 (qx,qy,qz,qw)")
        return v
```

在 `mva/src/mva/contracts/__init__.py` 的 import 区加：
```python
from mva.contracts.geometry import CameraPose, Ray, WorldPoint
```
并把 `"CameraPose", "Ray", "WorldPoint"` 加进 `__all__`。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/contracts/test_geometry_contracts.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/contracts/geometry.py mva/src/mva/contracts/__init__.py mva/tests/contracts/test_geometry_contracts.py
git commit -m "feat(contracts): geometry types (WorldPoint/Ray/CameraPose) for M2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 全局对象契约（M3）

**Files:**
- Create: `mva/src/mva/contracts/global_state.py`
- Modify: `mva/src/mva/contracts/__init__.py`
- Test: `mva/tests/contracts/test_global_state_contracts.py`

**Interfaces:**
- Produces:
  - `GlobalObject(global_id, class_name, first_t, last_t, n_views>=1, confidence∈[0,1])`；`last_t>=first_t`。
  - `GlobalObservation(global_id, view_id, view_track_id, t, bbox: 4-tuple, world_xyz: 3-tuple|None)`
  - `GlobalTrajectory(global_id, t, x, y, z=0.0, vx=None, vy=None)`

- [ ] **Step 1: Write the failing test**

`mva/tests/contracts/test_global_state_contracts.py`:
```python
import pytest
from pydantic import ValidationError
from mva.contracts import GlobalObject, GlobalObservation, GlobalTrajectory


def test_global_object_valid():
    g = GlobalObject(global_id="g1", class_name="car", first_t=0.0, last_t=10.0,
                     n_views=2, confidence=0.8)
    assert g.n_views == 2


def test_global_object_rejects_last_before_first():
    with pytest.raises(ValidationError):
        GlobalObject(global_id="g", class_name="car", first_t=5.0, last_t=1.0,
                     n_views=1, confidence=0.5)


def test_global_object_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        GlobalObject(global_id="g", class_name="car", first_t=0, last_t=1,
                     n_views=1, confidence=1.5)


def test_observation_world_xyz_optional():
    o = GlobalObservation(global_id="g1", view_id="cam01", view_track_id="t1",
                          t=0.0, bbox=(1, 2, 3, 4))
    assert o.world_xyz is None


def test_trajectory_defaults():
    p = GlobalTrajectory(global_id="g1", t=0.0, x=5.0, y=6.0)
    assert (p.z, p.vx, p.vy) == (0.0, None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/contracts/test_global_state_contracts.py -q`
Expected: FAIL (`ImportError: cannot import name 'GlobalObject'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/contracts/global_state.py`:
```python
"""M3 全局对象契约：全局对象注册表 / 观测 / 轨迹。§5。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class GlobalObject(BaseModel):
    global_id: str
    class_name: str
    first_t: float
    last_t: float
    n_views: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _last_after_first(self):
        if self.last_t < self.first_t:
            raise ValueError("last_t must be >= first_t")
        return self


class GlobalObservation(BaseModel):
    global_id: str
    view_id: str
    view_track_id: str
    t: float
    bbox: tuple[float, float, float, float]
    world_xyz: Optional[tuple[float, float, float]] = None   # 未三角化时为 None


class GlobalTrajectory(BaseModel):
    global_id: str
    t: float
    x: float
    y: float
    z: float = 0.0
    vx: Optional[float] = None
    vy: Optional[float] = None
```

在 `contracts/__init__.py` 加：
```python
from mva.contracts.global_state import GlobalObject, GlobalObservation, GlobalTrajectory
```
并把三者加进 `__all__`。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/contracts/test_global_state_contracts.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/contracts/global_state.py mva/src/mva/contracts/__init__.py mva/tests/contracts/test_global_state_contracts.py
git commit -m "feat(contracts): global-object types (GlobalObject/Observation/Trajectory) for M3

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 时空契约（M4）

**Files:**
- Create: `mva/src/mva/contracts/spatiotemporal.py`
- Modify: `mva/src/mva/contracts/__init__.py`
- Test: `mva/tests/contracts/test_spatiotemporal_contracts.py`

**Interfaces:**
- Produces:
  - `SceneGraphEdge(t, subj_global_id, rel: str, obj: str, confidence∈[0,1])`
  - `SituationEvent(event_id, kind: str, t_start, t_end, global_ids: list[str]=[], region: str|None, confidence∈[0,1])`；`t_end>=t_start`。
  - `GlobalPrediction(global_id, t_future, x, y, confidence∈[0,1])`

- [ ] **Step 1: Write the failing test**

`mva/tests/contracts/test_spatiotemporal_contracts.py`:
```python
import pytest
from pydantic import ValidationError
from mva.contracts import SceneGraphEdge, SituationEvent, GlobalPrediction


def test_scene_graph_edge():
    e = SceneGraphEdge(t=1.0, subj_global_id="g1", rel="near", obj="g2", confidence=0.7)
    assert e.rel == "near"


def test_situation_event_defaults_and_validator():
    ev = SituationEvent(event_id="e1", kind="gathering", t_start=0.0, t_end=5.0,
                        confidence=0.6)
    assert ev.global_ids == [] and ev.region is None
    with pytest.raises(ValidationError):
        SituationEvent(event_id="e2", kind="x", t_start=5.0, t_end=1.0, confidence=0.5)


def test_global_prediction():
    p = GlobalPrediction(global_id="g1", t_future=2.0, x=10.0, y=11.0, confidence=0.5)
    assert (p.x, p.y) == (10.0, 11.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/contracts/test_spatiotemporal_contracts.py -q`
Expected: FAIL (`ImportError: cannot import name 'SceneGraphEdge'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/contracts/spatiotemporal.py`:
```python
"""M4 时空契约：场景图边 / 态势事件 / 全局点预测。§5。
命名避让既有 contracts.Event(dataclass) 与 contracts.TrajectoryPrediction。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SceneGraphEdge(BaseModel):
    t: float
    subj_global_id: str
    rel: str                       # near / left_of / approaching / inside_region ...
    obj: str                       # global_id 或 region 名
    confidence: float = Field(ge=0.0, le=1.0)


class SituationEvent(BaseModel):
    event_id: str
    kind: str                      # gathering/dispersal/intrusion/loitering/collision/anomaly/change
    t_start: float
    t_end: float
    global_ids: list[str] = []
    region: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")
        return self


class GlobalPrediction(BaseModel):
    global_id: str
    t_future: float
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)
```

在 `contracts/__init__.py` 加：
```python
from mva.contracts.spatiotemporal import GlobalPrediction, SceneGraphEdge, SituationEvent
```
并把三者加进 `__all__`。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/contracts/test_spatiotemporal_contracts.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/contracts/spatiotemporal.py mva/src/mva/contracts/__init__.py mva/tests/contracts/test_spatiotemporal_contracts.py
git commit -m "feat(contracts): spatiotemporal types (SceneGraphEdge/SituationEvent/GlobalPrediction) for M4

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: M2 geometry Protocol + fake

**Files:**
- Create: `mva/src/mva/geometry/__init__.py`, `mva/src/mva/geometry/protocol.py`, `mva/src/mva/geometry/fakes.py`
- Test: `mva/tests/unit/test_geometry_protocol.py`

**Interfaces:**
- Consumes: `CameraPose`, `Ray`, `WorldPoint`（Task 1）。
- Produces（Protocol）：
  - `PoseProvider.pose(view_id, t) -> CameraPose`
  - `Projector.ray(view_id, pixel, t) -> Ray`；`Projector.backproject(view_id, pixel, t, ground_z=0.0) -> WorldPoint`
  - `TimeSync.align(view_timestamps: dict[str, list[float]], tol=0.05) -> list[dict[str, float]]`
  - fake：`StaticPoseProvider(pose)`、`DownwardProjector`（垂直向下投影桩）、`NearestTimeSync`。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_geometry_protocol.py`:
```python
from mva.geometry import PoseProvider, Projector, TimeSync
from mva.geometry.fakes import StaticPoseProvider, DownwardProjector, NearestTimeSync
from mva.contracts import CameraPose, Ray, WorldPoint


_POSE = CameraPose(view_id="cam01", t=0.0, fx=600, fy=600, cx=320, cy=240,
                   quat=(0, 0, 0, 1), translation=(0, 0, 20))


def test_fakes_satisfy_protocols():
    assert isinstance(StaticPoseProvider(_POSE), PoseProvider)
    assert isinstance(DownwardProjector(StaticPoseProvider(_POSE)), Projector)
    assert isinstance(NearestTimeSync(), TimeSync)


def test_static_pose_provider_returns_pose():
    p = StaticPoseProvider(_POSE).pose("cam01", 0.0)
    assert p.translation == (0, 0, 20)


def test_downward_projector_backprojects_to_ground():
    proj = DownwardProjector(StaticPoseProvider(_POSE))
    wp = proj.backproject("cam01", (320.0, 240.0), 0.0, ground_z=0.0)
    assert isinstance(wp, WorldPoint) and wp.z == 0.0


def test_nearest_time_sync_groups_by_tolerance():
    sets = NearestTimeSync().align({"a": [0.0, 1.0], "b": [0.02, 1.03]}, tol=0.05)
    assert len(sets) == 2
    assert set(sets[0].keys()) == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_geometry_protocol.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'mva.geometry'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/geometry/protocol.py`:
```python
"""M2 几何 Protocol：位姿、投影、时序同步。换算法=换实现，不动调用方。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from mva.contracts import CameraPose, Ray, WorldPoint


@runtime_checkable
class PoseProvider(Protocol):
    def pose(self, view_id: str, t: float) -> CameraPose: ...


@runtime_checkable
class Projector(Protocol):
    def ray(self, view_id: str, pixel: tuple[float, float], t: float) -> Ray: ...
    def backproject(self, view_id: str, pixel: tuple[float, float], t: float,
                    ground_z: float = 0.0) -> WorldPoint: ...


@runtime_checkable
class TimeSync(Protocol):
    def align(self, view_timestamps: dict[str, list[float]],
              tol: float = 0.05) -> list[dict[str, float]]: ...
```

`mva/src/mva/geometry/fakes.py`:
```python
"""M2 桩实现：仅为 Phase 0 并行开工，几何不求准确。真实算法由 M2 owner 替换。"""
from __future__ import annotations

from mva.contracts import CameraPose, Ray, WorldPoint


class StaticPoseProvider:
    """恒返回同一个位姿（忽略 view_id/t）。"""
    def __init__(self, pose: CameraPose):
        self._pose = pose

    def pose(self, view_id: str, t: float) -> CameraPose:
        return self._pose


class DownwardProjector:
    """桩：把任意像素当作相机正下方一条竖直射线，与地平面 z=ground_z 求交。"""
    def __init__(self, pose_provider: StaticPoseProvider):
        self._pp = pose_provider

    def ray(self, view_id: str, pixel: tuple[float, float], t: float) -> Ray:
        cam = self._pp.pose(view_id, t)
        return Ray(origin=WorldPoint(x=cam.translation[0], y=cam.translation[1],
                                     z=cam.translation[2]),
                   direction=(0.0, 0.0, -1.0))

    def backproject(self, view_id: str, pixel: tuple[float, float], t: float,
                    ground_z: float = 0.0) -> WorldPoint:
        cam = self._pp.pose(view_id, t)
        return WorldPoint(x=cam.translation[0], y=cam.translation[1], z=ground_z)


class NearestTimeSync:
    """桩：以第一路的时间戳为锚，其余路各取 tol 内最近的一帧，凑成同刻集合。"""
    def align(self, view_timestamps: dict[str, list[float]],
              tol: float = 0.05) -> list[dict[str, float]]:
        if not view_timestamps:
            return []
        anchor_view = next(iter(view_timestamps))
        out: list[dict[str, float]] = []
        for at in view_timestamps[anchor_view]:
            group = {anchor_view: at}
            for v, ts in view_timestamps.items():
                if v == anchor_view:
                    continue
                near = [x for x in ts if abs(x - at) <= tol]
                if near:
                    group[v] = min(near, key=lambda x: abs(x - at))
            out.append(group)
        return out
```

`mva/src/mva/geometry/__init__.py`:
```python
from mva.geometry.protocol import PoseProvider, Projector, TimeSync

__all__ = ["PoseProvider", "Projector", "TimeSync"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_geometry_protocol.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/geometry/ mva/tests/unit/test_geometry_protocol.py
git commit -m "feat(geometry): M2 Protocols (PoseProvider/Projector/TimeSync) + fakes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: M3 fusion Protocol + fake

**Files:**
- Create: `mva/src/mva/fusion/__init__.py`, `mva/src/mva/fusion/protocol.py`, `mva/src/mva/fusion/fakes.py`
- Test: `mva/tests/unit/test_fusion_protocol.py`

**Interfaces:**
- Consumes: `Ray`, `WorldPoint`, `GlobalObject`（Task 1/2）。
- Produces（Protocol）：
  - `CrossViewAssociator.associate(view_tracklets_at_t: list[Any], geometry: Any) -> list[list[Any]]`（每个内层 list = 同一物理目标的观测组）
  - `Triangulator.triangulate(rays: list[Ray]) -> WorldPoint`
  - `GlobalTracker.step(groups_at_t: list[list[Any]], t: float) -> list[GlobalObject]`
  - fake：`SingletonAssociator`（每个观测各自成组）、`CentroidTriangulator`（取射线原点均值）、`CountingGlobalTracker`（每组一个 GlobalObject）。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_fusion_protocol.py`:
```python
from mva.fusion import CrossViewAssociator, Triangulator, GlobalTracker
from mva.fusion.fakes import SingletonAssociator, CentroidTriangulator, CountingGlobalTracker
from mva.contracts import Ray, WorldPoint, GlobalObject


def test_fakes_satisfy_protocols():
    assert isinstance(SingletonAssociator(), CrossViewAssociator)
    assert isinstance(CentroidTriangulator(), Triangulator)
    assert isinstance(CountingGlobalTracker(), GlobalTracker)


def test_singleton_associator_one_group_per_obs():
    groups = SingletonAssociator().associate(["a", "b", "c"], geometry=None)
    assert groups == [["a"], ["b"], ["c"]]


def test_centroid_triangulator_averages_ray_origins():
    r1 = Ray(origin=WorldPoint(x=0, y=0, z=10), direction=(0, 0, -1))
    r2 = Ray(origin=WorldPoint(x=4, y=2, z=10), direction=(0, 0, -1))
    wp = CentroidTriangulator().triangulate([r1, r2])
    assert (wp.x, wp.y, wp.z) == (2.0, 1.0, 10.0)


def test_counting_global_tracker_emits_one_object_per_group():
    objs = CountingGlobalTracker().step([["a", "b"], ["c"]], t=1.0)
    assert len(objs) == 2
    assert all(isinstance(o, GlobalObject) for o in objs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_fusion_protocol.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'mva.fusion'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/fusion/protocol.py`:
```python
"""M3 全局融合 Protocol：跨视角关联、三角化、时序全局跟踪。"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mva.contracts import GlobalObject, Ray, WorldPoint


@runtime_checkable
class CrossViewAssociator(Protocol):
    def associate(self, view_tracklets_at_t: list[Any],
                  geometry: Any) -> list[list[Any]]: ...


@runtime_checkable
class Triangulator(Protocol):
    def triangulate(self, rays: list[Ray]) -> WorldPoint: ...


@runtime_checkable
class GlobalTracker(Protocol):
    def step(self, groups_at_t: list[list[Any]], t: float) -> list[GlobalObject]: ...
```

`mva/src/mva/fusion/fakes.py`:
```python
"""M3 桩实现：Phase 0 并行用，关联/三角化不求准确。真实算法由 M3 owner 替换。"""
from __future__ import annotations

from typing import Any

from mva.contracts import GlobalObject, Ray, WorldPoint


class SingletonAssociator:
    """桩：不做跨视角关联，每个观测各自成一组。"""
    def associate(self, view_tracklets_at_t: list[Any], geometry: Any) -> list[list[Any]]:
        return [[obs] for obs in view_tracklets_at_t]


class CentroidTriangulator:
    """桩：取所有射线原点的均值当作 3D 位置（真实实现应做射线求交）。"""
    def triangulate(self, rays: list[Ray]) -> WorldPoint:
        if not rays:
            return WorldPoint(x=0.0, y=0.0, z=0.0)
        n = len(rays)
        sx = sum(r.origin.x for r in rays) / n
        sy = sum(r.origin.y for r in rays) / n
        sz = sum(r.origin.z for r in rays) / n
        return WorldPoint(x=sx, y=sy, z=sz)


class CountingGlobalTracker:
    """桩：每组发一个 GlobalObject，global_id 用序号。"""
    def step(self, groups_at_t: list[Any], t: float) -> list[GlobalObject]:
        return [
            GlobalObject(global_id=f"g{i}", class_name="unknown",
                         first_t=t, last_t=t, n_views=max(1, len(g)), confidence=0.5)
            for i, g in enumerate(groups_at_t)
        ]
```

`mva/src/mva/fusion/__init__.py`:
```python
from mva.fusion.protocol import CrossViewAssociator, GlobalTracker, Triangulator

__all__ = ["CrossViewAssociator", "GlobalTracker", "Triangulator"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_fusion_protocol.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/fusion/ mva/tests/unit/test_fusion_protocol.py
git commit -m "feat(fusion): M3 Protocols (CrossViewAssociator/Triangulator/GlobalTracker) + fakes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: M4 reasoning Protocol + fake

**Files:**
- Create: `mva/src/mva/reasoning/__init__.py`, `mva/src/mva/reasoning/protocol.py`, `mva/src/mva/reasoning/fakes.py`
- Test: `mva/tests/unit/test_reasoning_protocol.py`

**Interfaces:**
- Consumes: `GlobalTrajectory`, `SituationEvent`, `GlobalPrediction`（Task 2/3）；`RelationModeler`（既有 `mva/perception/relation.py`，本任务转出口不重定义）。
- Produces（Protocol）：
  - `EventDetector.detect(trajectories: list[GlobalTrajectory], t_window: tuple[float,float]) -> list[SituationEvent]`
  - `TrajectoryPredictor.predict(trajectory: list[GlobalTrajectory], horizon_s: float) -> list[GlobalPrediction]`
  - 从本包再导出既有 `RelationModeler`（M4 单一 import 入口）。
  - fake：`NullEventDetector`（返回 []）、`ConstantVelocityPredictor`（末两点外推）。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_reasoning_protocol.py`:
```python
from mva.reasoning import EventDetector, TrajectoryPredictor, RelationModeler
from mva.reasoning.fakes import NullEventDetector, ConstantVelocityPredictor
from mva.contracts import GlobalTrajectory, GlobalPrediction


def test_reexports_relation_modeler():
    assert RelationModeler is not None            # 从既有 perception.relation 转出口


def test_fakes_satisfy_protocols():
    assert isinstance(NullEventDetector(), EventDetector)
    assert isinstance(ConstantVelocityPredictor(), TrajectoryPredictor)


def test_null_event_detector_empty():
    assert NullEventDetector().detect([], (0.0, 10.0)) == []


def test_cv_predictor_extrapolates():
    traj = [GlobalTrajectory(global_id="g1", t=0.0, x=0.0, y=0.0),
            GlobalTrajectory(global_id="g1", t=1.0, x=2.0, y=0.0)]
    preds = ConstantVelocityPredictor().predict(traj, horizon_s=1.0)
    assert len(preds) == 1
    assert isinstance(preds[0], GlobalPrediction)
    assert abs(preds[0].x - 4.0) < 1e-6 and abs(preds[0].y - 0.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_reasoning_protocol.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'mva.reasoning'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/reasoning/protocol.py`:
```python
"""M4 时空推理 Protocol：事件检测、轨迹预测。关系建模复用既有 RelationModeler。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from mva.contracts import GlobalPrediction, GlobalTrajectory, SituationEvent


@runtime_checkable
class EventDetector(Protocol):
    def detect(self, trajectories: list[GlobalTrajectory],
               t_window: tuple[float, float]) -> list[SituationEvent]: ...


@runtime_checkable
class TrajectoryPredictor(Protocol):
    def predict(self, trajectory: list[GlobalTrajectory],
                horizon_s: float) -> list[GlobalPrediction]: ...
```

`mva/src/mva/reasoning/fakes.py`:
```python
"""M4 桩实现：Phase 0 并行用。真实算法由 M4 owner 替换。"""
from __future__ import annotations

from mva.contracts import GlobalPrediction, GlobalTrajectory, SituationEvent


class NullEventDetector:
    """桩：不检测任何事件。"""
    def detect(self, trajectories, t_window) -> list[SituationEvent]:
        return []


class ConstantVelocityPredictor:
    """桩：用轨迹末两点的速度做常速外推一个点。"""
    def predict(self, trajectory: list[GlobalTrajectory],
                horizon_s: float) -> list[GlobalPrediction]:
        if len(trajectory) < 2:
            return []
        a, b = trajectory[-2], trajectory[-1]
        dt = (b.t - a.t) or 1.0
        vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
        return [GlobalPrediction(
            global_id=b.global_id, t_future=b.t + horizon_s,
            x=b.x + vx * horizon_s, y=b.y + vy * horizon_s, confidence=0.5)]
```

`mva/src/mva/reasoning/__init__.py`:
```python
from mva.perception.relation import RelationModeler   # 复用既有 ABC，单一 M4 入口
from mva.reasoning.protocol import EventDetector, TrajectoryPredictor

__all__ = ["RelationModeler", "EventDetector", "TrajectoryPredictor"]
```

> 注：若 `mva/perception/relation.py` 未导出 `RelationModeler`，改为 `from mva.perception.relation import RelationModeler` 对应的实际类名（执行时确认；该文件已定义 `RelationModeler` ABC 与 `NullRelationModeler`）。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_reasoning_protocol.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/reasoning/ mva/tests/unit/test_reasoning_protocol.py
git commit -m "feat(reasoning): M4 Protocols (EventDetector/TrajectoryPredictor) + fakes; re-export RelationModeler

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: 世界模型 store — 位姿 + 全局对象表

**Files:**
- Modify: `mva/src/mva/l5_state/duckdb_store.py`（`_create_shared_tables` 加表 + 新增 insert/query 方法）
- Test: `mva/tests/unit/test_worldstate_global_tables.py`

**Interfaces:**
- Consumes: `CameraPose`、`GlobalObject`、`GlobalObservation`、`GlobalTrajectory`（Task 1/2）。
- Produces（`WorldStateStore` 方法）：
  - `insert_camera_pose(pose: CameraPose)` / `query_camera_poses(view_id: Optional[str]=None) -> list[dict]`
  - `insert_global_object(obj: GlobalObject)` / `query_global_objects() -> list[dict]`
  - `insert_global_observation(obs: GlobalObservation)` / `query_global_observations(global_id: Optional[str]=None) -> list[dict]`
  - `insert_global_trajectory(pt: GlobalTrajectory)` / `query_global_trajectory(global_id: str) -> list[dict]`

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_worldstate_global_tables.py`:
```python
from mva.l5_state.duckdb_store import WorldStateStore
from mva.contracts import CameraPose, GlobalObject, GlobalObservation, GlobalTrajectory


def _store():
    return WorldStateStore(":memory:")


def test_camera_pose_roundtrip():
    s = _store()
    s.insert_camera_pose(CameraPose(view_id="cam01", t=0.0, fx=600, fy=600,
                                    cx=320, cy=240, quat=(0, 0, 0, 1),
                                    translation=(1, 2, 20)))
    rows = s.query_camera_poses("cam01")
    assert len(rows) == 1 and rows[0]["tz"] == 20.0
    s.close()


def test_global_object_roundtrip():
    s = _store()
    s.insert_global_object(GlobalObject(global_id="g1", class_name="car",
                                        first_t=0, last_t=5, n_views=2, confidence=0.8))
    rows = s.query_global_objects()
    assert rows[0]["global_id"] == "g1" and rows[0]["n_views"] == 2
    s.close()


def test_global_observation_and_trajectory():
    s = _store()
    s.insert_global_observation(GlobalObservation(global_id="g1", view_id="cam01",
                                view_track_id="t1", t=0.0, bbox=(1, 2, 3, 4),
                                world_xyz=(5, 6, 0)))
    s.insert_global_trajectory(GlobalTrajectory(global_id="g1", t=0.0, x=5, y=6))
    assert s.query_global_observations("g1")[0]["wx"] == 5.0
    assert s.query_global_trajectory("g1")[0]["y"] == 6.0
    s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_worldstate_global_tables.py -q`
Expected: FAIL (`AttributeError: 'WorldStateStore' object has no attribute 'insert_camera_pose'`)

- [ ] **Step 3: Write minimal implementation**

在 `duckdb_store.py` 的 `_create_shared_tables`（`segments` 索引之后、方法结束前）追加：
```python
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
```

在类中新增方法（放在 `insert_cross_view_link` 附近的写方法区）：
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_worldstate_global_tables.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/l5_state/duckdb_store.py mva/tests/unit/test_worldstate_global_tables.py
git commit -m "feat(l5): world-model tables for camera_poses + global objects/observations/trajectory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: 世界模型 store — 场景图 / 事件 / 预测表

**Files:**
- Modify: `mva/src/mva/l5_state/duckdb_store.py`
- Test: `mva/tests/unit/test_worldstate_reasoning_tables.py`

**Interfaces:**
- Consumes: `SceneGraphEdge`、`SituationEvent`、`GlobalPrediction`（Task 3）。
- Produces：
  - `insert_scene_graph_edge(e: SceneGraphEdge)` / `query_scene_graph_edges(t: Optional[float]=None) -> list[dict]`
  - `insert_situation_event(ev: SituationEvent)` / `query_situation_events() -> list[dict]`
  - `insert_global_prediction(p: GlobalPrediction)` / `query_global_predictions(global_id: Optional[str]=None) -> list[dict]`

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_worldstate_reasoning_tables.py`:
```python
from mva.l5_state.duckdb_store import WorldStateStore
from mva.contracts import SceneGraphEdge, SituationEvent, GlobalPrediction


def test_scene_graph_and_event_and_prediction_roundtrip():
    s = WorldStateStore(":memory:")
    s.insert_scene_graph_edge(SceneGraphEdge(t=1.0, subj_global_id="g1", rel="near",
                                             obj="g2", confidence=0.7))
    s.insert_situation_event(SituationEvent(event_id="e1", kind="gathering",
                              t_start=0.0, t_end=5.0, global_ids=["g1", "g2"],
                              confidence=0.6))
    s.insert_global_prediction(GlobalPrediction(global_id="g1", t_future=2.0,
                               x=10.0, y=11.0, confidence=0.5))
    assert s.query_scene_graph_edges()[0]["rel"] == "near"
    assert s.query_situation_events()[0]["kind"] == "gathering"
    assert s.query_global_predictions("g1")[0]["x"] == 10.0
    s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_worldstate_reasoning_tables.py -q`
Expected: FAIL (`AttributeError: ... 'insert_scene_graph_edge'`)

- [ ] **Step 3: Write minimal implementation**

在 `_create_shared_tables` 追加：
```python
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
```

新增方法（`json` 已在文件顶部 import）：
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_worldstate_reasoning_tables.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/l5_state/duckdb_store.py mva/tests/unit/test_worldstate_reasoning_tables.py
git commit -m "feat(l5): world-model tables for scene_graph_edges + situation_events + global_predictions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: AirSim GT 适配器

**Files:**
- Create: `mva/src/mva/datasets/airsim_gt.py`
- Test: `mva/tests/unit/test_airsim_gt_adapter.py`

**Interfaces:**
- Consumes: `CameraPose`、`GlobalObject`、`WorldPoint`（Task 1/2）。
- Produces：
  - GT JSON 格式：`{"cameras":[{view_id,t,fx,fy,cx,cy,quat[4],translation[3]}...], "objects":[{global_id,class_name,t,world[3]}...]}`
  - `AirSimGT(path)`；`.camera_poses() -> list[CameraPose]`；`.object_positions() -> list[tuple[GlobalObject, WorldPoint]]`（评测 GT：每个真值目标 + 其世界坐标）。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_airsim_gt_adapter.py`:
```python
import json
from mva.datasets.airsim_gt import AirSimGT
from mva.contracts import CameraPose, GlobalObject, WorldPoint


def _write_gt(tmp_path):
    gt = {
        "cameras": [{"view_id": "cam01", "t": 0.0, "fx": 600, "fy": 600,
                     "cx": 320, "cy": 240, "quat": [0, 0, 0, 1],
                     "translation": [1, 2, 20]}],
        "objects": [{"global_id": "obj1", "class_name": "car", "t": 0.0,
                     "world": [5.0, 6.0, 0.0]}],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    return str(p)


def test_camera_poses_parsed(tmp_path):
    a = AirSimGT(_write_gt(tmp_path))
    poses = a.camera_poses()
    assert len(poses) == 1
    assert isinstance(poses[0], CameraPose)
    assert poses[0].translation == (1, 2, 20)


def test_object_positions_parsed(tmp_path):
    a = AirSimGT(_write_gt(tmp_path))
    objs = a.object_positions()
    assert len(objs) == 1
    obj, wp = objs[0]
    assert isinstance(obj, GlobalObject) and isinstance(wp, WorldPoint)
    assert obj.global_id == "obj1" and (wp.x, wp.y) == (5.0, 6.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_airsim_gt_adapter.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'mva.datasets.airsim_gt'`)

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/datasets/airsim_gt.py`:
```python
"""AirSim 真值适配器：读真值相机位姿 + 目标 3D 位置。

既做 M2 的位姿来源起步（不必先啃 SLAM/GPS 标定），又当 M3/M4/预测的评测 GT。
GT JSON 格式见本模块 docstring / plan Task 9。真实无人机换 GPS/IMU+VO 时，
只需另写一个提供相同 CameraPose 契约的适配器，下游不变。
"""
from __future__ import annotations

import json
from pathlib import Path

from mva.contracts import CameraPose, GlobalObject, WorldPoint


class AirSimGT:
    def __init__(self, path: str):
        self._data = json.loads(Path(path).read_text())

    def camera_poses(self) -> list[CameraPose]:
        out = []
        for c in self._data.get("cameras", []):
            out.append(CameraPose(
                view_id=c["view_id"], t=float(c["t"]),
                fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
                quat=tuple(c["quat"]), translation=tuple(c["translation"]),
            ))
        return out

    def object_positions(self) -> list[tuple[GlobalObject, WorldPoint]]:
        out = []
        for o in self._data.get("objects", []):
            t = float(o["t"])
            obj = GlobalObject(global_id=o["global_id"], class_name=o["class_name"],
                               first_t=t, last_t=t, n_views=1, confidence=1.0)
            wx, wy, wz = o["world"]
            out.append((obj, WorldPoint(x=wx, y=wy, z=wz)))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_airsim_gt_adapter.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add mva/src/mva/datasets/airsim_gt.py mva/tests/unit/test_airsim_gt_adapter.py
git commit -m "feat(datasets): AirSim GT adapter (ground-truth poses + object 3D positions)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: 全量回归 + 文档收尾

**Files:**
- Modify: `MODIFICATIONS.md`

**Interfaces:** 无新接口，仅回归与文档。

- [ ] **Step 1: 跑 MVA 全量(非 GPU)确认无回归**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest -m "not gpu" -q`
Expected: PASS（≥ 584 + 本计划新增用例，全绿）

- [ ] **Step 2: 更新 MODIFICATIONS.md**

在 `MODIFICATIONS.md` 的「## 当前使用速览」之前插入：
```markdown
## 10. Phase 0 契约层（多视角全局 3D 态势融合）
设计见 `docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md`；
计划见 `docs/superpowers/plans/2026-07-19-phase0-contract-layer.md`。
只锁契约、不含算法，解锁 6 模块并行：
- 契约类型（`contracts/`）：几何 `WorldPoint/Ray/CameraPose`；全局对象 `GlobalObject/GlobalObservation/GlobalTrajectory`；时空 `SceneGraphEdge/SituationEvent/GlobalPrediction`。
- Protocol + fake 桩：`geometry/`（PoseProvider/Projector/TimeSync）、`fusion/`（CrossViewAssociator/Triangulator/GlobalTracker）、`reasoning/`（EventDetector/TrajectoryPredictor，复用 RelationModeler）。
- 世界模型表（`l5_state/duckdb_store`）：camera_poses、global_objects/observations/trajectory、scene_graph_edges、situation_events、global_predictions。
- AirSim GT 适配器（`datasets/airsim_gt`）：真值位姿 + 目标 3D 位置（M2 起步 + 评测 GT）。
```

- [ ] **Step 3: Commit + push**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo SECRET-ABORT || true
git add MODIFICATIONS.md
git commit -m "docs: record Phase 0 contract layer (MODIFICATIONS)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

## 交付后：6 模块如何并行

Phase 0 合入后，每个 owner 对着契约 + fake 起步、各写各的 spec→plan→实现：
- **M1 检测**：把现有 YOLO 收敛到 `Detector`，产 `Detection`。
- **M2 几何**：用 `AirSimGT` 真值位姿实现真正的 `Projector.ray/backproject`，替换 `geometry/fakes`。
- **M3 融合（中心）**：实现 `CrossViewAssociator`+`Triangulator`+`GlobalTracker`，产 `GlobalObject` 写世界模型。
- **M4 关系/预测**：在 `GlobalTrajectory` 上实现 `RelationModeler`/`EventDetector`/`TrajectoryPredictor`。
- **M5 压缩/检索**：把检索绑到 `GlobalObject`（目标级命中全局 ID）。
- **M6 平台**：加空间问答工具（`count_global`/`where_is`/`spatial_relation`/`predict_where`）读世界模型；评测框架用 `AirSimGT` 当 GT。

**关键路径**：M2（真值位姿→Projector）→ M3（关联+三角化→GlobalObject）→ M6（counting + 空间问答），先跑通首个端到端全局 demo。
