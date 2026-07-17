# 模块化架构：多视角全局 3D 态势融合与理解 — 设计

日期：2026-07-17
状态：已批准，待写实现计划
关联：`2026-07-10-omniuav-mva-integration-design.md`、`2026-07-12-query-conditioned-retrieval-design.md`

## 1. 目标

用多路无人机视角信息，形成对**全局 3D 的态势融合**——每个真实目标在**同一度量坐标系**里有一条世界坐标轨迹——进而完成：
**全局目标检测、全局跟踪、counting、空间关系问答、时空预测**。

关键判断：这些任务需要的是**目标级 3D 定位**（多视角三角化），**不是**稠密重建。
故 **3DGS 不在关键路径上，降为可选后续**，不作为第一类模块。

## 2. 架构原则（让 5-6 人各自迭代而实现相互独立）

系统是围绕**共享世界模型**的生产者—消费者 DAG：`生产者 → 世界模型 → 消费者`。
模块效果可相互影响（更好的跟踪→更好的关系），但影响只经**数据**传播，实现彼此隔离。三道接缝：

1. **带版本的数据契约**：`contracts/` 的 pydantic 类型 + DuckDB/Chroma 表结构是唯一 API。改 schema 必须走 M6 owner + 升版本号。任何模块**不 import 其他模块内部实现**。
2. **Protocol 接口**：每个可换算法藏在 ABC/Protocol 后；换实现=换注入，不动调用方。
3. **每模块独立评测**：各自 benchmark + 指标，CI 单独跑；改算法只看本模块指标回归。

## 3. 六个模块

> 每个模块 = (输入契约, 输出契约, Protocol 接口, 评测指标, 依赖, 现有代码, 首里程碑)。

### M1 感知与单视角跟踪
- **职责**：检测/分割（可开放词表）→ 单视角多目标跟踪。产出每视角逐帧观测。
- **输入**：`frames`（世界模型 / 采集）。
- **输出**：`Detection`、`ViewTracklet`（view 内 track_id + bbox 序列 + class + conf）→ 世界模型。
- **Protocol**：`Detector.detect(frame) -> list[Detection]`；`Tracker.update(dets, h, w, frame=None) -> list[tuple[Detection, int]]`（已有）。
- **评测**：检测 mAP；单视角 MOTA / IDF1 / ID-switches。
- **依赖**：无（最上游）。
- **现有**：`l1_perception`（YOLO + ByteTrack/IoU）。
- **首里程碑**：把现有 detect+track 收敛到 `Detector`/`Tracker` Protocol 后，产出标准 `ViewTracklet` 写世界模型。

### M2 跨视角几何与度量对齐
- **职责**：相机内外参 / 位姿、跨视角**时序同步**、极线约束、地面反投影、（可选）BEV。把像素接到世界坐标——**全局 3D 的地基**。
- **输入**：`frames` + 位姿元数据（AirSim 真值位姿起步；真实无人机用 GPS/IMU/飞控 + 视觉里程计）。
- **输出**：`CameraPose(view_id, t, intrinsics, extrinsics)`；`Projector`（像素↔世界射线、地面交点）→ 世界模型 `camera_poses` 表。
- **Protocol**：
  - `PoseProvider.pose(view_id, t) -> CameraPose`
  - `Projector.ray(view_id, pixel, t) -> Ray`（相机中心 + 方向，世界系）
  - `Projector.backproject(view_id, pixel, t, ground_z=0.0) -> WorldPoint`（射线与地平面求交）
  - `TimeSync.align(view_streams) -> list[SyncedFrameSet]`（同一时刻对齐 N 路）
- **评测**：位姿 ATE/RPE；重投影误差(px)；地面反投影误差(m，用 AirSim 真值)。
- **依赖**：采集提供位姿元数据（M6/L0）。
- **现有**：无（`l2_crossview/geometric.py` 是雏形，仅 bbox 几何配对）。
- **首里程碑**：接入 AirSim 真值位姿 → 实现 `Projector.ray` + `backproject`，在 GT 上验证地面反投影误差。

### M3 全局融合与多目标跟踪 —（系统中心）
- **职责**：把各视角观测 + 几何合成**全局对象注册表**：跨视角关联同一物理目标（外观 + 极线）→ 2+ 视角射线**三角化**出 3D 位置 → 跨时维护 **GlobalObject**（一个全局 ID + 世界 3D 轨迹）。**counting / 空间问答 / 预测全都建立在它之上。**
- **输入**：`ViewTracklet`（M1）、`CameraPose`/`Projector`（M2 契约）。
- **输出**：`GlobalObject`、`GlobalObservation`、`GlobalTrajectory` → 世界模型。
- **Protocol**：
  - `CrossViewAssociator.associate(view_tracklets_at_t, geometry) -> list[ObjectGroup]`（同一目标的跨视角分组，可用外观 cos + 极线距离）
  - `Triangulator.triangulate(rays) -> WorldPoint`（≥2 射线最小二乘求交；退化时落地面反投影）
  - `GlobalTracker.step(groups_at_t, prev_state) -> GlobalState`（时序维护 global_id + 卡尔曼/匈牙利）
  - 组合成 `GlobalFuser.fuse(view_tracklets, geometry) -> list[GlobalObject]`
- **评测**：**全局 MOTA/IDF1（世界系）**；**counting MAE**；3D 位置误差(m)。
- **依赖**：M1（观测）、M2（位姿/投影）——**均经契约**，不 import 内部。
- **现有**：`l2_crossview`（弱成对外观 link，需重写为全局注册表）。
- **首里程碑**：AirSim GT 位姿下，两视角关联+三角化产出 GlobalObject，counting 对上 GT。

### M4 时空关系与预测
- **职责**：在全局对象/轨迹上算**空间关系谓词**（near / left-of / approaching / 进入区域）构建场景图；**时空预测**（轨迹/关系外推）。
- **输入**：`GlobalObject`/`GlobalTrajectory`（M3）、区域定义。
- **输出**：`SceneGraphEdge`（t, subj, rel, obj/region, conf）、`Prediction`（global_id, t_future, world_xy, conf）→ 世界模型。
- **Protocol**：
  - `RelationModeler.relations(objects_at_t) -> list[Relation]`（已有 ABC / `NullRelationModeler`）
  - `TrajectoryPredictor.predict(trajectory, horizon_s) -> list[WorldPoint]`
- **评测**：场景图 SGGen recall@k；预测 ADE/FDE。
- **依赖**：M3（全局轨迹）、M2（坐标）。
- **现有**：`l3_events` + `perception/relation.py`（stub）。
- **首里程碑**：几何谓词（距离/方位阈值）版关系 + 常速(CV)预测基线，跑通指标。

### M5 信息压缩与检索
- **职责**：语义嵌入 / visual-token 压缩（段级 + 目标级）；多视角检索（query 解析/视角时间过滤/排序）。
- **输入**：`frames` + `tracks`（取目标裁剪）。
- **输出**：`Embedding` 向量（Chroma）+ 检索 API（`RetrieveResponse`）。
- **Protocol**：`Embedder/Compressor.encode(...)`；`ConstraintParser.parse(text)`（已有）；`Retriever.retrieve(req) -> RetrieveResponse`。
- **评测**：检索 recall@k / mAP。
- **依赖**：M1（tracks 供裁剪）——经契约。
- **现有**：`segmentation` + `l5_state` 嵌入 + `service/retrieval` + `query_understanding`（已建）。
- **首里程碑**：把检索绑到 `GlobalObject`（目标级检索命中全局 ID，而非仅段级）。

### M6 平台：世界模型 + 问答 + 评测
- **职责**：**拥有共享世界模型契约（schema）与存储**；采集/时序接入；sidecar 服务；问答/推理编排 + **空间问答工具**；OmniUAV UI；**跨模块评测框架** + AirSim GT 适配器。
- **输入**：一切（消费端）。
- **输出/拥有**：`contracts/` 类型、DuckDB/Chroma schema、`WorldStateStore`、`service`、`l6_interaction` 工具、UI、`eval` runner。
- **空间问答工具**（新，查全局 3D 态势而非 per-view 检测表）：`count_global(class, region?)`、`where_is(global_id/desc)`、`spatial_relation(a, b)`、`predict_where(global_id, horizon)`。
- **评测**：端到端空间问答准确率（在标注 benchmark 上）。
- **依赖**：读所有模块产物。
- **现有**：`contracts`、`l5_state` 存储、`l6_interaction`、`service`、`omni-uav`、`cli/eval`、`datasets`。
- **首里程碑**：落 GlobalObject/CameraPose/SceneGraph schema + 契约类型 + Protocol stub + AirSim GT 适配器——**这一步解锁其余 5 人并行**。

## 4. 核心数据契约（M6 拥有，`contracts/` + 世界模型表）

新增/关键类型（pydantic + 对应 DuckDB 表）：

```python
CameraPose(view_id, t, fx, fy, cx, cy, R|quat, translation)      # camera_poses
WorldPoint(x, y, z)                                              # 世界系(米)
GlobalObject(global_id, class_name, first_t, last_t, n_views, conf)   # global_objects
GlobalObservation(global_id, view_id, view_track_id, t, bbox, world_xyz)  # global_observations
GlobalTrajectory(global_id, t, x, y, z, vx?, vy?)               # global_trajectory
SceneGraphEdge(t, subj_global_id, rel, obj_global_id|region, conf)    # scene_graph_edges
Prediction(global_id, t_future, x, y, conf)                     # predictions
```

`GlobalObject` 是枢纽：**counting = 数去重后的 GlobalObject**；**空间问答 = 对其世界坐标做几何查询**；**预测 = 在其轨迹上外推**。

## 5. 全局融合流水线

```
M1  per-view 检测+跟踪 ───────────────▶ ViewTracklet
M2  位姿/标定/时序同步 ───────────────▶ CameraPose + Projector(射线/地面反投影)
M3  外观+极线关联同一目标 ─▶ 2+视角射线三角化 ─▶ GlobalObject(3D轨迹) ─▶ 世界模型
M4  在 GlobalTrajectory 上算 空间关系/场景图/预测 ─▶ 世界模型
M5  frames+tracks ─▶ 目标级/段级嵌入 ─▶ 检索
M6  空间问答/检索/UI/评测 只读世界模型
```

**唯一跨模块代码依赖**：M3←M2、M4←M2/M3、M5←M1——全部经暴露的 Protocol/契约，不碰内部实现。

## 6. 依赖 DAG

```
采集/同步(M6) ──frames+pose_meta──▶ 世界模型
世界模型 ──frames──▶ M1 M2 M5
M2 ──CameraPose/Projector(契约)──▶ M3 M4
M1 ──ViewTracklet──▶ M3 M5
M3 ──GlobalObject/Trajectory──▶ M4 M5(目标级检索) M6
M4 ──SceneGraph/Prediction──▶ M6
M6(问答/检索/UI) 只读世界模型
```

## 7. 分工与并行（关键：契约先行）

- **Phase 0（解锁并行，M6 负责）**：世界模型 schema（第 4 节）+ 契约类型 + 各模块 Protocol stub + **AirSim GT 适配器**（真值位姿 + 目标 3D 位置）。落地后 6 个 owner 各自对着契约 + fake 数据并行开工。
- **Phase 1 关键路径（首个端到端全局 demo）**：M2（AirSim 真值位姿→Projector）→ M3（关联+三角化→GlobalObject）→ M6（counting + 空间问答工具）。M1 先用现有 YOLO；M4/M5 并行推进。
- 之后各模块沿自己的 eval 独立迭代。

## 8. AirSim 起步策略

实验数据是 AirSim：直接给**真值相机位姿 + 目标 3D 位置**。一箭双雕——M2 位姿先用仿真真值（不必先啃 SLAM/GPS 标定），M3/M4/预测的评测直接拿它当 GT。真实无人机再换 GPS/IMU + VO，**接口不变**。

## 9. 非目标 / 明确排除

- **3DGS 稠密重建**：不在核心 6 模块内；仅当需要稠密几何/逼真可视化/遮挡先验时作为可选后续（新增一个 `Reconstructor` Protocol 消费 M2 位姿即可，届时不影响其他模块）。
- 不追求实时；离线本地库为主，近实时为辅。
- 本 spec 是**架构/分工边界文档**，每个模块后续各自走 spec→plan→实现循环；不在此展开单模块实现细节。

## 10. 首个实现计划的范围（供 writing-plans）

**Phase 0 契约层**：在 `contracts/` 落第 4 节全部类型；在 `l5_state` 世界模型加对应表 + 读写方法；为 M1–M5 各定义 Protocol（`Detector`/`Tracker` 已有，新增 `PoseProvider`/`Projector`/`TimeSync`/`CrossViewAssociator`/`Triangulator`/`GlobalTracker`/`TrajectoryPredictor`）；`datasets` 加 AirSim GT 适配器（位姿 + 3D GT）。全部 TDD，各带 fake/stub 实现，确保 6 模块可并行对接。这一层不含具体算法，只锁契约。
