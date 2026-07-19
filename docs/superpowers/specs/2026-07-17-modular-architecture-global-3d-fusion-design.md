# 模块化架构：多视角全局 3D 态势融合与理解 — 设计

日期：2026-07-17（2026-07-19 定稿：检测与跟踪分家、跟踪合并单+多视角、补 M2/M3 关系与任务清单）
状态：已批准，待写 Phase 0 实现计划
关联：`2026-07-10-omniuav-mva-integration-design.md`、`2026-07-12-query-conditioned-retrieval-design.md`

## 1. 目标

用多路无人机视角信息，形成对**全局 3D 的态势融合**——每个真实目标在**同一度量坐标系**里有一条世界坐标轨迹——进而完成：
**全局检测、全局跟踪、counting、空间关系问答、时空预测**（及其派生的事件/异常/流量/覆盖等态势任务）。

关键判断：这些任务需要的是**目标级 3D 定位**（多视角三角化），**不是**稠密重建。
故 **3DGS 不在关键路径上，降为可选后续**，不作为第一类模块。

## 2. 架构原则（让 5-6 人各自迭代而实现相互独立）

系统是围绕**共享世界模型**的生产者—消费者 DAG：`生产者 → 世界模型 → 消费者`。
模块效果可相互影响（更好的跟踪→更好的关系），但影响只经**数据**传播，实现彼此隔离。三道接缝：

1. **带版本的数据契约**：`contracts/` 的 pydantic 类型 + DuckDB/Chroma 表结构是唯一 API。改 schema 必须走 M6 owner + 升版本号。任何模块**不 import 其他模块内部实现**。
2. **Protocol 接口**：每个可换算法藏在 ABC/Protocol 后；换实现=换注入，不动调用方。
3. **每模块独立评测**：各自 benchmark + 指标，CI 单独跑；改算法只看本模块指标回归。

## 3. 六个模块（各 1 人）

> 每个模块 = (输入契约, 输出契约, Protocol 接口, 子任务, 评测指标, 依赖, 现有代码, 首里程碑)。
> **检测（M1）与跟踪（M3）分家**：两者是不同的可换算法领域，分开=换检测器不动跟踪器。
> **跟踪只设一人（M3），单视角 + 多视角都归他。**

### M1 检测 / 分割
- **职责**：逐视角、逐帧检测 / 分割（可开放词表 / 小目标 / 航拍）。系统的感知前端。
- **输入**：`frames`（世界模型 / 采集）。
- **输出**：`Detection`（view_id, t, bbox, class, conf, mask?）→ 世界模型。
- **Protocol**：`Detector.detect(frame) -> list[Detection]`；（可选）`Segmenter.segment(frame) -> list[Mask]`。
- **子任务**：闭集/开放词表检测、实例/语义分割、小目标增强、（可选）细粒度属性识别。
- **评测**：检测 mAP / mAP@小目标；分割 mIoU。
- **依赖**：无（最上游）。
- **现有**：`l1_perception`（YOLO/YOLOE/YOLO-World，Detector 雏形）。
- **首里程碑**：现有检测收敛到 `Detector` Protocol，产出标准 `Detection` 写世界模型。

### M2 跨视角几何与度量对齐
- **职责**：内外参 / 位姿、跨视角**时序同步**、极线约束、地面反投影、（可选）BEV。像素→世界坐标——**全局 3D 的地基**。
- **输入**：`frames` + 位姿元数据（AirSim 真值位姿起步；真实机用 GPS/IMU/飞控 + 视觉里程计）。
- **输出**：`CameraPose(view_id, t, intrinsics, extrinsics)`；`Projector`（像素↔世界射线、地面交点）→ 世界模型 `camera_poses`。
- **Protocol**：
  - `PoseProvider.pose(view_id, t) -> CameraPose`
  - `Projector.ray(view_id, pixel, t) -> Ray`（相机中心 + 方向，世界系）
  - `Projector.backproject(view_id, pixel, t, ground_z=0.0) -> WorldPoint`（射线与地平面求交）
  - `TimeSync.align(view_streams) -> list[SyncedFrameSet]`（同一时刻对齐 N 路）
- **子任务**：相机自定位（SLAM/VO）、高度/深度估计、BEV/占据栅格、覆盖与盲区分析（谁看得到哪）。
- **评测**：位姿 ATE/RPE；重投影误差(px)；地面反投影误差(m，用 AirSim 真值)。
- **依赖**：采集提供位姿元数据（M6/L0）。
- **现有**：无（`l2_crossview/geometric.py` 仅 bbox 几何配对雏形）。
- **首里程碑**：接入 AirSim 真值位姿 → `Projector.ray`+`backproject`，在 GT 上验证地面反投影误差。

### M3 多目标跟踪（单视角 + 多视角）—（系统中心）
- **职责**：**单视角 MOT**（消费 M1 检测 → `ViewTracklet`）→ **跨视角关联**同一物理目标（外观 + 极线）→ **三角化** 3D 位置（用 M2 射线）→ 跨时维护 **GlobalObject**（一个全局 ID + 世界 3D 轨迹）。**counting / 空间问答 / 预测全都建立在它之上。**
- **输入**：`Detection`（M1）、`CameraPose`/`Projector`（M2 契约）。
- **输出**：`ViewTracklet`（单视角轨迹，内部产物但落库供 M5）、`GlobalObject`、`GlobalObservation`、`GlobalTrajectory` → 世界模型。
- **Protocol**：
  - `Tracker.update(dets, h, w, frame=None) -> list[tuple[Detection, int]]`（单视角，已有）
  - `CrossViewAssociator.associate(view_tracklets_at_t, geometry) -> list[ObjectGroup]`（外观 cos + 极线距离）
  - `Triangulator.triangulate(rays) -> WorldPoint`（≥2 射线最小二乘；退化落地面反投影）
  - `GlobalTracker.step(groups_at_t, prev_state) -> GlobalState`（时序维护 global_id，卡尔曼/匈牙利）
- **子任务**：单视角 MOT、跨视角关联、全局 ID/ReID、3D 定位、目标接力（handoff）、冗余去重、（跨时）计数去重。
- **评测**：单视角 MOTA/IDF1；**全局 MOTA/IDF1（世界系）**；**counting MAE**；3D 位置误差(m)。
- **依赖**：M1（检测）、M2（位姿/投影）——**均经契约**，不 import 内部。
- **现有**：`l1_perception`（ByteTrack/IoU，单视角）+ `l2_crossview`（弱成对外观 link，需重写为全局注册表）。
- **首里程碑**：AirSim GT 位姿下，两视角单跟踪→关联→三角化产出 GlobalObject，counting 对上 GT。

### M4 时空关系与预测
- **职责**：在全局对象/轨迹上算**空间关系谓词**与**事件**，并做**时空预测**。
- **输入**：`GlobalObject`/`GlobalTrajectory`（M3）、`CameraPose`/区域定义（M2）。
- **输出**：`SceneGraphEdge`、`Event`、`Prediction` → 世界模型。
- **Protocol**：`RelationModeler.relations(objects_at_t) -> list[Relation]`（已有 ABC）；`EventDetector.detect(window) -> list[Event]`；`TrajectoryPredictor.predict(trajectory, horizon_s) -> list[WorldPoint]`。
- **子任务**：空间关系/场景图（near/left-of/approaching）、交互与群组/编队、区域推理（进入/离开/停留 dwell）、流量统计（越线 in/out）、行为/动作识别、事件检测（聚集/疏散/闯入/徘徊/碰撞）、异常检测、变化检测、轨迹/意图预测。
- **评测**：场景图 SGGen recall@k；事件/异常 F1；预测 ADE/FDE。
- **依赖**：M3（全局轨迹）、M2（坐标）。
- **现有**：`l3_events` + `perception/relation.py`（stub）。
- **首里程碑**：几何谓词（距离/方位阈值）关系 + 常速(CV)预测基线，跑通指标。

### M5 信息压缩与检索
- **职责**：语义 / visual-token 压缩（段级 + 目标级嵌入）；多视角检索。
- **输入**：`frames` + `ViewTracklet`/`GlobalObject`（M3，取目标裁剪）。
- **输出**：`Embedding` 向量（Chroma）+ 检索 API（`RetrieveResponse`）。
- **Protocol**：`Embedder/Compressor.encode(...)`；`ConstraintParser.parse(text)`（已有）；`Retriever.retrieve(req) -> RetrieveResponse`。
- **子任务**：跨视角检索、以文/图搜目标、指代定位（grounding：把"楼旁那辆红车"绑到 GlobalObject 的全局 ID）。
- **评测**：检索 recall@k / mAP；grounding 命中率。
- **依赖**：M3（目标供裁剪/全局 ID）——经契约。
- **现有**：`segmentation` + `l5_state` 嵌入 + `service/retrieval` + `query_understanding`（已建）。
- **首里程碑**：把检索绑到 `GlobalObject`（目标级命中全局 ID，而非仅段级）。

### M6 平台：世界模型 + 问答 + 评测
- **职责**：**拥有共享世界模型契约（schema）与存储**；采集/时序接入；sidecar 服务；问答/推理编排 + **空间问答工具**；OmniUAV UI；**跨模块评测框架** + AirSim GT 适配器。
- **输入**：一切（消费端）。
- **输出/拥有**：`contracts/` 类型、DuckDB/Chroma schema、`WorldStateStore`、`service`、`l6_interaction` 工具、UI、`eval` runner。
- **空间问答工具**（查全局 3D 态势而非 per-view 检测表）：`count_global(class, region?)`、`where_is(global_id/desc)`、`spatial_relation(a, b)`、`predict_where(global_id, horizon)`。
- **子任务**：态势问答（count/where/when/relation/predict）、多机一致性/置信度融合（consensus，化解冲突观测）、态势摘要/报告生成、评测框架 + GT 适配。
- **评测**：端到端空间问答准确率（标注 benchmark）。
- **依赖**：读所有模块产物。
- **现有**：`contracts`、`l5_state` 存储、`l6_interaction`、`service`、`omni-uav`、`cli/eval`、`datasets`。
- **首里程碑**：落 GlobalObject/CameraPose/… schema + 契约类型 + Protocol stub + AirSim GT 适配器——**解锁其余 5 人并行**。

## 4. M2 / M3 关系（为什么分得开）

- **M2 = 与 object 无关的场景几何**（相机在哪、像素怎么映射到世界）；**M3 = object 的身份 + 定位**（哪些观测是同一个、它在世界的哪）。二者**概念独立**。
- **M3 经契约依赖 M2**（`CameraPose`/`Projector`），不重造几何、不 import M2 内部。
- **优雅降级**：M2 未就位时，M3 可**纯外观(Re-ID)**先出全局 ID；M2 一到，叠加极线/世界位置约束提精度并三角化出 3D。→ **M3 不被 M2 阻塞，两人真并行。**
- **三角化是 M2×M3 的联合产物**：目标世界坐标需同时有「射线」(M2) 与「哪些射线是同一目标」(M3)。这是耦合点，但发生在契约边界上，不影响两模块各自迭代。

## 5. 核心数据契约（M6 拥有，`contracts/` + 世界模型表）

```python
Detection(view_id, t, bbox, class_name, conf, mask?)             # detections（M1→M3）
ViewTracklet(view_id, track_id, t_start, t_end, bboxes[], class) # view_tracklets（M3 单视角产物）
CameraPose(view_id, t, fx, fy, cx, cy, R|quat, translation)      # camera_poses（M2）
WorldPoint(x, y, z)                                              # 世界系(米)
GlobalObject(global_id, class_name, first_t, last_t, n_views, conf)   # global_objects（M3）
GlobalObservation(global_id, view_id, view_track_id, t, bbox, world_xyz)  # global_observations
GlobalTrajectory(global_id, t, x, y, z, vx?, vy?)               # global_trajectory
SceneGraphEdge(t, subj_global_id, rel, obj_global_id|region, conf)    # scene_graph_edges（M4）
Event(t_start, t_end, kind, global_ids[], region?, conf)        # events（M4）
Prediction(global_id, t_future, x, y, conf)                     # predictions（M4）
```

`GlobalObject` 是枢纽：**counting = 数去重后的 GlobalObject**；**空间问答 = 对其世界坐标做几何查询**；**预测 = 在其轨迹上外推**。
**契约归属**：`Detection` 是 M1→M3 的接口；`ViewTracklet` 由 M3 单视角阶段产出（落库供 M5 裁剪）。

## 6. 全局融合流水线

```
M1 检测/分割 ──Detection──▶ M3
M2 位姿/标定/时序同步 ──CameraPose/Projector──▶ M3
M3  单视角 MOT(ViewTracklet) → 跨视角关联(外观+极线) → 2+视角三角化 → GlobalObject(3D轨迹) → 世界模型
M4  在 GlobalTrajectory 上算 空间关系/事件/预测 ──▶ 世界模型
M5  frames + ViewTracklet/GlobalObject ──▶ 目标级/段级嵌入 ──▶ 检索
M6  空间问答/检索/UI/评测 只读世界模型
```

**唯一跨模块代码依赖**：M3←M1/M2、M4←M2/M3、M5←M3——全部经暴露的 Protocol/契约。

## 7. 依赖 DAG

```
采集/同步(M6) ──frames+pose_meta──▶ 世界模型
世界模型 ──frames──▶ M1 M2 M5
M1 ──Detection──▶ M3
M2 ──CameraPose/Projector(契约)──▶ M3 M4
M3 ──GlobalObject/Trajectory──▶ M4 M6 ；──ViewTracklet/全局ID──▶ M5
M4 ──SceneGraph/Event/Prediction──▶ M6
M6(问答/检索/UI) 只读世界模型
```

## 8. 分工与并行（契约先行）

- **Phase 0（解锁并行，M6 负责）**：世界模型 schema（第 5 节）+ 契约类型 + 各模块 Protocol stub + **AirSim GT 适配器**（真值位姿 + 目标 3D 位置）。落地后 6 个 owner 各自对着契约 + fake 数据并行开工。
- **Phase 1 关键路径（首个端到端全局 demo）**：M2（AirSim 真值位姿→Projector）→ M3（单跟踪+关联+三角化→GlobalObject）→ M6（counting + 空间问答工具）。M1 先用现有 YOLO；M4/M5 并行。
- 之后各模块沿自己的 eval 独立迭代。

## 9. AirSim 起步策略

实验数据是 AirSim：直接给**真值相机位姿 + 目标 3D 位置**。一箭双雕——M2 位姿先用仿真真值（不必先啃 SLAM/GPS 标定），M3/M4/预测的评测直接拿它当 GT。真实无人机再换 GPS/IMU + VO，**接口不变**。

## 10. 非目标 / 明确排除

- **3DGS 稠密重建**：不在核心 6 模块内；仅当需稠密几何/逼真可视化/遮挡先验时作可选后续（新增 `Reconstructor` Protocol 消费 M2 位姿即可，不影响其他模块）。
- **控制侧**（主动感知/下一最佳视角、视角调度/任务分配、威胁评估）：属无人机控制/决策，不在"理解"系统内，留接口不占模块。
- 不追求实时；离线本地库为主，近实时为辅。
- 本 spec 是**架构/分工边界文档**，每个模块后续各自走 spec→plan→实现循环。

## 11. 首个实现计划的范围（供 writing-plans — Phase 0 契约层）

在 `contracts/` 落第 5 节全部类型；在 `l5_state` 世界模型加对应表 + 读写方法；为各模块定义 Protocol
（`Detector`/`Tracker` 已有，新增 `Segmenter?`/`PoseProvider`/`Projector`/`TimeSync`/`CrossViewAssociator`/`Triangulator`/`GlobalTracker`/`EventDetector`/`TrajectoryPredictor`/`Retriever`）；
`datasets` 加 **AirSim GT 适配器**（位姿 + 3D GT）。全部 TDD，各带 fake/stub 实现，确保 6 模块可并行对接。**这一层只锁契约，不含具体算法。**
