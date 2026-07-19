# 变更记录 (MODIFICATIONS)

本文件汇总 OmniUAV-MVA 集成过程中的所有改动，便于查验与回溯。
每条带 commit 短哈希；代码内同时用 `# [MOD 日期]` 注释标注。
设计依据见 `docs/superpowers/specs/2026-07-10-omniuav-mva-integration-design.md`；
实现计划见 `docs/superpowers/plans/`。

> 仓库：`git@github.com:fanyunfeng-bit/OmniUAV-MVA.git`（monorepo：`omni-uav/` 前端 + `mva/` 引擎）。
> ⚠️ 安全：DashScope API key 只存本地 `omni-uav/configs/config_llm.yaml`/`config.json`（gitignore，不入库）；启动脚本运行时自动读取。

---

## 0. 仓库搭建
- `2f2ed52` 初始化 monorepo：拷入 omni-uav 前端 + MVA 源码(`mva/`) + 设计文档；`.gitignore` 屏蔽密钥/权重/数据/媒体。

## 1. P0 —— MVA sidecar 骨架 + grounded 问答
架构：MVA 以本地 sidecar(FastAPI :8900)运行，独占世界状态 DuckDB+ChromaDB；OmniUAV 经 RPC 调用。问答统一走 sidecar(引擎掉线本地降级)，指令(跟踪/检测/重建)留在 OmniUAV。
- `ceff5a0` service 依赖 + RPC 模型(`mva/service/models.py`) + `EngineProtocol`
- `f642e7e` FastAPI 工厂 + `/health`
- `3d94f9a` `POST /answer`
- `9766a78` `/ingest/{start,status,stop}`
- `5238cba` **DashScope 云端 LLM 适配器**(`mva/l4_llm/cloud_client.py`, qwen3-vl-plus)
- `311e9bd` `QueryService` 支持注入生成 LLM(用云端 LLM 做问答)
- `fe92635` `AnalysisEngine`(QueryService + 入库任务表) + uvicorn 入口
- `0f4045a` OmniUAV `MvaClient`(RPC 客户端)
- `bf624f9` OmniUAV 问答分支路由 sidecar + 本地降级 + 引擎状态灯
- `e755ccc` sidecar 启停脚本(`scripts/start_mva_sidecar.sh` / `stop_mva_sidecar.sh`)

## 2. 进程内入库 + GUI 触发
- `463cc09` **进程内 ingest**：复用 sidecar 已加载的 store/embedder/vstore，避开 DuckDB 跨进程锁、免重载 16G 嵌入。
- `834962a` OmniUAV **"入库到分析引擎"按钮** + 状态轮询(选文件夹显示 / 是否入库 解耦)。
- `c68f573` 启动脚本**自动从本地配置读 DASHSCOPE_API_KEY**，免每次手动 export。

## 3. 设计文档更新
- `54b13e2` spec 增补：**双流解耦(D10)**(语义流粗/感知流密分开取帧)、**离线优先(D11)**、检索粒度(§3.3, top-3/缩略图top-1/帧级)。
- `b92e6c8` P1 实现计划(检索面板 + 感知流接口)。

## 4. P1 —— 多视角检索面板 + 感知流接口
- `ba3d44b` 检索 RPC 模型(`RetrieveRequest/Hit/Response`)
- `9bdbe8f` 检索纯逻辑(解析 chroma 命中 + 从 DuckDB 富化段时间，处理 `scene::view` 前缀)
- `a34e7ca` 抽帧缩略图 helper
- `e735749` `/retrieve` 端点 + `Engine.retrieve`(段级 + top-1 缩略图)
- `9da4487` `MvaClient.retrieve`
- `8917882` **多视角检索面板**(top-3 + top-1 缩略图 + 透明化 + 跳帧信号)
- `4a62587` `camera_tab.seek_to`(检索命中跳帧)
- `edcc8dc` 感知流接口：`FrameSource` + `UniformFrameSource`(密集取帧基线)
- `2b0a36f` 感知流接口：`Tracker`/`PerceptionPipeline`/`RelationModeler` + 基线(留口，未接入)

## 5. 数据采集
- `1c6a9f6` `scripts/record_airsim_4view.py`：经 rosbridge 录 4 无人机视角为本地 mp4。
  已采：`~/OmniUAV-MVA-data/airsim_downtown_4view/cam01-04.mp4`(3 分钟, 640×480, ~3.26fps；drone2 曾卡住→cam02 近静态)。

## 6. 存储/易用性
- `dbef5c7` sidecar 默认库改为持久 `~/.omniuav-mva`(重启不丢)；后续见 §8。
- `82ff738` **按 scene 名自动分独立库**：`<库根>/<scene>/{world.duckdb,chroma}`；`QueryService` 支持注入已加载 embedder(切库复用免重载)；`select_scene` 秒切；`/select_scene` 端点 + OmniUAV 选文件夹自动切库。

## 7. 修复
- `630fa75` **视频按原生 fps 播放**：`VideoStream` 按视频自身帧率节流(原固定 25fps 把 3.26fps 视频放快 ~7.7×)；缓冲仅收真正前进的帧。
- `e4adf46` **无入库也能问答**：
  - OmniUAV 把当前/时序帧作为附件一起发给 sidecar `/answer`(空世界状态也能看当前画面作答)。
  - 修 `engine.answer` 崩溃：`Attachment.path` 需 `pathlib.Path`(取 `.name`)，原传 str 触发 `AttributeError`。
  - OmniUAV 默认数据目录改为 `~/OmniUAV-MVA-data/airsim_downtown_4view`(不存在回退 examples)。

## 8. Query 条件化检索
设计见 `docs/superpowers/specs/2026-07-12-query-conditioned-retrieval-design.md`，
计划见 `docs/superpowers/plans/2026-07-12-query-conditioned-retrieval.md`。
检索时先从 query 解析「视角/时间段」约束，用剩余语义文本嵌入、用约束做 chroma 硬过滤(空则回退全库)。
- `51bd112` 规则约束解析器(视角/时间正则 + 剩余语义文本剥离，`mva/service/query_understanding.py`)。
- `d2675b7` LLM 兜底 + 混合解析器(`HybridConstraintParser`：规则优先，未命中且含指代才调云端)。
- `b1e5589` 约束→where 纯函数(`resolve_view_ref` 视角数字解析 / `resolve_time` 相对末尾换算 / `build_metadata_where`，`mva/service/retrieval.py`)。
- `223a099` `VectorStore.query` 通用 `where` 逃生口(与内建子句 `$and` 合并，不二次嵌套，`mva/l5_state/chromadb_store.py`)。
- `61cdcec` `RetrieveResponse.applied` 透明化模型(视角/时间/语义/来源/回退，`mva/service/models.py`)。
- `5e4fa3b` `AnalysisEngine.retrieve` 串联：硬过滤 + 空回退 + `applied`；`_scene_views`/`_scene_duration`(按 scene 缓存，切库/入库失效)。
- `c949c7d` bbox 向量补 `start_t/end_t`(前瞻目标级时间过滤，需重新入库生效)。
- `e5eda81` OmniUAV 检索面板展示 `applied`(如 `已限定 视角 cam01 · 语义"黄车"(规则)`；回退时提示扩展全库)。

## 9. 修复:视角限定问答弃答(view_id 前缀不匹配)
- `9260e1a` **view-scoped 检索按 raw 或前缀 view_id 都能命中**。
  根因:入库把 chroma 元数据写成 `view_id=<scene>::<view>`(前缀) + `view_id_raw=<view>`(裸),
  但调用方(planner/`look_at`/`find_by_description`)传的是**裸** view_id(如 `cam02`)。
  `VectorStore.query` 的 `view_id` 过滤只匹配前缀元数据 → 裸 id 匹配 0 条 →
  问"描述视角2里的内容"时 `look_at` 以 `no_segment` 弃答(而不带 view_id 的
  `find_segment_by_description` 却能返回 cam02 段——正是这一差异定位了根因)。
  修复:`_build_where` 的 view_id 子句改为 `$or([view_id, view_id_raw])`,两种写法都命中。
  真实库验证:`query(view_id="cam02")` 由 0 → 4 段;端到端 `/answer 描述视角2里的内容`
  现返回对 cam02 画面的实际描述(城市街景/黄色出租车/公交站台/现代建筑)。

## 11. Phase 0 契约层（多视角全局 3D 态势融合）
设计见 `docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md`；
计划见 `docs/superpowers/plans/2026-07-19-phase0-contract-layer.md`。
**只锁契约、不含算法，纯增量**（现有功能全绿：584 → 614 passed），解锁 6 模块并行：
- `abb0131`/`0cdb402`/`0b03876` 契约类型（`contracts/`）：几何 `WorldPoint/Ray/CameraPose`；全局对象 `GlobalObject/GlobalObservation/GlobalTrajectory`；时空 `SceneGraphEdge/SituationEvent/GlobalPrediction`。
- `0359257`/`947de90`/`b48278b` Protocol + fake 桩：`geometry/`（PoseProvider/Projector/TimeSync）、`fusion/`（CrossViewAssociator/Triangulator/GlobalTracker）、`reasoning/`（EventDetector/TrajectoryPredictor，复用 RelationModeler）。
- `2a87647`/`75b86e9` 世界模型表（`l5_state/duckdb_store`，只增不改老表）：camera_poses、global_objects/observations/trajectory、scene_graph_edges、situation_events、global_predictions。
- `dc026ce` AirSim GT 适配器（`datasets/airsim_gt`）：真值位姿 + 目标 3D 位置（M2 起步 + 评测 GT）。
- 复用不重造：`l1_perception.Detection`、`perception.Track/Tracker`、`perception.relation.RelationModeler`。

## 12. 接口补齐 + 交接材料
- `5d8be78` **M1 检测接口**：`mva/detection/`（`ObjectDetector`/`Segmenter` Protocol + 桩）；现有 `l1_perception.Detector` 结构上满足。
- `c84f685` **M5 检索接口**：`mva/retrieval/`（`Embedder`/`Retriever` Protocol + 桩）；现有 `MultimodalEmbedder` 满足。
- 至此 M1–M6 均有一致的「Protocol + fake 桩」接缝（M2/M3/M4 来自 Phase 0）。纯接口、不含算法，回归 614 → 621 passed。
- **交接文档** `MODULE_OWNERS.md`：完整项目=git 仓库（232 文件，`git clone` 交接、勿拷工作目录以免带出密钥）；环境搭建；6 模块 owner 卡片（接口/现有代码/契约/表/指标/起步点）；并行开发约定；关键路径 M2→M3→M6。

---

## 当前使用速览
```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
bash scripts/start_mva_sidecar.sh          # 自动读 key + 默认持久库 ~/.omniuav-mva
cd omni-uav && DISPLAY=:0.0 /home/fyf/miniconda3/envs/simsys/bin/python app.py
```
- 镜头 tab：选文件夹→按原生帧率显示 + sidecar 切到该 scene 库；"入库"→写入该 scene 独立库。
- 问答：没入库→模型看当前帧作答；入库后→grounded 从世界状态作答。
- 检索 tab：文字查询→top-3 + top-1 缩略图 + 点击跳帧。
- 停止：`bash scripts/stop_mva_sidecar.sh`

## 测试
- MVA：`cd mva && <mva-env>/python -m pytest -m "not gpu"`（621 passed）
- OmniUAV：`cd omni-uav && QT_QPA_PLATFORM=offscreen <simsys>/python -m pytest tests/`（13 passed）
