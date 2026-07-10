# 设计文档：OmniUAV × Multi-Video-Analysis 集成 —— 多视角视频态势感知理解系统

- 日期：2026-07-10
- 状态：已评审通过（brainstorming），待写实现计划
- 作者：Claude + 用户
- 相关系统：
  - 宿主前端 `OmniUAV`：`/home/fyf/fyf/PCL/Simulation-System/omni-uav`（PyQt5，conda env `simsys`）
  - 能力引擎 `MVA`：`/home/fyf/fyf/PCL/Multi-Video-Analysis`（Python 包 `mva`，独立 env）
  - 仿真：AirSim/UE4 DownTown + ROS(容器 `uav_sim_live`) + ego-planner，详见 `../../MODIFICATIONS.md`

---

## 1. 背景与目标

已有两套独立系统：

- **OmniUAV**：多无人机实时视频前端（PyQt5）。有多路镜头网格、FrameBuffer、rosbridge 实时流、云端 VLM 问答（DashScope Qwen3-VL-Plus）、TSDF 重建 tab、benchmark 评估 tab。缺：结构化世界状态、跨视角身份、检索、可溯源问答。
- **MVA**：一套 L0–L7 的**批处理**多视角视频理解管线（检测→跟踪→跨视角关联→多模态嵌入→DuckDB+ChromaDB 世界状态→检索增强 QA agent）。有干净的库门面 `QueryService`。缺：真实时接入、UI、3D/几何空间理解。

**最终目标**：构造一个**多视角视频态势感知理解系统**，能高效、精准地对多路视频内容进行理解与问答。**交付形态 = OmniUAV（宿主/唯一界面）承载各功能模块**，MVA 作为其隐形分析引擎被集成进来；不再维护两个面向用户的系统。

**要展示的功能模块**（用户诉求）：跨视角多目标跟踪、多视角信息检索、时空关系建模、3D 空间理解 QA；视觉信息压缩（visual-token）列为未来。

## 2. 范围

### 目标（In scope）
- 把 MVA 的世界状态存储 + 跨视角关联 + 检索 + 可溯源 QA 集成进 OmniUAV。
- OmniUAV 新增分析面板：多视角检索、跨视角跟踪、空间关系理解；升级右侧问答为 grounded 问答。
- 两种运行模式：**离线（主力：存视频/开源 benchmark，用于测模块+demo）** 与 **近实时滚动窗口（偶尔：实时可行性验证）**。
- 新增"空间关系理解"能力链：geo-grounding → 时空关系建模 → 3D 空间理解 QA（MLLM）。

### 非目标（Out of scope / 未来）
- **视觉信息压缩（visual-token 剪枝/合并）**：因问答走云端 API，本地 token 压缩无收益 → **完全不做，列为未来**。
- **照片级/拼接式 BEV 地图**：不做（多无人机视角拼接不稳、不准）。空间关系可视化只用**关系列表/场景图**。
- **BEV 示意散点图**：本轮不做。
- 3D 几何**重建**（TSDF）：不作为 3D 方向；OmniUAV 现有 `场景重建` tab 保持原样、不投入。
- Reranker、本地问答模型、HITL、流式输出、关系向量检索深化：未来。
- MM-UAVBench 评估独立保留（现有 EvaluationTab，不动）。

## 3. 关键决策（已锁定）

| # | 决策 | 理由 |
|---|---|---|
| D1 | **宿主 = OmniUAV**；MVA 作为能力来源被集成 | 用户要求单一界面/系统 |
| D2 | **集成架构 = Sidecar 服务(方案 A)**：MVA 在独立进程/环境运行，OmniUAV 经本地 RPC 调用 | 隔离重依赖(cu126 torch/chromadb/duckdb/boxmot)+GPU，避免依赖地狱，重活不进 Qt UI 线程，复用 MVA 已测管线 |
| D3 | **MVA 独占 DuckDB+ChromaDB**；OmniUAV 可视化数据也走 RPC | DuckDB 跨进程锁：不能"一进程读写 + 另一进程只读"并存。单一数据源、零锁冲突 |
| D4 | **问答模型 = 云端 DashScope `qwen3-vl-plus`** | 已有、质量高、省显存；实测 key 可用 |
| D5 | **嵌入模型 = 本地 `Qwen/Qwen3-VL-Embedding-8B`**（768 维 MRL）；不用云端 `multimodal-embedding-v1`、不用 reranker | 用户最终选定；检索为初步功能"能用就行" |
| D6 | **运行模式**：离线为主 + 近实时滚动为辅 | 用户日常=离线测模块/demo/benchmark，偶尔实时 |
| D7 | **图像跨进程走共享文件系统路径**（`/tmp/mva_frames`），RPC 只传路径 | 同机、免 base64 膨胀，复用 OmniUAV 现有 `_save_frame` |
| D8 | **空间关系理解**（geo-grounding + 关系建模 + 3D QA）为核心新增能力 | 用户明确要做几何关系问答，非轨迹预测 |
| D9 | **所有模块"基线先用、留好接口"**：每个功能模块先上基础方法（能用即可），但都置于统一接口/契约之后，后续可零改动上层地换更先进方法 | 用户要求：现在用基础方法，接口预留，后续更新更先进的方法 |

> ⚠️ 安全遗留：DashScope API key 目前明文写在 `omni-uav/configs/config_llm.yaml` 等文件（见 `MODIFICATIONS.md`）。建议改环境变量并轮换，勿提交到公共仓库。

### 3.1 模块可插拔原则（D9 展开：基线先用、留好接口）

**两道天然的稳定边界**先说清：
1. **RPC 边界**：OmniUAV 只依赖 §5 的 RPC 契约，与引擎内部实现解耦——sidecar 里换任何算法，OmniUAV **零改动**。
2. **MVA 契约**：MVA 已是契约驱动（同一 Pydantic/Protocol 契约下有多个可互换实现，如 L2 跨视角关联的 geometric/appearance/llm）。新增模块沿用同风格：**面向接口编程，实现可换**。

每个模块的"基线(现在) / 接口契约 / 未来升级候选"：

| 模块 | 基线（现在用） | 接口/契约 | 未来升级候选 |
|---|---|---|---|
| 检测 | YOLOv11 / YOLOE(开放词表) | `Detector.detect(img)->list[Detection]` | YOLO-World、GroundingDINO、专用微调权重 |
| 单视角跟踪 | `iou_greedy`(默认) | `Tracker.update(dets,h,w,frame)->[(det,id)]`+`reset()` | ByteTrack、BoT-SORT、带 ReID 的 tracker |
| 跨视角关联 | `appearance`(外观,适配独立无人机) | `CrossViewLinker.link(obs)->list[CrossViewLink]` | geometric+标定、Transformer link model、BEV 融合、真 ReID(OSNet/TransReID) |
| 嵌入/检索 | 本地 `Qwen3-VL-Embedding-8B`(768d) | `MultimodalEmbedder.encode_*` + `VectorStore.query` | 换模型/维度、加 `Qwen3-VL-Reranker` 两阶段精排、VLM2Vec-V2 |
| **geo-grounding**(新) | 检测框中心+深度反投影(近似) | `GeoGrounder.locate(obj,depth,pose,gps)->world_xyz` | 多视角三角化、地平面约束、SLAM/位姿优化 |
| **时空关系建模**(新) | 规则(相对方位/距离/接近远离) | `RelationModeler.model(objects_over_time)->SceneGraph` | 学习式场景图生成(STEP/时空 SGG)、关系向量化 |
| 空间/几何 QA | 关系 NL 上下文+帧 → 云端 Qwen3-VL-Plus | `/answer` 的 prompt 组装模块化 | VLM-3R/GPT4Scene 式 3D-aware VLM、本地 3D 推理 |
| 问答编排 | MVA Orchestrator(Mode A) | `QueryService.answer` | 多轮 ReAct、Mode B briefing、reranker 注入 |

**落地要求**：新增的 `GeoGrounder`、`RelationModeler` 必须先定义**抽象基类/Protocol + 一个基线实现**，配置里以名字选实现（对齐 MVA 现有 `tracker_algorithm`、`cross_view_linking_mode` 的选择方式）。基线实现要有单测（见 §11），换实现时测试与上层不变。

## 4. 架构

```
┌──────────────────────────── localhost（单机）─────────────────────────────┐
│                                                                            │
│  OmniUAV (env: simsys, PyQt5)          MVA 引擎 sidecar (env: mva, cu126)   │
│  ┌───────────────────────┐  HTTP/JSON  ┌──────────────────────────────┐    │
│  │ 实时镜头 grid          │────────────▶│ FastAPI (localhost:8900)      │    │
│  │ 跨视角跟踪 / 多视角检索 │◀────────────│  /ingest /answer /retrieve    │    │
│  │ 空间关系理解 面板       │             │  /state/* /health /events     │    │
│  │ 右侧 grounded 问答/态势 │             ├──────────────────────────────┤    │
│  │ (保留快捷问题四组)      │             │ QueryService / Orchestrator   │    │
│  │ 大模型评估 tab(现有)    │             │ ingest_scene / LiveIngestor   │    │
│  └───────────────────────┘             │ Detector/Tracker/CrossView(L1-2)│  │
│        │ 共享帧目录(jpg 路径)           │ **geo-grounding + 关系建模(新)** │   │
│        └──── /tmp/mva_frames ──────────▶│ Embedder=本地 Qwen3-VL-Emb-8B  │    │
│                                         │ DuckDB(独占) + ChromaDB        │    │
│  云端 Qwen3-VL-Plus ◀──(MVA 问答时调)───│                              │    │
│  UE4 + ROS 容器(仅实时模式)             └──────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
```

**职责边界**
- **OmniUAV（前端，唯一界面）**：多路显示、所有面板渲染、发问答、触发/停止入库、引擎状态灯。**不碰 ML、不碰 DB**，一律经 RPC。保持 Qt 轻、不卡。
- **MVA sidecar（引擎，隐形）**：独占 DuckDB+ChromaDB + 全部模型（本地嵌入 8B、检测、跟踪、跨视角、geo-grounding、关系建模、QA agent）。被启动脚本悄悄拉起。
- **两套 conda 环境隔离**，互不污染依赖。

## 5. 集成缝 / RPC 接口（FastAPI，localhost:8900）

| 端点 | 方法 | 作用 | 复用 MVA 现成能力 |
|---|---|---|---|
| `/health` | GET | 探活；OmniUAV 据此灰化面板/降级 | 新增 |
| `/ingest/start` | POST | `{source, mode:offline\|live, config}` → `job_id` | `ingest_scene()` / `LiveIngestor` |
| `/ingest/status` | GET | `?job=` 进度(已处理段数/当前时刻) | 新增薄封装 |
| `/ingest/stop` | POST | 停止实时滚动入库 | 新增 |
| `/answer` | POST | `{query, attachments?, session_id?}` → 带 grounding 的答案 | `QueryService.answer()` |
| `/retrieve` | POST | `{text\|image_path, top_k, vector_type}` → 命中列表(view/time/track/缩略图路径/向量数) | ChromaDB + L6 检索工具 |
| `/state/cross_view_links` | GET | 跨视角关联 | `query_cross_view_links()` |
| `/state/tracklets` | GET | `?view=&t_range=` 轨迹/时间线 | `query_tracklets()` |
| `/state/relations` | GET | `?t=` 时空关系/场景图（NL+结构化） | **新增（关系建模产物）** |
| `/state/scene_stats` | GET | 概况 | `query_*` |
| `/state/events` | GET | 事件/异常（弱） | L3 |
| `/events/stream` | GET(SSE) | 入库进度 + 状态变更推送（可选，先用轮询亦可） | 新增 |

- 传输：HTTP+JSON。图像**只传路径**（共享 `/tmp/mva_frames`）。
- OmniUAV 侧封装一个 `MvaClient`（`requests`），各面板通过它取数据。

## 6. 数据流

### 6.1 离线模式（主力：存视频 / 开源 benchmark）
1. OmniUAV 选一个视频文件或 benchmark scene → `POST /ingest/start {source, mode:offline}`。
2. MVA 跑 检测→跟踪→跨视角关联→**本地嵌入**→（若有深度/位姿/GPS）geo-grounding+关系建模→写 DuckDB+Chroma；`/ingest/status` 报进度。
3. 入库完：各面板从 `/state/*`、`/retrieve` 拉数据渲染；问答走 `/answer`（内部调云端 Qwen3-VL-Plus，答案带"视角/时刻/轨迹"溯源）。
   → 复用 MVA 的 `ingest_scene()` + `QueryService`，几乎不改 MVA 核心。

### 6.2 近实时滚动（偶尔：实时可行性验证）
1. OmniUAV 已有 rosbridge 实时帧；把滚动窗口的帧写进 `/tmp/mva_frames/<view>/`，`POST /ingest/start {mode:live, source:该目录}`。
2. MVA 的 `LiveIngestor` 式 worker 滚动入库（**嵌入每 N 秒批处理一次**，给 UE4 让显存）。
3. UI 与面板不变，问答基于"最新世界状态"。
   → 复用 MVA 现成 `LiveIngestor`，把"循环喂文件"换成"读共享帧目录"。

## 7. UI 面板设计

窗口在现有 OmniUAV 结构上扩展：左侧 QTabWidget 承载分析模块，右侧常驻 grounded 问答/态势（**保留现有"快捷问题"四组：二维检测 / 时序跟踪 / 三维空间 / 语义理解**），底部系统日志。

### 7.1 模块 → 面板 → 数据源 → 成熟度

| 面板（左侧 tab） | 对应模块 | 展示 | 数据源(RPC) | 成熟度 |
|---|---|---|---|---|
| 实时镜头 | (基础) | 2×2 播放 + 检测框 + 全局 ID | `/state/tracklets` | ✅ 现成 grid + 接 ID |
| 跨视角跟踪 | 跨视角多目标跟踪 | 同目标跨视角同色全局 ID 高亮 + 关联列表 | `/state/cross_view_links`+`/state/tracklets` | 🟡 可用但粗：段内 ID、无全局身份 |
| 多视角检索 | 多视角信息检索 | 搜索(文/图)→命中缩略图；**并显示命中的帧/段、向量数、信息量**（检索透明化） | `/retrieve` | ✅ MVA 强项 |
| 空间关系理解 | 时空关系建模 + 3D 空间理解 QA | 视频网格 + 关系列表/场景图 + 几何问答（**无 BEV**） | `/state/relations`+`/answer` | 🔬 新建、最有挑战 |
| 大模型评估 | (基础) | 开源 benchmark 跑分 | 现有 EvaluationTab | ✅ 保留不动 |
| 右侧 问答/态势 | (核心粘合) | grounded 问答 + 周期态势总结，答案带溯源；保留快捷问题 | `/answer` | ✅ 最大升级 |

> 砍掉：原设想的"嵌入压缩" tab（其内容本质是检索，已并入多视角检索面板做透明化）。

### 7.2 空间关系理解面板布局

```
┌ 空间关系理解 ─────────────────────────────────────────────────────┐
│ ┌──────────┬──────────┐          │  关系列表 / 场景图(主)          │
│ │ view1 #7 │ view2 #7 │  ← 视频  │  #7船 在 #11船 左侧 ~8m         │
│ ├──────────┼──────────┤   网格   │  #11船 正靠近 #14浮标            │
│ │ view3    │ view4    │   照常   │  #7 远离 view3 区域             │
│ │          │ #7 #14   │   展示   │  (关系随时间轴变化)             │
│ └──────────┴──────────┘          │                              │
│ 时间轴 ◀━━━●━━━▶                  │                              │
├───────────────────────────────────┴───────────────────────────────┤
│ 几何问答: > #7 在 #11 的什么方位?  答:左侧约 8m,正在靠近 #14…     │
└─────────────────────────────────────────────────────────────────────┘
```
- 视频网格（左）始终在。
- 关系列表/场景图（右·主）：`geo-grounding → 关系建模`的产物，文本/节点连线；**这就是注入 MLLM 的关系上下文**。
- 几何问答（底）：走 `/answer`，把关系上下文 + 相关帧喂云端 Qwen3-VL-Plus。

## 8. 空间关系理解 pipeline（sidecar 内新增）

```
每视角 检测/跟踪(L1) ──┐
                        ├─ 跨视角关联(L2) ── 全局目标(去重后的物体)
深度 + 相机位姿 + GPS ──┘            │
                                    ▼
  ① geo-grounding：2D框 + 深度 + 位姿/GPS → 每个全局目标的【世界坐标 3D 位置】
                                    │
                                    ▼
  ② 时空关系建模：对全局目标两两算 相对方位(左/右/前/后)、距离、接近/远离、相对运动
                → 【场景图,随时间演变】→ 序列化成 自然语言(主) + 向量(可选)
                                    │
                                    ▼
  ③ 3D 空间理解 QA：MLLM(云端 Qwen3-VL-Plus)输入 =
                相关帧(视觉) + 关系上下文(NL/结构化) + 深度/位姿/GPS 数值
                → 回答 "A 在 B 什么方位" / "B 是否在靠近 C" 等几何关系问题
```

- **表示**：关系主用**自然语言**注入云端 MLLM（Qwen3-VL-Plus 直接吃文本上下文）；**向量表示可选**，顺带解锁"关系检索"（如"找 A 靠近 B 的时刻"），反哺检索面板。
- **落库**：关系/场景图写进 DuckDB（新增场景图表）。
- **依赖**：MVA `telemetry` 字段目前是 stub，需接 AirSim 深度/位姿/GPS。相机内外参来自 AirSim。

### 8.1 数据前提（决定"空间关系 QA"在哪些数据上可用）

| 数据源 | 深度 | 相机位姿 | GPS/telemetry | 空间关系 QA |
|---|---|---|---|---|
| AirSim 仿真 | ✅ | ✅ | ✅ | **✅ 完整可用** |
| MATRIX 等标定数据集 | 部分 | ✅ 标定 | 部分 | 🟡 部分可用 |
| VisDrone / MVU-Eval 纯 RGB | ❌ | ❌ | ❌ | **❌ 退化**（只能靠 VLM 从像素猜方位，不可靠） |

结论：**空间关系理解主要是"仿真/有标定数据"的功能**；检索、跨视角、grounded 问答在任何数据上都能用。纯 RGB benchmark 上该模块面板显示"当前数据无深度/位姿，空间关系不可用"。

## 9. 分阶段落地

| 阶段 | 内容 | 交付 | 适用数据 |
|---|---|---|---|
| **P0 骨架** | MVA sidecar(独立环境)+FastAPI+`/health`；OmniUAV `MvaClient`+引擎状态灯；共享帧目录；离线入库一段视频→落库；`/answer` 接通 | 选视频→入库→**grounded 问答跑通**（替换现有"直接喂帧"） | 任何 |
| **P1 基础面板** | 多视角检索面板(`/retrieve`+透明化)；跨视角跟踪面板(`/state/cross_view_links`+网格叠全局 ID)；实时镜头接全局 ID | 检索 + 跨视角跟踪 + grounded 问答，**离线/benchmark 都能 demo** | 任何 |
| **P2 空间关系理解** | sidecar 新增 geo-grounding + 关系建模；`/state/relations`；空间关系面板；`/answer` 注入关系上下文 | 仿真数据上**几何关系问答 + 关系列表可视化** | 仿真/标定 |
| **P3 近实时+打磨** | live 模式：滚动帧→共享目录→`/ingest live`；嵌入时间片；SSE 刷新 | 偶尔的实时可行性验证 | 仿真 |
| **未来** | visual-token 压缩、reranker、本地问答、关系向量检索深化、HITL、流式输出 | — | — |

## 10. 错误处理 / 降级

- **sidecar 挂了/未连**：`/health` 失败 → 引擎灯灰、分析面板显示"引擎未连接"，**问答降级回现有"直接喂帧给云端 VLM"路径**（OmniUAV 已有此逻辑），不至于完全没问答。
- **未入库（DB 空）**：面板提示"请先选视频入库"，不报错。
- **入库失败**（视频损坏/零检测）：`/ingest/status` 报错 + 日志 + 面板提示。
- **GPU OOM**（嵌入 8B ⇄ 仿真）：实时模式嵌入**时间片**；仍 OOM → 退 CPU 嵌入(慢)或只保检测/跟踪并告警。离线无仿真基本不触发。
- **空间数据缺失**（纯 RGB benchmark）：空间关系面板显示"当前数据无深度/位姿，空间关系不可用"，其他模块照常。
- **云端 API 错误**：复用 OmniUAV 现有处理。

## 11. 测试策略

- **API 契约测试**：每个端点 request/response schema。
- **冒烟端到端**：短样例视频 → `/ingest` → 断言 DuckDB/Chroma 有数据 → `/answer` 返回带溯源答案、`/retrieve` 有命中。
- **面板渲染**：用 fixture DuckDB（预置 tracklets/links/relations）驱动各面板，不依赖真模型。
- **geo-grounding 单测**：已知 depth+pose 的合成样例 → 反投影世界坐标误差在阈值内。
- **关系建模单测**：给定世界坐标 → 断言 左/右/靠近 判定正确。
- **降级测试**：停 sidecar → 问答回退路径可用。
- **MVA 现有 436 测试复用**（不动核心，只加 sidecar 层 + geo/relation 新模块测试）。

## 12. 风险与开放问题

- **R1 显存**：实时模式下本地嵌入 8B(~16–18GB) 与 UE4 抢卡。缓解：嵌入时间片；离线主力模式无此问题。
- **R2 geo-grounding 精度**：反投影依赖检测框中心 + 深度质量；仿真尚可，真实数据需标定。仅用于近似方位/距离。
- **R3 MVA telemetry 接入**：GPS/位姿字段现为 stub，需新增 AirSim → sidecar 的 telemetry 通路。
- **R4 跨视角全局身份**：MVA 只有段内 track id、无跨段全局身份 → 跨视角跟踪面板"可用但粗"，关系里同一物体可能跨段换 id。是否补全局 ReID 待定（当前"能用就行"）。
- **R5 检索模型维度**：本地 Qwen3-VL-Embedding-8B = 768 维；若未来切云端 v1(1024 维)需整库重嵌入（本设计已定本地，不涉及）。
- **R6 两进程启停编排**：需扩展 `start_live_demo.sh`/`stop_live_demo.sh` 拉起/关闭 sidecar（复用现有多进程编排；注意 pkill 自匹配坑，见 MODIFICATIONS.md）。

## 13. 附录：关键代码锚点（来自代码勘察）

**OmniUAV**（`omni-uav/`）
- 主窗口/tab 注册/依赖注入：`app.py`（tab 用 `tabs.addTab` 硬编码，无插件系统，MainWindow 是中枢）
- 共享缓冲：`utils/frame_buffer.py` `FrameBuffer`（`get_recent_frames/get_frames_by_time/get_latest_frame`）
- 流接口：`widgets/video_stream.py` `StreamBase`；`widgets/ros_bridge_stream.py` `RosBridgeLiveStream`
- VLM 调用：`utils/llm_client.py` `LlmClient`(`MODEL_CONFIGS`,`chat`)；`workers/llm_worker.py` `LlmWorker`；`app.py` `_enqueue_llm_request/_start_next_llm`
- 问答路径：`app.py` `_execute_regular_query`,`parse_temporal_scope`,`get_latest_frames`
- 现有检测/跟踪(将被 MVA 取代)：`utils/visdrone_detector.py`,`utils/object_tracker_manager.py`,`utils/cross_camera_tracker.py`
- 配置/开关：`configs/config_llm.yaml`,`configs/config.json`；env `OMNIUAV_ROS_LIVE`,`ROSBRIDGE_HOST/PORT`

**MVA**（`Multi-Video-Analysis/src/mva/`）
- 库门面：`cli/query.py` `QueryService.answer()`
- 入库：`cli/ingest.py` `ingest_scene()`；实时参考：`cli/live_ingest.py` `LiveIngestor`
- 世界状态：`l5_state/duckdb_store.py` `WorldStateStore`(`execute_readonly(sql)`,`query_tracklets/cross_view_links/...`)；`l5_state/chromadb_store.py` `VectorStore`
- 嵌入：`l5_state/embedder.py` `MultimodalEmbedder`（`DEFAULT_MODEL="Qwen/Qwen3-VL-Embedding-8B"`, 768 维）
- 感知：`l1_perception/detector.py` `Detector`；`l1_perception/tracker.py`
- 跨视角：`l2_crossview/{geometric,appearance,llm_mode}.py`
- 事件(弱)：`l3_events/algorithmic.py`
- 交互/QA：`l6_interaction/{orchestrator,planner,tools}.py`
- **新增（本设计）**：sidecar FastAPI 服务层；geo-grounding + 时空关系建模模块；AirSim telemetry 通路

---
*本文档为设计定稿，供评审。评审通过后进入实现计划（writing-plans）。*
