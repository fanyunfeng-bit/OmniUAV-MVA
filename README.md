# OmniUAV-MVA

多视角视频态势感知理解系统：以 **OmniUAV**（PyQt5 前端 / 唯一界面）为宿主，集成 **MVA**（多视角视频理解引擎），对多路无人机视频做高效、精准、可溯源的理解与问答。

**现状**：离线可用——对录制好的本地多视角视频做 编码 → 检索 → grounded 问答。
**方向**：正按 **6 个模块并行** 推进「多视角全局 3D 态势融合」，让每个真实目标在同一世界坐标系里有一条轨迹，支撑全局检测/跟踪/counting/空间关系问答/时空预测。

> 👥 **要参与开发 / 认领模块**：先读 **[`MODULE_OWNERS.md`](MODULE_OWNERS.md)**（谁改哪、怎么并行）。
> 🧭 **设计与走向**：`docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md` + `docs/diagrams/architecture.html`、`framework.html`（浏览器打开）。

## 功能一览

| 功能 | 状态 | 说明 |
|---|---|---|
| grounded 问答 | ✅ | 入库后从世界状态作答(计数/跨视角等)、带溯源；未入库则模型看当前帧作答；引擎掉线本地降级 |
| 入库(编码) | ✅ | GUI"入库"一键把当前文件夹作为一个场景送入引擎(检测/跟踪/跨视角/嵌入)，进程内进行 |
| 多视角检索 | ✅ | 文字查询→解析「视角/时间段」约束(规则优先→LLM兜底)硬过滤+空回退→top-3命中+top-1缩略图+透明化+跳帧 |
| 按 scene 分库 | ✅ | 每个场景独立库 `~/.omniuav-mva/<scene>/`，选文件夹自动切库、互不干扰 |
| 数据采集 | ✅ | 从 AirSim 4 无人机视角经 rosbridge 录制为本地 mp4 |
| 视频按原生 fps 播放 | ✅ | 低帧率视频不再被放快 |
| **模块化契约层 (Phase 0)** | ✅ | 6 模块的 Protocol 接口 + 共享世界模型契约/表 + AirSim GT 适配器已就位，各模块可并行开工 |
| 全局 3D 融合 / 跟踪 (M2/M3) | 🔜 | 跨视角几何 + 全局对象注册表；接口已留(`geometry/`、`fusion/`)，桩待换真算法 |
| 时空关系 + 预测 (M4) | 🔜 | 场景图/事件/预测；接口已留(`reasoning/`) |

## 快速开始

```bash
# 1) 起 MVA sidecar（自动读本地 key，默认持久库 ~/.omniuav-mva，首次加载嵌入约 60s）
cd /home/fyf/fyf/PCL/OmniUAV-MVA
bash scripts/start_mva_sidecar.sh
#    探活: curl http://127.0.0.1:8900/health   → {"status":"ok","engine_ready":true,...}

# 2) 启动 OmniUAV（simsys 环境；注意 DISPLAY=:0.0 绕过内置的 :0→:1 改写）
cd omni-uav && DISPLAY=:0.0 /home/fyf/miniconda3/envs/simsys/bin/python app.py

# 停止
bash scripts/stop_mva_sidecar.sh
```

GUI 流程：
1. **多无人机镜头** tab →「选择视频文件夹」选一个含多个 `camNN.mp4/*.mp4` 的目录(每个视频=一个视角) → 按原生帧率播放，sidecar 自动切到该场景库。
2. 点「**入库到分析引擎**」(可选) → 该场景编码入库(进度在底部日志)。
3. 右侧问答：未入库→模型看当前帧作答；入库后→grounded 从世界状态作答(带溯源)。
4. **多视角检索** tab → 输入 `一艘船`(全库) 或带约束的 `视角1里的黄车`/`最后10秒的车`(自动解析视角/时间段做硬过滤，命中为空再回退全库) → top-3 命中 + top-1 缩略图 + 透明化，点击命中跳到对应视角该帧。

> 默认打开 `~/OmniUAV-MVA-data/airsim_downtown_4view`(若存在)，否则回退内置 `omni-uav/examples/`。

## 采集仿真数据

```bash
# 前提：仿真已起(UE4 + airsim_node + rosbridge + planner + patrol，无人机在飞)
/home/fyf/miniconda3/envs/simsys/bin/python scripts/record_airsim_4view.py \
  --duration 180 --out ~/OmniUAV-MVA-data/airsim_downtown_4view
```
输出 `cam01-04.mp4`(每个视频=一个视角)，可直接在 OmniUAV 里选该文件夹显示 + 入库 + 问答/检索。

## 架构

**运行架构（方案 A：sidecar）**
- **MVA 以本地 sidecar(FastAPI `localhost:8900`)运行**，独占世界状态 DuckDB+ChromaDB；封装 `QueryService`（本地 `Qwen3-VL-Embedding-8B` 嵌入 + 云端 `qwen3-vl-plus` 问答）。
- **OmniUAV 经 RPC 调用**：问答走 `/answer`(带当前帧附件；引擎掉线本地降级)；入库/检索/切库走各自端点。
- **入库进程内进行**：复用 sidecar 已加载的 store/embedder/vstore，避开 DuckDB 跨进程锁、免重载 16G 嵌入。
- **RPC 端点**：`/health` `/answer` `/ingest/{start,status,stop}` `/retrieve` `/select_scene`。

**模块架构（6 模块围绕共享世界模型的生产者→消费者 DAG）**
- M1 检测/分割 → M2 跨视角几何/坐标系 → **M3 多目标跟踪(单+多视角，全局对象)** → M4 时空关系/预测；M5 压缩/检索；M6 平台(世界模型契约+问答+UI+评测)。
- **模块间只经「数据契约 + Protocol 接口」通信**，实现相互独立。详见 `MODULE_OWNERS.md` 与 spec。

## 代码结构（monorepo）

```
OmniUAV-MVA/
├── MODULE_OWNERS.md          # 👥 分工与交接：6 模块 owner 卡片 + 并行开发约定（开发者先读这份）
├── MODIFICATIONS.md          # 变更记录（带 commit 号）
├── README.md                 # 本文件：功能 + 结构 + 入口
├── docs/
│   ├── superpowers/specs/    # 设计 spec（含 2026-07-17 模块化全局3D融合）
│   ├── superpowers/plans/    # 分阶段实现计划（含 Phase 0 契约层）
│   └── diagrams/             # architecture.html（分工图）/ framework.html（框架图）
├── scripts/                  # start/stop_mva_sidecar、start/stop_live_mva(一条龙实时)、record_airsim_4view
├── sim/                      # 仿真侧启动脚本(AirSim+ROS+rosbridge)；重型运行时(UE4/镜像/工作区)为外部前置，见 sim/README.md
├── omni-uav/                 # 【M6·UI】PyQt5 前端（唯一界面）
│   ├── app.py                #   主窗口、问答路由、入库/检索/切库联动
│   ├── tabs/                 #   camera_tab / retrieval_tab / reconstruction / evaluation
│   ├── widgets/              #   VideoStream(按原生fps播放) / ros 流 等
│   ├── utils/mva_client.py   #   → sidecar 的 RPC 客户端
│   ├── workers/ dialogs/     #   后台线程 / 对话框
│   └── configs/              #   config_llm.yaml(.example) —— API key 只存本地、不入库
└── mva/                      # MVA 引擎（pip install -e mva）
    ├── pyproject.toml
    ├── tests/                #   unit / contracts / smoke（pytest -m "not gpu"）
    └── src/mva/
        │  ── 共享契约（M6 拥有，全员共用）──
        ├── contracts/        # 数据契约(pydantic)：stream / cross_view / events /
        │                     #   geometry(CameraPose/Ray/WorldPoint) /
        │                     #   global_state(GlobalObject/Observation/Trajectory) /
        │                     #   spatiotemporal(SceneGraphEdge/SituationEvent/GlobalPrediction)
        │  ── M1 检测/分割 ──
        ├── detection/        # 接口：ObjectDetector / Segmenter Protocol + fakes
        ├── l1_perception/    # 现有实现：YOLO Detector/Detection、ByteTracker
        │  ── M2 跨视角几何 ──
        ├── geometry/         # 接口：PoseProvider / Projector / TimeSync + fakes
        ├── l2_crossview/     # 现有跨视角关联(几何/外观/LLM，弱；M3 会重写)
        │  ── M3 多目标跟踪(单+多视角)·中心 ──
        ├── fusion/           # 接口：CrossViewAssociator / Triangulator / GlobalTracker + fakes
        │  ── M4 时空关系+预测 ──
        ├── reasoning/        # 接口：EventDetector / TrajectoryPredictor + fakes
        ├── l3_events/        # 现有事件层
        ├── perception/       # 感知流接口基线：FrameSource / Tracker / Pipeline / RelationModeler
        │  ── M5 压缩+检索 ──
        ├── retrieval/        # 接口：Embedder / Retriever Protocol + fakes
        ├── segmentation/     # 段级切分（滑窗取帧）
        │  ── M6 平台：世界模型 / LLM / 问答 / 服务 ──
        ├── l5_state/         # 世界模型存储：duckdb_store(全部表) / chromadb_store / embedder(M5)
        ├── l4_llm/           # LLM 客户端：本地 + cloud_client(DashScope 云端适配器)
        ├── l6_interaction/   # 问答编排：orchestrator / planner / tools / vlm_tools / memory
        ├── service/          # sidecar：app(FastAPI) / engine / models / retrieval / query_understanding / thumbnails
        │  ── 数据 / 命令行 / 其它 ──
        ├── datasets/         # 数据集适配器：pcl-sim / mvu_eval / matrix / reservoir / airsim_gt(真值 GT)
        ├── cli/              # 命令行：ingest / query / eval / index ...
        ├── l0_stream/        # 流 / 采集接入
        └── l7_hitl/          # human-in-the-loop
```
> 大文件(模型权重 / DATASETS / 录制视频 / 世界状态库 / 密钥)不入库，见 `.gitignore`；`git clone` 交接不会带出密钥/数据。

## 面向开发者

- **认领模块、去哪改、怎么并行**：见 **[`MODULE_OWNERS.md`](MODULE_OWNERS.md)**（每个模块的 接口/现有代码/契约/世界模型表/评测指标/起步点）。
- **原则**：只对着 `contracts/` 契约 + `*/protocol.py` 接口编码；把对应 `fakes.py` 桩换成真实现；改契约或表结构须经 M6 owner 升版本；提交前跑全量测试门。
- **关键路径**：M2(AirSim 真值位姿→Projector) → M3(关联+三角化→GlobalObject) → M6(counting+空间问答)，先出首个端到端全局 demo。

## 配置与密钥

- **API key**：只存本地 `omni-uav/configs/config_llm.yaml` / `config.json`(gitignore，不入库)；启动脚本运行时**自动读取**，无需手动 export。也可 `export DASHSCOPE_API_KEY=...` 覆盖。
- 参考 `omni-uav/configs/config_llm.yaml.example` 创建本地配置。
- 世界状态库默认在 `~/.omniuav-mva/`(可用 `MVA_DB`/`MVA_CHROMA` 覆盖)。

## 环境

- **MVA/sidecar**：conda env `mva`(含 `fastapi/uvicorn/duckdb/chromadb/sentence-transformers/torch cu126/ultralytics`)，`pip install -e mva`。本地嵌入 `Qwen3-VL-Embedding-8B` 约 16GB 显存。
- **OmniUAV**：conda env `simsys`(PyQt5 + torch + opencv + requests + roslibpy)，`pip install -r omni-uav/requirements.txt`。

## 测试

```bash
cd mva      && /home/fyf/miniconda3/envs/mva/bin/python -m pytest -m "not gpu" -q      # 621 passed
cd omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/ -q   # 13 passed
```
