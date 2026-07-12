# OmniUAV-MVA

多视角视频态势感知理解系统：以 **OmniUAV**（PyQt5 前端 / 唯一界面）为宿主，集成 **MVA**（多视角视频理解引擎：检测 / 跟踪 / 跨视角关联 / 多模态嵌入 / 世界状态 / 检索增强问答），对多路无人机视频做高效、精准、可溯源的理解与问答。**当前聚焦离线：对录制好的本地多视角视频做 编码 → 检索 → 问答。**

## 功能一览

| 功能 | 状态 | 说明 |
|---|---|---|
| grounded 问答 | ✅ | 入库后从世界状态作答(计数/跨视角等)、带溯源；未入库则模型看当前帧作答；引擎掉线本地降级 |
| 入库(编码) | ✅ | GUI"入库"一键把当前文件夹作为一个场景送入引擎(检测/跟踪/跨视角/嵌入)，进程内进行 |
| 多视角检索 | ✅ | 文字查询 → top-3 命中(view/时刻/分数) + top-1 缩略图 + 检索透明化 + 点击跳帧 |
| 按 scene 分库 | ✅ | 每个场景独立库 `~/.omniuav-mva/<scene>/`，选文件夹自动切库、互不干扰 |
| 数据采集 | ✅ | 从 AirSim 4 无人机视角经 rosbridge 录制为本地 mp4 |
| 视频按原生 fps 播放 | ✅ | 低帧率视频不再被放快 |
| 跨视角跟踪面板 | 🔜 | 待"密集感知流 + 更强 tracker"(接口已留，见 spec §3.2) |
| 3D 空间关系理解 | 🔜 | 需深度+位姿+GPS(仿真/标定数据)，接口规划中 |

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
4. **多视角检索** tab → 输入 `airplane`/`一艘船` → top-3 命中 + top-1 缩略图，点击命中跳到对应视角该帧。

> 默认打开 `~/OmniUAV-MVA-data/airsim_downtown_4view`(若存在)，否则回退内置 `omni-uav/examples/`。

## 采集仿真数据

```bash
# 前提：仿真已起(UE4 + airsim_node + rosbridge + planner + patrol，无人机在飞)
/home/fyf/miniconda3/envs/simsys/bin/python scripts/record_airsim_4view.py \
  --duration 180 --out ~/OmniUAV-MVA-data/airsim_downtown_4view
```
输出 `cam01-04.mp4`(每个视频=一个视角)，可直接在 OmniUAV 里选该文件夹显示 + 入库 + 问答/检索。

## 架构（方案 A：sidecar）

- **MVA 以本地 sidecar(FastAPI `localhost:8900`)运行**，独占世界状态 DuckDB+ChromaDB；封装
  `QueryService`（本地 `Qwen3-VL-Embedding-8B` 嵌入 + 云端 `qwen3-vl-plus` 问答）。
- **OmniUAV 经 RPC 调用**：问答统一走 `/answer`(带当前帧附件；引擎掉线本地降级)；入库/检索/切库走各自端点。
  指令(跟踪/检测/重建)留在 OmniUAV 自己理解+执行。
- **入库进程内进行**：复用 sidecar 已加载的 store/embedder/vstore，避开 DuckDB 跨进程锁、免重载 16G 嵌入。
- **RPC 端点**：`/health` `/answer` `/ingest/{start,status,stop}` `/retrieve` `/select_scene`。
- 设计详见 `docs/superpowers/specs/2026-07-10-omniuav-mva-integration-design.md`；变更记录见 `MODIFICATIONS.md`。

## 结构（monorepo）

```
OmniUAV-MVA/
├── omni-uav/     # PyQt5 前端：多路显示、检索面板、问答（唯一界面）
│   ├── app.py            # 主窗口、问答路由、入库/检索/切库联动
│   ├── tabs/             # camera_tab / retrieval_tab / reconstruction / evaluation
│   ├── widgets/          # VideoStream(按原生fps播放) / ros 流 等
│   └── utils/mva_client.py   # → sidecar 的 RPC 客户端
├── mva/          # MVA 引擎源码（src/ + tests/ + pyproject.toml）
│   └── src/mva/
│       ├── service/      # sidecar：app(FastAPI) / engine(库切换+进程内入库) / retrieval / thumbnails
│       ├── l4_llm/cloud_client.py   # DashScope 云端 LLM 适配器
│       └── perception/   # 感知流(密)接口 + 基线：FrameSource/Tracker/Pipeline/RelationModeler
├── scripts/      # start/stop sidecar、record_airsim_4view
├── docs/         # 设计 spec + 分阶段实现计划
└── MODIFICATIONS.md      # 汇总变更记录(带 commit 号)
```
> 大文件(模型权重 / DATASETS / 录制视频 / 世界状态库)不入库，留在本地/HF 缓存。

## 配置与密钥

- **API key**：只存本地 `omni-uav/configs/config_llm.yaml` / `config.json`(gitignore，不入库)；
  启动脚本运行时**自动读取**，无需手动 export。也可 `export DASHSCOPE_API_KEY=...` 覆盖。
- 参考 `omni-uav/configs/config_llm.yaml.example` 创建本地配置。
- 世界状态库默认在 `~/.omniuav-mva/`(可用 `MVA_DB`/`MVA_CHROMA` 覆盖)。

## 环境

- **MVA/sidecar**：conda env `mva`(含 `fastapi/uvicorn/duckdb/chromadb/sentence-transformers/torch cu126/ultralytics`)，`pip install -e mva`。本地嵌入 `Qwen3-VL-Embedding-8B` 约 16GB 显存。
- **OmniUAV**：conda env `simsys`(PyQt5 + torch + opencv + requests + roslibpy)。

## 测试

```bash
cd mva      && /home/fyf/miniconda3/envs/mva/bin/python -m pytest -m "not gpu" -q      # 547 passed
cd omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/ -q   # 9 passed
```
