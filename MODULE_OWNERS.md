# 模块分工与交接（MODULE_OWNERS）

多视角全局 3D 态势融合系统 —— 6 个模块，不同人分布式并行开发。
**每个人只实现自己模块的算法，对着共享契约 + Protocol，互不阻塞。**

- 架构设计：`docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md`
- 契约层计划：`docs/superpowers/plans/2026-07-19-phase0-contract-layer.md`
- 框架图 / 分工图：`docs/diagrams/framework.html`、`docs/diagrams/architecture.html`（浏览器打开）
- 变更记录：`MODIFICATIONS.md`

---

## 一、拿代码（完整项目 = git 仓库）

**完整项目就是 git 仓库里被跟踪的 232 个文件**（`mva/`、`omni-uav/`、`docs/`、`scripts/`、`README.md`、`MODIFICATIONS.md`、本文件）。

```bash
git clone git@github.com:fanyunfeng-bit/OmniUAV-MVA.git
```

> ⚠️ **用 `git clone`，不要直接拷工作目录。** 本地工作目录里有 **gitignored 的密钥文件**（`omni-uav/configs/config_llm.yaml`、`config.json` 含 API key），直接打包会把密钥泄露给别人。clone 只取被跟踪文件，自动不含密钥/数据/权重/缓存。

**不在仓库里、每个人各自准备的东西**（见 `.gitignore`）：
- 模型权重（`*.pt/*.safetensors/...`）——首次运行 ultralytics/HF 自动下载；
- 世界状态库（`*.duckdb`、`chroma/`）——运行时生成；
- 媒体/数据集（`*.mp4`、`mva/DATASETS/`）——按需自备；
- **API key**——各自建自己的 `omni-uav/configs/config_llm.yaml`（见下），**绝不提交**。

## 二、环境搭建

```bash
# MVA 引擎（检测/几何/融合/关系/检索/世界模型/服务/评测）
conda create -n mva python=3.10 -y && conda activate mva
pip install -e mva            # 装 mva 及其依赖（fastapi/duckdb/chromadb/torch/ultralytics...）

# OmniUAV 前端（PyQt5 UI）
conda create -n simsys python=3.10 -y && conda activate simsys
pip install -r omni-uav/requirements.txt

# API key（问答用云端 qwen3-vl-plus；仅涉及问答/云端 LLM 的模块需要）
cp omni-uav/configs/config_llm.yaml.example omni-uav/configs/config_llm.yaml
#   填入自己的 DASHSCOPE_API_KEY；或 export DASHSCOPE_API_KEY=...
```

跑测试确认环境 OK：
```bash
cd mva      && python -m pytest -m "not gpu" -q      # 契约/接口/store 全绿
cd omni-uav && QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q
```

## 三、六个模块（认领一列，改一个模块）

> 通则：**只对着「接口 + 契约」编码**；产出写世界模型表，消费读世界模型；把对应 `fakes.py` 桩换成真实现即可，不碰别人代码。

| 模块 | 接口（Protocol 包） | 现有代码 / 起步点 | 消费 → 产出（契约） | 世界模型表 | 评测指标 |
|---|---|---|---|---|---|
| **M1 检测/分割** | `mva/detection/`（ObjectDetector/Segmenter）| `l1_perception/detector.py`（YOLO 已满足接口）| frames → `Detection` | —（转 M3）| 检测 mAP / 分割 mIoU |
| **M2 几何/度量对齐** | `mva/geometry/`（PoseProvider/Projector/TimeSync）| 新写；用 `datasets/airsim_gt.AirSimGT` 真值位姿起步，替换 `geometry/fakes.py` | frames + 位姿元数据 → `CameraPose`/射线/地面反投影 | `camera_poses` | ATE/RPE、反投影误差(m) |
| **M3 多目标跟踪(单+多视角) ★** | `mva/fusion/`（CrossViewAssociator/Triangulator/GlobalTracker）+ 单视角 `perception/pipeline.Tracker` | `l1_perception`(ByteTrack) + `l2_crossview`(弱关联，需重写) | `Detection`(M1)+`CameraPose/Projector`(M2) → `GlobalObject`/`GlobalObservation`/`GlobalTrajectory` | `global_objects`/`_observations`/`_trajectory` | 全局 MOTA/IDF1、counting MAE、3D 误差(m) |
| **M4 时空关系+预测** | `mva/reasoning/`（EventDetector/TrajectoryPredictor）+ `perception/relation.RelationModeler` | `l3_events` + `perception/relation.py`(stub) | `GlobalTrajectory`(M3) → `SceneGraphEdge`/`SituationEvent`/`GlobalPrediction` | `scene_graph_edges`/`situation_events`/`global_predictions` | SGGen recall、事件/异常 F1、ADE/FDE |
| **M5 信息压缩+检索** | `mva/retrieval/`（Embedder/Retriever）+ `service/query_understanding.ConstraintParser` | `l5_state/embedder.py` + `service/retrieval.py` + `service/query_understanding.py`（已可用） | frames + `GlobalObject`/tracks → 向量 + `RetrieveResponse` | chroma 向量库 | recall@k / mAP |
| **M6 平台** | 拥有 `contracts/`（改 schema 的唯一入口）| `contracts/` + `l5_state/`(存储) + `service/`(sidecar) + `l6_interaction/`(问答) + `omni-uav/`(UI) + `cli/eval.py`(评测) + `datasets/airsim_gt.py`(GT) | 读所有产物 → 空间问答/UI/评测 | 拥有全部表 | 端到端空间问答准确率 |

**契约类型（M6 拥有，全员共用，`mva/src/mva/contracts/`）**：
`Detection`（l1）、`CameraPose`/`Ray`/`WorldPoint`（geometry）、`GlobalObject`/`GlobalObservation`/`GlobalTrajectory`（global_state）、`SceneGraphEdge`/`SituationEvent`/`GlobalPrediction`（spatiotemporal）。

## 四、并行开发约定

1. **对着契约编码**：谁都不 import 别人模块的内部实现；跨模块只经 `contracts/` 类型 + `*/protocol.py` 接口 + 世界模型表。
2. **改契约要协调**：动 `contracts/` 或世界模型表结构，必须走 M6 owner + 升版本号；别人才不会被打断。
3. **各自测 + 全量门**：每个模块有自己的 `tests/`；提交前跑 `pytest -m "not gpu"` 全绿（现基线 621 passed），保证没碰坏别人。
4. **协作方式**：小队可直接提交 `main`，或每模块开分支走 PR（推荐多人时用 PR + review）。提交前务必
   `git grep --cached -nE "sk-[A-Za-z0-9]{20,}"` 确认没把密钥提交进去。
5. **从桩到真**：每个模块先跑通 `fakes.py`（占位），再逐步把桩换成真算法，沿自己的评测指标迭代。

## 五、关键路径（先出首个端到端全局 demo）

**M2（AirSim 真值位姿 → Projector）→ M3（关联 + 三角化 → GlobalObject）→ M6（counting + 空间问答工具）。**
M1 先用现有 YOLO，M4/M5 并行推进。3DGS 稠密重建不在关键路径，作可选后续。
