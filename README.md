# OmniUAV-MVA

多视角视频态势感知理解系统：以 **OmniUAV**（PyQt5 前端/唯一界面）为宿主，集成 **MVA**（多视角视频理解引擎：检测/跟踪/跨视角关联/多模态嵌入/世界状态/检索增强问答），实现对多路无人机视频的高效、精准理解与问答。

## 结构（monorepo）

```
OmniUAV-MVA/
├── omni-uav/     # PyQt5 前端：多路显示、面板、问答（唯一界面）
├── mva/          # MVA 引擎源码（src/ + tests/ + pyproject.toml）
│                 #   大文件(模型权重/DATASETS/runs)不入库，留在本地/HF 缓存
├── docs/         # 设计 spec + 分阶段实现计划
└── scripts/      # sidecar 启停等脚本
```

## 架构（方案 A：sidecar）

- **MVA 以本地 sidecar 服务运行**（FastAPI `localhost:8900`），独占世界状态 DuckDB+ChromaDB，
  封装 `QueryService`（本地 `Qwen3-VL-Embedding-8B` 嵌入 + 云端 `qwen3-vl-plus` 问答）。
- **OmniUAV 经 RPC 调用**：问答统一走 sidecar `/answer`（引擎掉线自动降级本地直连 VLM）；
  指令（跟踪/检测/重建）留在 OmniUAV 自己理解+执行。
- 设计详见 `docs/superpowers/specs/2026-07-10-omniuav-mva-integration-design.md`。

## 配置与密钥

- **API key 走环境变量**：`export DASHSCOPE_API_KEY=<你的key>`。
- `omni-uav/configs/config_llm.yaml`、`config.json` **不入库**（含密钥，本地保留）；
  参考 `omni-uav/configs/config_llm.yaml.example` 创建本地配置。

## 环境

- MVA/sidecar：conda env（含 `fastapi/uvicorn/duckdb/chromadb/sentence-transformers/torch cu126`），
  `pip install -e mva`。
- OmniUAV：PyQt5 运行环境。
