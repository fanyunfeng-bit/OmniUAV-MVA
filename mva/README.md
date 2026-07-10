# Multi-Video-Analysis (MVA)

[![tests](https://github.com/fanyunfeng-bit/Multi-Video-Analysis/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/fanyunfeng-bit/Multi-Video-Analysis/actions/workflows/test.yml)

PCL pre-research: multi-drone multi-view video understanding + 7B multimodal LLM (Qwen2.5-VL-7B), with a 6-layer pipeline and a natural-language interaction agent.

| Doc | Role |
|---|---|
| [`PLAN.md`](PLAN.md) | **Authoritative architecture** (v0.3, 2026-05-20). Treat as ground truth; code/PLAN mismatches are bugs. |
| [`PROGRESS.md`](PROGRESS.md) | Per-milestone status, API surface deltas, and the "如何手动测试" cookbook. |
| [`CLAUDE.md`](CLAUDE.md) | Guidance for AI coding agents working in this repo. |

## Status

**M0 – M5.4 complete** (M5.1-pilot + M5.2-pilot partial + M5.2-prep + M5.4 UI landed; M5.1-scale + M5.2-full pending). **436 passing tests, pyflakes 0 warning.** Pipeline runs end-to-end from raw mp4 / PNG sequence → cross-view links (geometric + appearance + LLM-fallback) in DuckDB → ChromaDB multimodal index (Qwen3-VL-Embedding-8B, MRL-768, L2-normalized) → NL question answered by Qwen2.5-VL-7B (INT4 coexists with embedder on a 24 GB GPU) → Gradio chat UI (`mva ui`) with NL chat + segment playback. Three real datasets plugged in: MATRIX (multi-drone, full cross-cam GT), MVU-Eval (1824 multiple-choice QAs, accuracy eval pipeline working with P3-16 OOM mitigation), and VisDrone-MDMT (dual-drone test split).

Next: M5.1-scale (extend 26-pilot QAs → 100) and M5.2-full (1824 MVU-Eval async benchmark). See `PROGRESS.md` for the detailed punch list.

## Quickstart

Local dev box: NVIDIA driver ≥ 570 + a CUDA-capable GPU (3090/4090 verified).

```bash
# 1. Create the dedicated env (do NOT install into conda `base`)
conda create -n mva python=3.10 -y
conda activate mva

# 2. Install the package + all extras (detection, llm, storage, dev)
pip install -e .[all]

# 3. Pin torch to a CUDA wheel your driver supports
#    (default pip wheel currently ships cu130 which needs newer drivers;
#    cu126 covers driver 570+)
pip install --index-url https://download.pytorch.org/whl/cu126 \
    "torch>=2.10,<2.12" torchvision torchaudio

# 4. Sanity-check
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
pytest                          # → 436 passed
```

If `torch.cuda.is_available()` returns `False`, you got a cu13x wheel — repeat step 3.

## CLI

`pip install -e .` exposes a single `mva` entry-point. All subcommands
work on any registered dataset (matrix / mvu-eval / visdrone-mdmt; new ones plug in via
`mva.datasets.registry`).

**Mainline (M2.8 ingest → M5.4 UI):**

| Subcommand | What it does | Output |
|---|---|---|
| `mva ingest` | L0 → Segmenter (10s windows) → YOLO + ByteTracker + embed → DuckDB + ChromaDB in one pass | segments + tracklets tables (DuckDB) + segment/bbox vectors (ChromaDB) |
| `mva query` | NL REPL backed by Qwen2.5-VL + tools | interactive answers |
| `mva ask` | Single-shot NL question (+ `--image` / `--video`) | one answer + plan trace |
| `mva eval` | MVU-Eval QA accuracy benchmark (+ stratified `--per-task`, OOM-aware `--multi-video-nframes`) | JSONL + per-task report |
| `mva ui` | 🆕 M5.4 Gradio main page: NL chat + multimodal attachments + ffmpeg segment playback | local web app (default `127.0.0.1:7860`) |

**Legacy (frozen at M2.7, kept for back-reference, new work uses `ingest`):**

| Subcommand | What it does |
|---|---|
| `mva perceive` | L0 → L1 detect → L2 cross-view link → L5 DuckDB (old schema) |
| `mva index` | iter_indexable_units → Qwen3-VL-Embedding → ChromaDB (old one-vector-per-segment-only path) |

### Example workflows

```bash
# Unified M2.8 ingest: MATRIX two views at 10s segments + bbox embeddings
mva ingest --dataset matrix --scene MATRIX_30x30 --views D1 D3 \
    --db-path runs/m28-matrix.duckdb --chroma-dir runs/m28-matrix-chroma \
    --segments-per-view 30

# Unified M2.8 ingest: MVU-Eval scene (videos sliced into 10s segments)
mva ingest --dataset mvu-eval --scene qa-0 \
    --db-path runs/m28-mvu.duckdb --chroma-dir runs/m28-mvu-chroma \
    --segments-per-view 30 --detect-conf 0.3

# Same but lighter: skip bbox embedding (segment vectors only)
mva ingest --dataset mvu-eval --scene qa-0 \
    --db-path runs/m28-mvu-lite.duckdb --chroma-dir runs/m28-mvu-lite-chroma \
    --segments-per-view 0 --no-embed-bboxes

# Interactive REPL backed by Qwen2.5-VL (INT4) + Qwen3-VL-Embedding-8B
mva query --db-path runs/matrix.duckdb --chroma-dir runs/matrix-chroma \
    --llm Qwen/Qwen2.5-VL-7B-Instruct       # auto-INT4 when --chroma-dir set

# Single-shot question with image attachment (future-UI shape)
mva ask "画面里有没有这个人？" --image target.jpg \
    --db-path runs/matrix.duckdb --chroma-dir runs/matrix-chroma \
    --llm Qwen/Qwen2.5-VL-7B-Instruct

# MVU-Eval 5 Counting questions, INT4 inference, JSONL output
mva eval --dataset mvu-eval \
    --llm Qwen/Qwen2.5-VL-7B-Instruct --quantize int4 \
    --max-questions 5 --tasks Counting \
    --output runs/mvu-eval-5.jsonl

# Multi-video TR task with M5.2-prep dynamic nframes + OOM retry safety net
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True mva eval \
    --dataset mvu-eval --tasks TR --max-questions 10 \
    --quantize int4 --llm Qwen/Qwen2.5-VL-7B-Instruct \
    --output runs/mvu-eval-tr.jsonl

# M5.4 Gradio UI — NL chat + ffmpeg segment playback
PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True mva ui \
    --db-path runs/matrix.duckdb --chroma-dir runs/matrix-chroma \
    --llm Qwen/Qwen2.5-VL-7B-Instruct --quantize int4 \
    --port 7860 --no-browser
```

### Embedding into another application

`QueryService` is the public facade — own one instance for the
process lifetime, call `.answer(query)` per request. `mva ui` is the
in-tree Gradio reference; the same pattern works for FastAPI /
Streamlit / any other front-end.

```python
from mva.cli import QueryService
from mva.contracts import RichQuery, Attachment
from pathlib import Path

service = QueryService(
    db_path="runs/matrix.duckdb",
    chroma_dir="runs/matrix-chroma",
    llm_model="Qwen/Qwen2.5-VL-7B-Instruct",
    quantization="int4",
)
result = service.answer(RichQuery(
    text="画面里有没有这个人?",
    attachments=[Attachment(kind="image", path=Path("target.jpg"))],
))
print(result.answer)         # NL response
print(result.plan.tool_calls)  # what tools the planner picked
service.close()              # releases GPU + DB
```

### Dataset plug-in

Add a new dataset in three steps:

1. Write `src/mva/datasets/<name>.py` implementing `DatasetAdapter` (see
   `mva.datasets.base` for the Protocol).
2. Register `(YourClass, "DATASETS/<default-root>")` in
   `mva.datasets.registry.ADAPTERS`.
3. `mva ... --dataset <name>` now works across every subcommand.

## Architecture (PLAN.md §3)

```
L6 Interaction    Mode A NL Q&A (✅) + Gradio UI (✅ M5.4)  |  Mode B briefing (🔌 stub, v2+)
L5 World State    DuckDB per-view tables  +  single ChromaDB collection
L4 LLM            Qwen2.5-VL-7B (mock + real, INT4 quantization) + prompt templates
L3 Events         Algorithmic (✅ M3.2 const-vel + loitering/spike) + LLM (✅ M4.2)
L2 Cross-View     Geometric (✅ default) + Appearance (✅ M3.0) + LLM (✅ M4.1)
L1 Per-View       YOLOv11 / YOLOE / YOLO-World detection (✅ open-vocab via --detect-classes) + tracker iou_greedy/bytetrack/botsort (✅ M3.1)
L0 Stream         FileStreamSource + ImageDirStreamSource (OpenCV); RTSP deferred to v2+
L7 HITL           side-channel into L2/L5 (🔌 stub, M6 per LOCK)
```

L1.5 (ReID store) is folded into L5 as `vector_type=reid` (Eng Review 1B).

## Project Structure

```
src/mva/
├── contracts/      Pydantic + dataclass schemas:
│                     Frame, NLQuery, RichQuery, Attachment, Event,
│                     Briefing, ViewObservation, CrossViewLink,
│                     TrajectoryPrediction, Anomaly
├── l0_stream/      FileStreamSource + ImageDirStreamSource
├── l1_perception/  Detector (YOLOv11 / YOLOE / YOLO-World open-vocab) + ByteTracker (iou_greedy / bytetrack / botsort)
├── l2_crossview/   GeometricCrossViewLinker + AppearanceCrossViewLinker
│                   + LLMCrossViewLinker (M4.1 real; ROI hybrid loader)
├── l3_events/      AlgorithmicReasoner (M3.2 const-vel + loitering/spike)
│                   + LLMReasoner (M4.2 detect_anomaly + classify_behavior)
├── l4_llm/         LLMClient (mock + Qwen2.5-VL-7B + INT4/INT8 BnB) +
│                   complete_messages() (multi-video chat-template) +
│                   prompt templates
├── l5_state/       WorldStateStore (DuckDB) + VectorStore (ChromaDB) +
│                   MultimodalEmbedder (Qwen3-VL-Embedding-8B, MRL-768)
├── l6_interaction/ QueryPlan, ToolRegistry (incl. attachment-bound tools),
│                   QueryPlanner, Orchestrator (str | RichQuery), TextInput,
│                   VoiceInput (stub), NullBriefingAgent (stub)
├── l7_hitl/        HumanCorrectionInterface (returns 501; M6 per LOCK)
├── segmentation/   Segment dataclass + iter_video_segments +
│                   iter_image_dir_segments (M2.8)
├── datasets/       DatasetAdapter Protocol + registry +
│                   MatrixDataset + MVUEvalDataset
└── cli/            QueryService facade +
                    ingest / query / ask / eval / ui (M5.4) +
                    perceive / index (legacy, frozen at M2.7) +
                    `python -m mva` / `mva` entry-point

tests/                  (436 passing; pyflakes 0)
├── smoke/              §3.4 interface stubs + RichQuery passthrough
├── contracts/          Pydantic contract tests, parametrized over L2/L3 modes
├── unit/               Per-module behavior (L0 / L1 tracker / L2 geom+app+llm /
│                       L3 algorithmic+llm / L4 / L5 / L6 / datasets / cli)
└── integration/        ingest pipeline + Mode A end-to-end with scripted LLM
```

## Feature Matrix

| Layer / feature | Status | Milestone |
|---|---|---|
| L0 FileStreamSource + ImageDirStreamSource | ✅ | M0–M1 (RTSP live: v2+) |
| L1 YOLOv11 detection (+ YOLOE / YOLO-World open-vocab via `--detect-classes`) | ✅ | M0 (open-vocab 2026-06-02) |
| L1 tracker — iou_greedy / bytetrack / botsort (`--tracker`) | ✅ | M3.1 (botsort 2026-06-02) |
| L2 GeometricCrossViewLinker (synchronized, MATRIX) | ✅ | M1 (M3.0 wired to ingest) |
| L2 AppearanceCrossViewLinker (non-sync, MVU-Eval) | ✅ | M3.0 |
| L2 LLMCrossViewLinker (low-confidence fallback) | ✅ | M4.1 (M4.3 confidence-gated wire-in) |
| L3 AlgorithmicReasoner (const-vel + loitering + speed spike) | ✅ | M3.2 |
| L3 LLMReasoner (detect_anomaly + classify_behavior) | ✅ | M4.2 |
| L4 LLMClient (mock + Qwen2.5-VL-7B, FP16 + INT4/INT8 BnB) | ✅ | M0 / M2.5 quantization |
| L5 DuckDB WorldStateStore | ✅ | M1 (PG migration: v2+) |
| L5 ChromaDB VectorStore (single collection, vector_type metadata) | ✅ | M1 |
| L5 MultimodalEmbedder (Qwen3-VL-Embedding-8B, MRL-768) | ✅ | M2.5 |
| M2.8 unified `mva ingest` + Segmenter + bbox-level embeddings | ✅ | M2.8 |
| L6 Mode A (Planner + Orchestrator + RichQuery + attachment tools) | ✅ | M2 / M2.6 / M2.8 |
| L6 segment-level retrieval tools (find_segment / find_bbox / get_segment_clip) | ✅ | M2.8 |
| L6 Gradio main page (`mva ui`, NL chat + segment playback) | ✅ | M5.4 |
| L6 Mode B briefing | 🔌 stub | v2+ |
| L6 Voice input | 🔌 stub | v2+ |
| L7 HITL (HumanCorrectionInterface) | 🔌 stub | M6 (per LOCK) |
| Eval — MVU-Eval pilot (3-task baseline + OOM mitigation) | ✅ | M5.1-pilot + M5.2-pilot + M5.2-prep |
| Eval — MVU-Eval full (1824) | pending | M5.2-full |
| Eval — Self-built QA scale (26 → 100) | pending | M5.1-scale |
| L4 LoRA SFT | skipped per LOCK | v2+ candidate |

`🔌` = importable stub + smoke test (PLAN.md §3.4) — promote, do not delete.

## Tests

```bash
make test            # all 436
make test-smoke      # §3.4 stub + RichQuery passthrough (fast, no model)
make test-contracts  # Pydantic contract tests (parametrized over modes)
pytest tests/integration -v                  # ingest + Mode A end-to-end (scripted LLM)
pytest tests/unit/test_l2_geometric.py -v    # single file
```

## Datasets

Open datasets live under `DATASETS/` (git-ignored). PLAN.md §2.1 fixes the
order: VisDrone-MDMT for engineering shakedown (M0–M3), MATRIX for formal
eval (M4+), MVU-Eval as a reference benchmark.

## License & Citation

Pre-research code; not yet versioned for external release.
