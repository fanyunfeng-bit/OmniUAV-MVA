#!/bin/bash
# [MOD 2026-07-10 | P0] 在 MVA env 内拉起 sidecar(FastAPI :8900)。
#   需先: MVA env 已 pip install -e 'mva[service]'；export DASHSCOPE_API_KEY=...
# 环境变量:
#   MVA_PY      MVA env 的 python (默认 /home/fyf/miniconda3/envs/mva/bin/python)
#   MVA_DB      DuckDB 世界状态库 (默认 /tmp/mva/world.duckdb)
#   MVA_CHROMA  ChromaDB 目录 (默认 /tmp/mva/chroma；设为空串 "" 则不加载嵌入, 轻量启动)
set -u
MVA_PY=${MVA_PY:-/home/fyf/miniconda3/envs/mva/bin/python}
# 默认用持久位置 ~/.omniuav-mva（重启不丢；GUI 入库/检索/问答都用这同一个库）。
# 想换库再显式设 MVA_DB / MVA_CHROMA。
DB=${MVA_DB:-$HOME/.omniuav-mva/world.duckdb}
CHROMA=${MVA_CHROMA-$HOME/.omniuav-mva/chroma}   # 用 - : 允许显式空串跳过 chroma
LOGDIR=/tmp/sim_live_logs; mkdir -p "$LOGDIR" "$(dirname "$DB")"

# API key：优先用环境变量；没有就从本地(gitignore 的)配置自动读取，免得每次手动 export。
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  for CFG in "$SCRIPT_DIR/../omni-uav/configs/config.json" "$SCRIPT_DIR/../omni-uav/configs/config_llm.yaml"; do
    if [ -f "$CFG" ]; then
      K=$(grep -oE 'sk-[A-Za-z0-9]{20,}' "$CFG" | head -1)
      [ -n "$K" ] && { export DASHSCOPE_API_KEY="$K"; echo "已从 $(basename "$CFG") 读取 API key"; break; }
    fi
  done
fi
: "${DASHSCOPE_API_KEY:?未找到 key：请 export DASHSCOPE_API_KEY，或把 key 写进 omni-uav/configs/config.json}"

if ( ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null ) | grep -q ":8900"; then
  echo "sidecar 已在运行 (:8900)"; exit 0
fi

ARGS=(--db "$DB")
if [ -n "$CHROMA" ]; then mkdir -p "$CHROMA"; ARGS+=(--chroma-dir "$CHROMA"); fi

nohup "$MVA_PY" -m mva.service "${ARGS[@]}" > "$LOGDIR/mva_sidecar.log" 2>&1 &
echo "sidecar 启动中… (db=$DB chroma=${CHROMA:-<none>})"
echo "  日志: $LOGDIR/mva_sidecar.log   探活: curl http://127.0.0.1:8900/health"
