#!/bin/bash
# [MOD 2026-07-10 | P0] 在 MVA env 内拉起 sidecar(FastAPI :8900)。
#   需先: MVA env 已 pip install -e 'mva[service]'；export DASHSCOPE_API_KEY=...
# 环境变量:
#   MVA_PY      MVA env 的 python (默认 /home/fyf/miniconda3/envs/mva/bin/python)
#   MVA_DB      DuckDB 世界状态库 (默认 /tmp/mva/world.duckdb)
#   MVA_CHROMA  ChromaDB 目录 (默认 /tmp/mva/chroma；设为空串 "" 则不加载嵌入, 轻量启动)
set -u
MVA_PY=${MVA_PY:-/home/fyf/miniconda3/envs/mva/bin/python}
DB=${MVA_DB:-/tmp/mva/world.duckdb}
CHROMA=${MVA_CHROMA-/tmp/mva/chroma}     # 用 - : 允许显式空串跳过 chroma
LOGDIR=/tmp/sim_live_logs; mkdir -p "$LOGDIR" "$(dirname "$DB")"
: "${DASHSCOPE_API_KEY:?请先 export DASHSCOPE_API_KEY=<你的key>}"

if ( ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null ) | grep -q ":8900"; then
  echo "sidecar 已在运行 (:8900)"; exit 0
fi

ARGS=(--db "$DB")
if [ -n "$CHROMA" ]; then mkdir -p "$CHROMA"; ARGS+=(--chroma-dir "$CHROMA"); fi

nohup "$MVA_PY" -m mva.service "${ARGS[@]}" > "$LOGDIR/mva_sidecar.log" 2>&1 &
echo "sidecar 启动中… (db=$DB chroma=${CHROMA:-<none>})"
echo "  日志: $LOGDIR/mva_sidecar.log   探活: curl http://127.0.0.1:8900/health"
