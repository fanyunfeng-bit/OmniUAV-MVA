#!/bin/bash
# [MOD 2026-07-10 | P0] 停止 MVA sidecar(FastAPI :8900)。
# 正则 mva[.]service 打断字面量，避免 pkill -f 匹配到本命令自身。
pkill -9 -f 'mva[.]service' 2>/dev/null
echo "已停止 MVA sidecar (:8900)"
