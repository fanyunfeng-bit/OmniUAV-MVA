#!/bin/bash
# 停止「仿真实时分析」链路：前端 + MVA sidecar + 仿真/ROS/容器。
MVA=$(cd "$(dirname "$0")/.." && pwd)        # 仓库根
pkill -9 -f 'app[.]py' 2>/dev/null           # 前端
bash "$MVA/scripts/stop_mva_sidecar.sh"      # sidecar
bash "$MVA/sim/stop_live_demo.sh" "$@"       # 仿真 + ROS + 容器
echo "✅ 已停止：前端 + sidecar + 仿真"
