#!/bin/bash
# 一键启动「仿真实时分析」链路：仿真(4机) + MVA sidecar + MVA前端(ROS实时)。
#   AirSim(UE4) → airsim_node → rosbridge:9090 → OmniUAV-MVA 前端 + sidecar 分析
#
# 用法: bash scripts/start_live_mva.sh              # 4 机起飞巡逻(默认)
#       bash scripts/start_live_mva.sh --no-patrol  # 只起飞/悬停
#       bash scripts/start_live_mva.sh --no-fly     # 不飞，仅 4 路静态实时画面
# 停止: bash scripts/stop_live_mva.sh
set -u
MVA=$(cd "$(dirname "$0")/.." && pwd)        # 仓库根（脚本在 scripts/ 下）
SIMSYS_PY=${SIMSYS_PY:-/home/fyf/miniconda3/envs/simsys/bin/python}
LOGDIR=/tmp/sim_live_logs; mkdir -p "$LOGDIR"

echo "==================================================================="
echo " [1/3] 仿真 + ROS + rosbridge(:9090) + 4 机  (sim/)"
echo "==================================================================="
bash "$MVA/sim/start_live_demo.sh" "$@"      # 起 UE4 + airsim_node + rosbridge + planner + patrol（无前端）

echo "==================================================================="
echo " [2/3] MVA sidecar(分析引擎 :8900)"
echo "==================================================================="
bash "$MVA/scripts/start_mva_sidecar.sh"
echo -n "      等待 sidecar 就绪(首次加载嵌入约60s)"
for _ in $(seq 1 40); do
  curl -s --max-time 2 http://127.0.0.1:8900/health 2>/dev/null | grep -q '"engine_ready":true' \
    && { echo " ✓"; break; }
  echo -n "."; sleep 3
done

echo "==================================================================="
echo " [3/3] OmniUAV-MVA 前端(ROS 实时，自动连 rosbridge、订阅 4 路)"
echo "==================================================================="
( cd "$MVA/omni-uav" &&
  DISPLAY=:0.0 OMNIUAV_ROS_LIVE=1 ROSBRIDGE_HOST=localhost ROSBRIDGE_PORT=9090 \
    nohup "$SIMSYS_PY" -u app.py > "$LOGDIR/omniuav_mva.log" 2>&1 & )
sleep 3
echo ""
echo "✅ 实时链路启动完成：OmniUAV-MVA 窗口应在屏幕上(ROS实时，4 路无人机画面)。"
echo "   问答/检索经 sidecar；日志 $LOGDIR/   停止: bash $MVA/scripts/stop_live_mva.sh"
