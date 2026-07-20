#!/bin/bash
# =============================================================================
# 启动【仿真侧】：多无人机仿真 infra（UE4 + 容器ROS + rosbridge:9090），不含前端。
#   UE4 DownTown(GPU) → 容器ROS(airsim_node + rosbridge + planner + patrol)
#   分析前端 + sidecar 由 scripts/start_live_mva.sh 负责（本脚本只管仿真+ROS）。
#
# 外部前置（不在本仓库，见 sim/README.md）：docker 镜像、UE4 构建、ROS 工作区。
# 可用环境变量覆盖路径（默认对应本机既有安装）：
#   SIM_WS         ROS 工作区根 (默认 /home/fyf/fyf/PCL/UAV_Sim_Workspace)
#   SIM_AGENT_VOL  容器挂到 /root 的卷 (默认 $SIM_WS/Agent_volume)
#   SIM_UE4_DIR    UE4 构建目录 (默认 $SIM_WS/AirSim/UAV-ON-envs-test/DownTown)
#   SIM_UE4_LAUNCH UE4 启动脚本名 (默认 DownTown_test1.sh)
#   SIM_UE4_PROC   UE4 进程名(用于 pkill) (默认 DownTown_test1)
#   SIM_CONTAINER  容器名 (默认 uav_sim_live)
#   SIM_IMAGE      docker 镜像 (默认 noetic:v1.5)
#
# 用法: bash sim/start_live_demo.sh              # 4 机起飞并巡逻(默认)
#       bash sim/start_live_demo.sh --no-patrol  # 只起飞/悬停
#       bash sim/start_live_demo.sh --no-fly     # 不飞，仅 4 路静态实时画面
#       bash sim/start_live_demo.sh --restart-ue4 # 顺带重启 UE4(彻底干净)
# 停止: bash sim/stop_live_demo.sh
# =============================================================================
set -u
CONTAINER=${SIM_CONTAINER:-uav_sim_live}
IMAGE=${SIM_IMAGE:-noetic:v1.5}
WS=${SIM_WS:-/home/fyf/fyf/PCL/UAV_Sim_Workspace}
AGENT_VOL=${SIM_AGENT_VOL:-$WS/Agent_volume}
UE4_DIR=${SIM_UE4_DIR:-$WS/AirSim/UAV-ON-envs-test/DownTown}
UE4_LAUNCH=${SIM_UE4_LAUNCH:-DownTown_test1.sh}
UE4_PROC=${SIM_UE4_PROC:-DownTown_test1}
LOGDIR=/tmp/sim_live_logs; mkdir -p "$LOGDIR"
RS='source /opt/ros/noetic/setup.bash 2>/dev/null; source /root/airsim_swarm/devel/setup.bash 2>/dev/null; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY'

DO_FLY=1; DO_PATROL=1; RESTART_UE4=0
for a in "$@"; do
  [ "$a" = "--no-fly" ] && DO_FLY=0
  [ "$a" = "--no-patrol" ] && DO_PATROL=0
  [ "$a" = "--restart-ue4" ] && RESTART_UE4=1
done

port_open(){ ( ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null ) | grep -q ":$1"; }
wait_port(){ local p=$1 n=${2:-60}; for _ in $(seq 1 "$n"); do port_open "$p" && return 0; sleep 2; done; return 1; }
dex(){ docker exec "$CONTAINER" bash -lc "$RS; $1"; }
dexd(){ docker exec -d "$CONTAINER" bash -lc "$RS; $1"; }
scene_cnt(){ dex 'rostopic list 2>/dev/null | grep -c front_center_custom/Scene' 2>/dev/null; }

echo "==================================================================="
echo " 启动 多无人机仿真 infra (fly=$DO_FLY patrol=$DO_PATROL)"
echo "==================================================================="

# ---- 1) UE4 DownTown (GPU) ----
echo "[1/6] UE4 DownTown (GPU, headless)…"
if [ "$RESTART_UE4" = 1 ] && port_open 41451; then
  echo "      --restart-ue4: 关闭旧 UE4…"; pkill -9 "$UE4_PROC" 2>/dev/null; sleep 3
fi
if port_open 41451; then
  echo "      已在运行 (RPC :41451)"
else
  ( export DISPLAY=:0 __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only \
           __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    cd "$UE4_DIR" &&
    bash "$UE4_LAUNCH" -windowed -ResX=256 -ResY=256 -nosound -RenderOffScreen -vulkan \
      > "$LOGDIR/ue4.log" 2>&1 & )
  wait_port 41451 60 && echo "      AirSim RPC :41451 ✓" || { echo "      ✗ UE4 启动超时，见 $LOGDIR/ue4.log"; exit 1; }
  sleep 3
fi

# ---- 2) 容器：干净重来 ----
echo "[2/6] ROS 容器 $CONTAINER (干净重启)…"
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker restart "$CONTAINER" >/dev/null
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker start "$CONTAINER" >/dev/null
else
  docker run -d --name "$CONTAINER" --gpus all --net host --ipc host \
    -e DISPLAY=:0 -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$AGENT_VOL:/root" -v /tmp/.X11-unix:/tmp/.X11-unix "$IMAGE" sleep infinity >/dev/null
fi
sleep 2
DISPLAY=:0 xhost +local: >/dev/null 2>&1
echo "      容器就绪(全新 ROS 环境)"

# ---- 3) airsim_node 桥接 ----
echo "[3/6] airsim_node 桥接…"
dexd 'roslaunch airsim_ros_pkgs airsim_node.launch > /root/airsim_node.log 2>&1'
for _ in $(seq 1 25); do [ "$(scene_cnt)" -ge 1 ] 2>/dev/null && break; sleep 2; done
z=$(dex 'timeout 4 rostopic echo -n1 /airsim_node/drone1/odom_local_enu/pose/pose/position 2>/dev/null | grep -m1 x: | awk "{print \$2}"')
echo "      相机话题就绪 ✓ (drone1 x=$z)"

# ---- 4) rosbridge (9090) ----
echo "[4/6] rosbridge_server (:9090)…"
dexd 'roslaunch rosbridge_server rosbridge_websocket.launch > /root/rosbridge.log 2>&1'
wait_port 9090 15 && echo "      rosbridge :9090 ✓" || echo "      ✗ rosbridge 未起来"

# ---- 5) planner ----
if [ "$DO_FLY" = 1 ]; then
  echo "[5/6] planner (rviz:=false, 无人机起飞)…"
  dexd 'cd /root/airsim_swarm; roslaunch ego_planner multi_drone_interactive.launch rviz:=false > /root/planner.log 2>&1'
  sleep 16; echo "      planner 起飞中 ✓"
else
  echo "[5/6] planner 跳过 (--no-fly)"
fi

# ---- 6) patrol ----
if [ "$DO_FLY" = 1 ] && [ "$DO_PATROL" = 1 ]; then
  echo "[6/6] patrol (巡逻)…"
  dexd 'python3 /root/_patrol.py > /root/patrol.log 2>&1'
  echo "      巡逻已启动 ✓"
else
  echo "[6/6] patrol 跳过"
fi

echo ""
echo "==================================================================="
echo " ✅ 仿真 infra 就绪：4 路 /airsim_node/drone{1-4}/front_center_custom/Scene + rosbridge:9090"
echo "    起分析：bash scripts/start_live_mva.sh   （sim + sidecar + MVA前端一条龙）"
echo "    日志: $LOGDIR/   停止: bash sim/stop_live_demo.sh"
echo "==================================================================="
