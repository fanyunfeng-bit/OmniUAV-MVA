#!/bin/bash
# =============================================================================
# 停止【仿真侧】：UE4(host) + 整个 ROS 容器(docker stop，含占显存节点)。
#   默认 docker stop 容器，彻底释放显存；下次 start 会 docker start 并全新拉起 ROS。
#
# 环境变量：SIM_CONTAINER(默认 uav_sim_live)、SIM_UE4_PROC(默认 DownTown_test1)
# 用法: bash sim/stop_live_demo.sh                  # 全停(含 docker stop，释放显存)
#       bash sim/stop_live_demo.sh --keep-container # 只停容器内 ROS 节点(下次秒起)
# =============================================================================
set -u
CONTAINER=${SIM_CONTAINER:-uav_sim_live}
UE4_PROC=${SIM_UE4_PROC:-DownTown_test1}
KEEP=0
for a in "$@"; do [ "$a" = "--keep-container" ] && KEEP=1; done

echo "停止 UE4 DownTown (host)…"
pkill -9 -f "$UE4_PROC" 2>/dev/null
pkill -9 -f UE4_Downtown 2>/dev/null

if [ "$KEEP" = 1 ]; then
  echo "停止 容器内 ROS 节点 (保留容器)…"
  docker exec "$CONTAINER" bash /root/_stop_ros.sh 2>/dev/null
  echo "✅ 已停止 (UE4/ROS；容器保留，下次秒起)"
else
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "docker stop $CONTAINER (清空容器内全部 ROS 进程并释放显存)…"
    docker stop "$CONTAINER" >/dev/null 2>&1
  else
    echo "容器 $CONTAINER 未在运行，跳过。"
  fi
  echo "✅ 已停止 (UE4/ROS + 容器已 stop，显存已归还)"
fi

echo ""
echo "GPU 占用核查:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
  | grep -v '^$' || echo "  (无 compute 进程占用 GPU) ✓"
