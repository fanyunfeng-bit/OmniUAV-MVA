# sim/ — 仿真侧启动（AirSim 实时数据源）

本目录只放**启动脚本**。仿真的重型运行时（UE4 构建、docker 镜像、ROS 工作区）
是**外部前置**，和模型权重/数据集一样不进 git，按下表在目标机上准备。

```
sim/
├── start_live_demo.sh   # 起 UE4 + 容器ROS(airsim_node + rosbridge:9090 + planner + patrol)，不含前端
├── stop_live_demo.sh    # 停 UE4 + 容器（默认 docker stop，释放显存）
└── README.md
```

> 一般不单独调这两个脚本，而是用仓库根的一条龙：
> `bash scripts/start_live_mva.sh`（sim + sidecar + MVA 前端）/ `bash scripts/stop_live_mva.sh`。
> 这里的脚本是它内部调用的「仿真侧」部分，也可单独用来只起/停仿真。

## 外部前置（不在本仓库，需自备）

| 前置 | 默认位置 / 名称 | 覆盖用环境变量 | 说明 |
|---|---|---|---|
| ROS 工作区 | `/home/fyf/fyf/PCL/UAV_Sim_Workspace` | `SIM_WS` | 内含 `Agent_volume`（挂载到容器 `/root`：`airsim_swarm/devel`、`_patrol.py`、`_stop_ros.sh`）|
| UE4 构建 | `$SIM_WS/AirSim/UAV-ON-envs-test/DownTown` | `SIM_UE4_DIR` | GPU 场景二进制（GB 级）；启动脚本 `DownTown_test1.sh`（`SIM_UE4_LAUNCH`）|
| docker 镜像 | `noetic:v1.5` | `SIM_IMAGE` | ROS Noetic + airsim_ros_pkgs + ego_planner + rosbridge_server 的构建镜像 |
| 容器名 | `uav_sim_live` | `SIM_CONTAINER` | 首次自动 `docker run`，之后 `docker restart` 干净重来 |

> 仿真 ROS 源码（`airsim_swarm`、`situation_awareness*` 等）是上面 **docker 镜像/工作区**的构建源，
> 运行在容器内、不在主机 python 环境用，故不 vendored 进本仓库；由镜像/工作区一并提供。

## 换机器部署

```bash
# 把外部路径指到目标机的实际位置（不改脚本）：
export SIM_WS=/data/UAV_Sim_Workspace
export SIM_IMAGE=noetic:v1.5
bash scripts/start_live_mva.sh          # 或 sim/start_live_demo.sh
```

## 数据流

```
UE4 DownTown(GPU, :41451) → 容器 airsim_node → /airsim_node/drone{1-4}/front_center_custom/Scene
                                              → rosbridge_server(:9090)
                                              → OmniUAV-MVA 前端(roslibpy 订阅) + sidecar 分析
```
