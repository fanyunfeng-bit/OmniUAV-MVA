#!/usr/bin/env python3
"""录制 AirSim 4 无人机前视相机(经 rosbridge ws:9090)为 4 路本地视频。

用法:
  python record_airsim_4view.py --duration 180 --out ~/OmniUAV-MVA-data/airsim_downtown_4view

前提:仿真已启动(UE4 + airsim_node + rosbridge + planner + patrol，无人机在飞)。
每路输出 camNN.mp4；fps 按实际采集率写(AirSim 4 机 ~2.5-3.5Hz)。
"""
import argparse
import base64
import os
import time

import cv2
import numpy as np
import roslibpy

DRONES = [1, 2, 3, 4]
TOPIC = "/airsim_node/drone{}/front_center_custom/Scene/compressed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--out", default=os.path.expanduser("~/OmniUAV-MVA-data/airsim_downtown_4view"))
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()
    for _ in range(50):
        if client.is_connected:
            break
        time.sleep(0.1)
    print(f"rosbridge connected: {client.is_connected}")
    if not client.is_connected:
        raise SystemExit("无法连接 rosbridge :9090 —— 仿真是否已启动?")

    buffers = {d: [] for d in DRONES}       # d -> list[(t, jpeg_bytes)]

    def make_cb(d):
        def cb(msg):
            try:
                buffers[d].append((time.time(), base64.b64decode(msg["data"])))
            except Exception:               # noqa: BLE001
                pass
        return cb

    topics = []
    for d in DRONES:
        t = roslibpy.Topic(client, TOPIC.format(d), "sensor_msgs/CompressedImage")
        t.subscribe(make_cb(d))
        topics.append(t)

    print(f"recording {args.duration:.0f}s ...")
    t0 = time.time()
    while time.time() - t0 < args.duration:
        time.sleep(5)
        print(f"  t={time.time()-t0:5.0f}s  frames={{{', '.join(f'd{d}:{len(buffers[d])}' for d in DRONES)}}}")

    for t in topics:
        t.unsubscribe()
    client.terminate()

    print("写视频 …")
    for d in DRONES:
        frames = buffers[d]
        if not frames:
            print(f"  drone{d}: 0 帧, 跳过")
            continue
        elapsed = (frames[-1][0] - frames[0][0]) or 1.0
        fps = max(1.0, len(frames) / elapsed)
        img0 = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        if img0 is None:
            print(f"  drone{d}: 首帧解码失败, 跳过")
            continue
        h, w = img0.shape[:2]
        path = os.path.join(out, f"cam{d:02d}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        for _, raw in frames:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            vw.write(img)
            n += 1
        vw.release()
        print(f"  drone{d}: {n} 帧, fps={fps:.2f}, {w}x{h} -> {path}")
    print("完成:", out)


if __name__ == "__main__":
    main()
