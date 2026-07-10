# [MOD 2026-07-09 | 步骤1 真·实时接入] 新增文件
# 目的：让主机上的 omni-uav 不依赖 rospy 也能实时订阅仿真的 ROS 相机话题。
# 方案：通过 rosbridge_server(容器内, ws://localhost:9090) + roslibpy(纯 Python) 订阅
#       压缩图像话题(sensor_msgs/CompressedImage, JPEG)，解码为 QImage。
# 与 ros_live_stream.ROSLiveStream(基于 rospy) 接口保持一致，可在 camera_tab 中互换使用。
# 记录见 MODIFICATIONS.md 步骤1。
from typing import Optional
import os
import base64
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtGui

try:
    import roslibpy
    ROSBRIDGE_AVAILABLE = True
except ImportError:
    ROSBRIDGE_AVAILABLE = False
    print("Warning: roslibpy not available. Install with: pip install roslibpy "
          "(and run rosbridge_server in the ROS side)")


class RosBridgeLiveStream:
    """通过 rosbridge(websocket) + roslibpy 实时订阅 ROS 图像话题。

    与 ROSLiveStream 接口一致：read()/get_latest()/is_alive()/close()/resubscribe() 等，
    因此 camera_tab 可以无差别使用。

    Args:
        topic: 原始图像话题名(如 "/airsim_node/drone1/front_center_custom/Scene")。
               本类会优先订阅其 "/compressed"(JPEG, 带宽低)；解码失败再回退到原始 Image。
        camera_id: 相机标识
        host/port: rosbridge 地址(默认 localhost:9090；容器 --net host 时主机可直连)
    """

    def __init__(self, topic: str, camera_id: str,
                 host: Optional[str] = None, port: Optional[int] = None):
        if not ROSBRIDGE_AVAILABLE:
            raise ImportError("roslibpy 未安装。请: pip install roslibpy")

        self.base_topic = topic.rstrip("/")
        self.camera_id = camera_id
        self.host = host or os.getenv("ROSBRIDGE_HOST", "localhost")
        self.port = int(port or os.getenv("ROSBRIDGE_PORT", "9090"))

        self.last_frame: Optional[np.ndarray] = None   # RGB numpy
        self.last_image: Optional[QtGui.QImage] = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_timestamp: Optional[float] = None
        self._start_time: Optional[float] = None

        # 建立 rosbridge 连接(roslibpy 自带后台事件循环线程)
        self._ros = roslibpy.Ros(host=self.host, port=self.port)
        self._ros.run()  # 非阻塞，后台线程运行

        self._topic_obj = None
        self._use_compressed = True
        self._subscribe(self.base_topic)
        print(f"[RosBridge] Subscribed to {self._current_topic_name()} via ws://{self.host}:{self.port}")

    # ---- 内部 ----
    def _current_topic_name(self) -> str:
        return self.base_topic + ("/compressed" if self._use_compressed else "")

    def _subscribe(self, base_topic: str):
        """订阅 <base>/compressed(CompressedImage)；roslibpy 会把 data 作为 base64 传回。"""
        self.base_topic = base_topic.rstrip("/")
        name = self._current_topic_name()
        msg_type = "sensor_msgs/CompressedImage" if self._use_compressed else "sensor_msgs/Image"
        self._topic_obj = roslibpy.Topic(self._ros, name, msg_type, queue_length=1, throttle_rate=0)
        self._topic_obj.subscribe(self._on_msg)

    def _decode_bytes(self, data) -> Optional[bytes]:
        """rosbridge 的 uint8[] 字段可能是 base64 字符串 / int 列表 / bytes。"""
        try:
            if isinstance(data, str):
                return base64.b64decode(data)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, list):
                return bytes(data)
        except Exception as e:
            print(f"[RosBridge] decode data error: {e}")
        return None

    def _on_msg(self, message):
        """收到一帧：CompressedImage(JPEG) 用 imdecode；原始 Image 按 encoding reshape。"""
        try:
            raw = self._decode_bytes(message.get("data"))
            if raw is None:
                return
            if self._use_compressed:
                arr = np.frombuffer(raw, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    return
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                h, w = int(message["height"]), int(message["width"])
                enc = message.get("encoding", "bgr8")
                buf = np.frombuffer(raw, np.uint8)
                if enc == "rgb8":
                    rgb = buf.reshape(h, w, 3)
                elif enc == "mono8":
                    rgb = cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2RGB)
                else:  # bgr8 及其它默认按 bgr 处理
                    rgb = cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_BGR2RGB)

            with self._lock:
                self.last_frame = rgb
                self._frame_count += 1
                self._last_timestamp = time.time()
                if self._start_time is None:
                    self._start_time = self._last_timestamp
            if self._frame_count == 1:
                print(f"[RosBridge] First frame on {self._current_topic_name()}: {rgb.shape}")
        except Exception as e:
            print(f"[RosBridge] on_msg error ({self._current_topic_name()}): {e}")

    # ---- 与 ROSLiveStream 一致的公共接口 ----
    def read(self) -> Optional[QtGui.QImage]:
        with self._lock:
            if self.last_frame is None:
                return None
            f = self.last_frame
            h, w = f.shape[0], f.shape[1]
            image = QtGui.QImage(f.data, w, h, f.strides[0], QtGui.QImage.Format_RGB888).copy()
            self.last_image = image
            return image

    def get_latest(self) -> Optional[QtGui.QImage]:
        return self.last_image

    def has_frames(self) -> bool:
        with self._lock:
            return self._frame_count > 0

    def get_frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def get_fps(self) -> float:
        with self._lock:
            if self._start_time is None or self._frame_count < 2:
                return 0.0
            elapsed = max(time.time() - self._start_time, 1e-3)
            return self._frame_count / elapsed

    def is_alive(self) -> bool:
        try:
            return bool(self._ros and self._ros.is_connected)
        except Exception:
            return False

    def resubscribe(self, new_topic: str) -> bool:
        try:
            if self._topic_obj:
                self._topic_obj.unsubscribe()
            with self._lock:
                self._frame_count = 0
                self._last_timestamp = None
                self._start_time = None
                self.last_frame = None
                self.last_image = None
            self._use_compressed = True
            self._subscribe(new_topic)
            print(f"[RosBridge] Resubscribed to {self._current_topic_name()}")
            return True
        except Exception as e:
            print(f"[RosBridge] resubscribe error: {e}")
            return False

    def close(self):
        try:
            if self._topic_obj:
                self._topic_obj.unsubscribe()
        except Exception:
            pass
        try:
            if self._ros:
                self._ros.terminate()
        except Exception:
            pass
        with self._lock:
            self.last_frame = None
            self.last_image = None
        print(f"[RosBridge] Closed {self.base_topic}")
