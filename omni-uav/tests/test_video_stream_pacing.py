import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
import cv2
import numpy as np
from PyQt5 import QtWidgets
from widgets.video_stream import VideoStream

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_video(path, n=20, fps=5, wh=(32, 24)):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, wh)
    for i in range(n):
        vw.write(np.full((wh[1], wh[0], 3), i * 10 % 255, np.uint8))
    vw.release()


def test_paces_to_native_fps(tmp_path):
    vid = str(tmp_path / "v.mp4")
    _make_video(vid, n=20, fps=5)          # 5fps → 每 200ms 才该出下一帧
    s = VideoStream(vid, "cam01")
    assert abs(s.native_fps - 5.0) < 0.5
    assert s.read() is not None            # 首帧立即前进
    assert s.read() is None                # 立刻再读:未到 200ms → 节流(None=无新帧)
    time.sleep(0.25)
    assert s.read() is not None            # 250ms 后:该前进
    s.close()
