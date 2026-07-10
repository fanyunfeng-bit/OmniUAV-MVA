import cv2, numpy as np
from mva.perception.frame_source import UniformFrameSource, FrameSource


def _make_video(path, n=30, fps=10, wh=(64, 48)):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, wh)
    for i in range(n):
        vw.write(np.full((wh[1], wh[0], 3), i, np.uint8))
    vw.release()


def test_uniform_frame_source_density(tmp_path):
    vid = str(tmp_path / "v.mp4"); _make_video(vid, n=30, fps=10)   # 3s@10fps
    fs = UniformFrameSource(vid, target_fps=5)                      # 期望 ~15 帧
    frames = list(fs.iter_frames())
    assert isinstance(fs, FrameSource)          # 满足 Protocol
    assert 12 <= len(frames) <= 18
    ts = [t for t, _ in frames]
    assert ts == sorted(ts) and ts[0] >= 0.0    # 时间递增、绝对时间基准
