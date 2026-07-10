import cv2, numpy as np
from mva.service.thumbnails import extract_frame


def _make_video(path, n=20, fps=10, wh=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, wh)
    for i in range(n):
        img = np.full((wh[1], wh[0], 3), i * 10 % 255, np.uint8)
        vw.write(img)
    vw.release()


def test_extract_frame(tmp_path):
    vid = str(tmp_path / "v.mp4")
    _make_video(vid)
    out = str(tmp_path / "thumb.jpg")
    res = extract_frame(vid, t_sec=0.5, out_path=out)
    assert res == out
    img = cv2.imread(out)
    assert img is not None and img.shape[0] > 0


def test_extract_frame_bad_path(tmp_path):
    assert extract_frame("/no/such.mp4", 0.0, str(tmp_path / "x.jpg")) is None
