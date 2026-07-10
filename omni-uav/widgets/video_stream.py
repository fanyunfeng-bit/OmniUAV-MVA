from pathlib import Path
from typing import List, Optional, Protocol

import cv2
from PyQt5 import QtGui


class StreamBase(Protocol):
    def read(self) -> Optional[QtGui.QImage]:
        ...

    def get_latest(self) -> Optional[QtGui.QImage]:
        ...

    def close(self):
        ...


class VideoStream:
    def __init__(
        self,
        path: Path,
        camera_id: str,
    ):
        self.path = path
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(str(path))
        self.frame_index = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.last_image: Optional[QtGui.QImage] = None

    def read(self) -> Optional[QtGui.QImage]:
        if not self.cap.isOpened():
            return None

        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            ok, frame = self.cap.read()
            if not ok:
                return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        image = QtGui.QImage(
            frame_rgb.data,
            width,
            height,
            frame_rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()

        self.frame_index += 1
        if self.total_frames:
            self.frame_index %= self.total_frames

        self.last_image = image
        return image

    def get_latest(self) -> Optional[QtGui.QImage]:
        return self.last_image

    def close(self):
        if self.cap:
            self.cap.release()


class ImageSequenceStream:
    def __init__(self, folder: Path, camera_id: str):
        self.folder = folder
        self.camera_id = camera_id
        patterns = ["*.jpg", "*.jpeg", "*.png"]
        images: List[Path] = []
        for pattern in patterns:
            images.extend(sorted(folder.glob(pattern)))
        self.images = images
        self.frame_index = 0
        self.last_image: Optional[QtGui.QImage] = None

    def read(self) -> Optional[QtGui.QImage]:
        if not self.images:
            return None
        path = self.images[self.frame_index % len(self.images)]
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        image = QtGui.QImage(
            frame_rgb.data,
            width,
            height,
            frame_rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        self.frame_index = (self.frame_index + 1) % len(self.images)
        self.last_image = image
        return image

    def get_latest(self) -> Optional[QtGui.QImage]:
        return self.last_image

    def close(self):
        return
