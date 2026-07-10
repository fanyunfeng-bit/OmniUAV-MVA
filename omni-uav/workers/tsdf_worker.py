from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore


class TsdfReconstructionWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(object, object, object, int, int)

    def __init__(
        self,
        data_dir: Path,
        output_mesh: Path,
        step_size: int = 2,
        voxel_size: float = 0.025,
        live_preview: bool = True,
        preview_stride: int = 50,
        parent=None,
    ):
        super().__init__(parent)
        self.data_dir = data_dir
        self.output_mesh = output_mesh
        self.step_size = max(step_size, 1)
        self.voxel_size = voxel_size
        self.live_preview = live_preview
        self.preview_stride = max(preview_stride, 1)

    def run(self):
        try:
            mesh_path = self._run_fusion()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(mesh_path)

    def _run_fusion(self) -> str:
        data_dir = self.data_dir
        transforms_path = data_dir / "transforms.csv"
        depth_dir = data_dir / "depth"
        rgb_dir = data_dir / "rgb"
        if not transforms_path.exists():
            raise ValueError("缺少 transforms.csv")
        if not depth_dir.exists() or not rgb_dir.exists():
            raise ValueError("缺少 depth/ 或 rgb/ 目录")

        import torch
        from utils import TSDFVolume, get_camera_frustum, save_mesh

        transforms = np.loadtxt(
            transforms_path, skiprows=1, delimiter=",", dtype=np.float32
        )
        image_numbers = transforms[:, 0].astype(np.int32)
        poses = transforms[:, 1:-4].reshape(-1, 3, 4)
        fx, fy, cx, cy = transforms[0, -4:]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        K_t = torch.from_numpy(K)
        poses_t = torch.from_numpy(poses)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        volume_bounds = torch.zeros((3, 2))
        for idx in range(image_numbers.shape[0]):
            image_stem = f"{int(image_numbers[idx]):05d}"
            depth_path = depth_dir / f"{image_stem}.png"
            depth = cv2.imread(str(depth_path), -1)
            if depth is None:
                raise ValueError(f"无法读取深度图：{depth_path}")
            depth_t = torch.from_numpy(depth.astype(np.float32))
            depth_t /= 1000.0
            depth_t[depth_t == 65.535] = 0
            points_3d = get_camera_frustum(depth_t, K_t, poses_t[idx])
            volume_bounds[:, 0] = torch.minimum(
                volume_bounds[:, 0], torch.amin(points_3d, axis=0)
            )
            volume_bounds[:, 1] = torch.maximum(
                volume_bounds[:, 1], torch.amax(points_3d, axis=0)
            )

        tsdf_vol = TSDFVolume(
            volume_bounds, voxel_size=self.voxel_size, device=device
        )

        total_frames = image_numbers.shape[0]
        for idx in range(0, total_frames, self.step_size):
            image_stem = f"{int(image_numbers[idx]):05d}"
            rgb_path = rgb_dir / f"{image_stem}.jpg"
            if not rgb_path.exists():
                rgb_path = rgb_dir / f"{image_stem}.png"
            bgr = cv2.imread(str(rgb_path), -1)
            if bgr is None:
                raise ValueError(f"无法读取彩色图：{rgb_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb_t = torch.from_numpy(rgb)

            depth_path = depth_dir / f"{image_stem}.png"
            depth = cv2.imread(str(depth_path), -1)
            if depth is None:
                raise ValueError(f"无法读取深度图：{depth_path}")
            depth_t = torch.from_numpy(depth.astype(np.float32))
            depth_t /= 1000.0
            depth_t[depth_t == 65.535] = 0

            tsdf_vol.fuse_frame(K_t, poses_t[idx], rgb_t, depth_t, weight=1.0)
            if self.live_preview and (
                (idx % self.preview_stride) == 0 or idx + self.step_size >= total_frames
            ):
                v, f, _, c = tsdf_vol.extract_mesh()
                self.progress.emit(v, f, c, min(idx + self.step_size, total_frames), total_frames)

        v, f, _, c = tsdf_vol.extract_mesh()
        self.output_mesh.parent.mkdir(parents=True, exist_ok=True)
        save_mesh(str(self.output_mesh), v, f, c)
        return str(self.output_mesh)
