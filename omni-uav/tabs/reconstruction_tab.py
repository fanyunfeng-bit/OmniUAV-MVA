from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets

from utils import read_ply
from workers import TsdfReconstructionWorker


class PlyMeshTab(QtWidgets.QWidget):
    def __init__(self, data_dir: Path, output_mesh: Path, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.output_mesh = output_mesh
        self.worker: Optional[TsdfReconstructionWorker] = None
        self.mesh_item: Optional[gl.GLMeshItem] = None
        self.scatter_item: Optional[gl.GLScatterPlotItem] = None
        self._build_ui()
        if self.output_mesh.exists():
            self._load_mesh(str(self.output_mesh))

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()

        self.recon_btn = QtWidgets.QPushButton("开始重建")
        self.recon_btn.clicked.connect(self._start_reconstruction)
        self.live_preview_cb = QtWidgets.QCheckBox("在线可视化")
        self.live_preview_cb.setChecked(True)
        self.load_btn = QtWidgets.QPushButton("加载 PLY")
        self.load_btn.clicked.connect(self._load_file)
        self.tsdf_btn = QtWidgets.QPushButton("加载 TSDF Mesh")
        self.tsdf_btn.clicked.connect(self._load_default)

        self.clear_btn = QtWidgets.QPushButton("清空")
        self.clear_btn.clicked.connect(self._clear)

        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.tsdf_btn)
        toolbar.addWidget(self.recon_btn)
        toolbar.addWidget(self.live_preview_cb)
        toolbar.addWidget(self.clear_btn)
        toolbar.addStretch(1)

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setWordWrap(True)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=40, elevation=18, azimuth=45)
        self.axis = gl.GLAxisItem()
        self.axis.setSize(10, 10, 10)
        self.view.addItem(self.axis)

        layout.addLayout(toolbar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.view, 1)

    def _clear(self):
        if self.mesh_item:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None
        if self.scatter_item:
            self.view.removeItem(self.scatter_item)
            self.scatter_item = None

    def _load_default(self):
        if not self.output_mesh.exists():
            QtWidgets.QMessageBox.warning(self, "未找到", "TSDF mesh 文件不存在。")
            return
        self._load_mesh(str(self.output_mesh))

    def _load_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开 PLY",
            "",
            "PLY 文件 (*.ply)",
        )
        if not file_path:
            return
        self._load_mesh(file_path)

    def _load_mesh(self, file_path: str):
        try:
            vertices, faces, colors = read_ply(file_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        self._set_mesh(vertices, faces, colors)

    def _set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray],
    ):
        self._clear()
        if vertices.size == 0:
            return
        if colors is not None:
            colors = self._enhance_colors(colors)
        if faces.size == 0:
            self.scatter_item = gl.GLScatterPlotItem(
                pos=vertices,
                color=colors,
                size=2,
            )
            self.view.addItem(self.scatter_item)
        else:
            mesh_data = gl.MeshData(
                vertexes=vertices,
                faces=faces,
                vertexColors=colors,
            )
            self.mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=False,
                drawEdges=False,
                shader="shaded",
            )
            self.view.addItem(self.mesh_item)
        self._auto_fit(vertices)

    def _enhance_colors(self, colors: np.ndarray) -> np.ndarray:
        if colors.shape[1] == 4:
            rgb = colors[:, :3]
        else:
            rgb = colors
        rgb = np.clip(rgb * 1.25, 0.0, 1.0)
        alpha = np.ones((rgb.shape[0], 1), dtype=np.float32)
        return np.concatenate([rgb, alpha], axis=1)

    def _auto_fit(self, vertices: np.ndarray):
        if vertices.size == 0:
            return
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = (mins + maxs) / 2.0
        span = np.linalg.norm(maxs - mins)
        distance = max(span * 1.6, 10.0)
        self.view.opts["center"] = QtGui.QVector3D(*center.tolist())
        self.view.setCameraPosition(distance=distance, elevation=18, azimuth=45)

    def _start_reconstruction(self):
        if self.worker and self.worker.isRunning():
            return
        self.status_label.setText("重建中...")
        self.recon_btn.setEnabled(False)
        self.live_preview_cb.setEnabled(False)
        self.worker = TsdfReconstructionWorker(
            self.data_dir,
            self.output_mesh,
            live_preview=self.live_preview_cb.isChecked(),
        )
        self.worker.completed.connect(self._on_reconstruct_done)
        self.worker.failed.connect(self._on_reconstruct_failed)
        self.worker.progress.connect(self._on_reconstruct_progress)
        self.worker.finished.connect(self._on_reconstruct_finished)
        self.worker.start()

    def _on_reconstruct_done(self, mesh_path: str):
        self.status_label.setText(f"重建完成：{mesh_path}")
        self._load_mesh(mesh_path)

    def _on_reconstruct_failed(self, message: str):
        self.status_label.setText(f"重建失败：{message}")
        QtWidgets.QMessageBox.critical(self, "重建失败", message)

    def _on_reconstruct_finished(self):
        self.recon_btn.setEnabled(True)
        self.live_preview_cb.setEnabled(True)

    def _on_reconstruct_progress(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray],
        current: int,
        total: int,
    ):
        if total > 0:
            self.status_label.setText(f"重建中... {current}/{total}")
        if vertices is not None and faces is not None:
            self._set_mesh(vertices, faces, colors)
