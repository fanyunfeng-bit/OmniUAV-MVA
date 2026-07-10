from typing import List, Optional, Tuple

import numpy as np


def read_ply(file_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Read PLY file in ASCII format.

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face indices
        colors: (N, 4) array of vertex colors (RGBA, 0-1 range) or None
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        if f.readline().strip() != "ply":
            raise ValueError("PLY 文件格式错误：缺少 ply 头")
        format_line = f.readline().strip()
        if not format_line.startswith("format ascii"):
            raise ValueError("仅支持 ASCII PLY")

        vertex_count = 0
        face_count = 0
        vertex_props: List[str] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY 头缺少 end_header")
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element face"):
                face_count = int(line.split()[-1])
                in_vertex = False
                continue
            if line.startswith("property") and in_vertex:
                vertex_props.append(line.split()[-1])
                continue
            if line == "end_header":
                break

        if vertex_count <= 0:
            raise ValueError("PLY 不包含顶点数据")
        xyz_idx = [vertex_props.index(axis) for axis in ("x", "y", "z")]
        color_idx = []
        for name in ("red", "green", "blue"):
            if name in vertex_props:
                color_idx.append(vertex_props.index(name))
        vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        colors = None
        if len(color_idx) == 3:
            colors = np.zeros((vertex_count, 4), dtype=np.float32)

        for i in range(vertex_count):
            parts = f.readline().strip().split()
            if len(parts) < len(vertex_props):
                raise ValueError("PLY 顶点数据长度不匹配")
            values = [float(p) for p in parts]
            vertices[i] = [values[idx] for idx in xyz_idx]
            if colors is not None:
                rgb = [values[idx] for idx in color_idx]
                colors[i] = [c / 255.0 for c in rgb] + [0.95]

        faces = []
        for _ in range(face_count):
            parts = f.readline().strip().split()
            if not parts:
                continue
            count = int(parts[0])
            indices = [int(p) for p in parts[1 : 1 + count]]
            if count == 3:
                faces.append(indices)
            elif count > 3:
                base = indices[0]
                for j in range(1, count - 1):
                    faces.append([base, indices[j], indices[j + 1]])

        faces_array = (
            np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
        )
        return vertices, faces_array, colors
