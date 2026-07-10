import torch
import numpy as np
from skimage import measure


def apply_transform(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    points_homogenous = torch.cat(
        (
            points,
            torch.ones((points.shape[0], 1), dtype=torch.float32, device=points.device),
        ),
        axis=1,
    )
    return torch.matmul(pose, points_homogenous.T).T


def get_camera_frustum(
    depth_image: torch.Tensor, K: torch.Tensor, pose: torch.Tensor
) -> torch.Tensor:
    H, W = depth_image.shape
    max_d = depth_image.max().item()
    frustum_points = torch.tensor(
        [
            [0, 0, 1],
            [W, 0, 1],
            [W, H, 1],
            [0, H, 1],
            [0, 0, 0],
        ],
        dtype=torch.float32,
        device=depth_image.device,
    )
    frustum_points = (torch.inverse(K) @ frustum_points.T).T
    frustum_points *= max_d
    frustum_points = apply_transform(pose, frustum_points)
    return frustum_points


def save_mesh(filename: str, vertices, faces, colors) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(vertices.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {}\n".format(faces.shape[0]))
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        for i in range(vertices.shape[0]):
            f.write(
                f"{vertices[i,0]} {vertices[i,1]} {vertices[i,2]} "
                f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n"
            )

        for i in range(faces.shape[0]):
            f.write(f"3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n")


class TSDFVolume:
    def __init__(self, volume_bounds: torch.Tensor, voxel_size: float, device="cpu"):
        assert volume_bounds.shape == (3, 2)
        self.volume_bounds = volume_bounds
        self.voxel_size = voxel_size
        self.device = device
        self.trunc_margin = 5 * voxel_size

        self.volume_dim = torch.ceil(
            (self.volume_bounds[:, 1] - self.volume_bounds[:, 0]) / self.voxel_size
        ).long()
        self.volume_bounds[:, 1] = self.volume_bounds[:, 0] + self.volume_dim * self.voxel_size
        self.origin = self.volume_bounds[:, 0].to(device)

        self.color_volume = torch.zeros(
            (*self.volume_dim, 3), dtype=torch.uint8, device=device
        )
        self.tsdf_volume = torch.ones(*self.volume_dim, dtype=torch.float32, device=device)
        self.weight_volume = torch.zeros_like(self.tsdf_volume, dtype=torch.float32, device=device)

        xv, yv, zv = torch.meshgrid(
            torch.arange(self.volume_dim[0]),
            torch.arange(self.volume_dim[1]),
            torch.arange(self.volume_dim[2]),
            indexing="ij",
        )
        self.voxel_coords = (
            torch.stack((xv, yv, zv), axis=3).reshape(-1, 3).to(device)
        )
        self.world_coords = (self.voxel_coords * self.voxel_size) + self.origin

    def fuse_frame(self, K: torch.Tensor, c2w, rgb, depth, weight=1.0):
        K, c2w, rgb, depth = [x.to(self.device) for x in [K, c2w, rgb, depth]]

        c2w = torch.cat(
            (c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=c2w.device)),
            axis=0,
        )
        w2c = torch.inverse(c2w)[:3]
        voxel_camera = apply_transform(w2c, self.world_coords)
        z = voxel_camera[:, 2].repeat(3, 1).T
        voxel_uv = torch.round((voxel_camera @ K.T) / z).long()
        px, py = voxel_uv[:, :2].T
        pz = voxel_camera[:, 2]

        view_mask = (
            (px >= 0)
            & (px < depth.shape[1])
            & (py >= 0)
            & (py < depth.shape[0])
            & (pz > 0)
        )
        valid_px, valid_py = px[view_mask], py[view_mask]
        valid_vx, valid_vy, valid_vz = (
            self.voxel_coords[view_mask, 0],
            self.voxel_coords[view_mask, 1],
            self.voxel_coords[view_mask, 2],
        )

        depth_val = depth[valid_py, valid_px]
        sdf_value = depth_val - pz[view_mask]
        tsdf_val = torch.clamp(sdf_value / self.trunc_margin, max=1)
        valid_pts = (
            (-self.trunc_margin < sdf_value)
            & (sdf_value < self.trunc_margin)
            & (depth_val > 0)
        )
        tsdf_val = tsdf_val[valid_pts]

        valid_vx = valid_vx[valid_pts]
        valid_vy = valid_vy[valid_pts]
        valid_vz = valid_vz[valid_pts]
        valid_px = valid_px[valid_pts]
        valid_py = valid_py[valid_pts]

        weight_old = self.weight_volume[valid_vx, valid_vy, valid_vz]
        tsdf_old = self.tsdf_volume[valid_vx, valid_vy, valid_vz]
        weight_new = weight_old + weight
        tsdf_new = (tsdf_val * weight + tsdf_old * weight_old) / weight_new
        self.weight_volume[valid_vx, valid_vy, valid_vz] = weight_new
        self.tsdf_volume[valid_vx, valid_vy, valid_vz] = tsdf_new

        weight_old = weight_old.reshape(-1, 1).tile((1, 3))
        color_old = self.color_volume[valid_vx, valid_vy, valid_vz].to(torch.float32)
        weight_new = weight_old + weight
        color_value = rgb[valid_py, valid_px]
        color_new = torch.clamp(
            (color_value * weight + color_old * weight_old) / weight_new, max=255
        ).to(torch.uint8)
        self.color_volume[valid_vx, valid_vy, valid_vz] = color_new

    def extract_mesh(self):
        vertices, faces, normals, _ = measure.marching_cubes(
            self.tsdf_volume.cpu().numpy()
        )
        vertex_coords = vertices.astype(np.int32)
        vertices = vertices * self.voxel_size + self.origin.cpu().numpy()
        color_volume = self.color_volume.cpu().numpy()
        colors = color_volume[
            vertex_coords[:, 0], vertex_coords[:, 1], vertex_coords[:, 2]
        ]
        return vertices, faces, normals, colors

