
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug visualization for CuriosityRewardManager.

- Generates a large cube surface point cloud as object_pointcloud (M,3).
- Uses the 8 vertices of a small cube as keypoint_positions (L=8, 3).
- Calls CuriosityRewardManager.update_contact_state / compute_reward
  to update contact_heatmap and (optionally) compute rewards.
- Visualizes:
    * Matplotlib heatmap of contact_heatmap (L x M)
    * Open3D point cloud colored by per-keypoint / aggregated heat
    * Spheres marking the 8 keypoints

Requirements:
    pip install open3d matplotlib numpy torch
Usage:
    python debug_curiosity_reward_manager_viz.py
Keys in Open3D:
    0 : colorize by aggregate (max over keypoints)
    1..8 : colorize by keypoint k
    R : reset view
    Q : quit
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from curiosity_reward_manager import CuriosityRewardManager
except Exception as e:
    print("[Error] Failed to import CuriosityRewardManager from curiosity_reward_manager.py")
    print("        Please ensure the file is on PYTHONPATH and defines CuriosityRewardManager.")
    print("        Original import error:\n", e)
    sys.exit(1)

import open3d as o3d

# ------------------------------
# Data generation
# ------------------------------
def sample_cube_surface(n_points=4000, size=1.0, center=(0.0, 0.0, 0.0), rng=42):
    rng = np.random.default_rng(rng)
    c = np.asarray(center, dtype=np.float32)
    s = float(size) / 2.0

    pts = np.zeros((n_points, 3), dtype=np.float32)
    faces = rng.integers(0, 6, size=n_points)
    uv = rng.uniform(-s, s, size=(n_points, 2))

    # map faces
    # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
    m = faces == 0
    pts[m] = np.c_[np.full(m.sum(), s), uv[m, 0], uv[m, 1]]
    m = faces == 1
    pts[m] = np.c_[np.full(m.sum(), -s), uv[m, 0], uv[m, 1]]
    m = faces == 2
    pts[m] = np.c_[uv[m, 0], np.full(m.sum(), s), uv[m, 1]]
    m = faces == 3
    pts[m] = np.c_[uv[m, 0], np.full(m.sum(), -s), uv[m, 1]]
    m = faces == 4
    pts[m] = np.c_[uv[m, 0], uv[m, 1], np.full(m.sum(), s)]
    m = faces == 5
    pts[m] = np.c_[uv[m, 0], uv[m, 1], np.full(m.sum(), -s)]
    pts += c[None, :]
    return pts

def small_cube_vertices(size=0.22, center=(0.35, 0.35, 0.35)):
    s = float(size)/2.0
    cx, cy, cz = center
    corners = np.array([[cx - s, cy - s, cz - s],
                        [cx - s, cy - s, cz + s],
                        [cx - s, cy + s, cz - s],
                        [cx - s, cy + s, cz + s],
                        [cx + s, cy - s, cz - s],
                        [cx + s, cy - s, cz + s],
                        [cx + s, cy + s, cz - s],
                        [cx + s, cy + s, cz + s]], dtype=np.float32)
    return corners

# ------------------------------
# Coloring helpers
# ------------------------------
def colorize_scalar(values, cmap_name="viridis"):
    v = np.asarray(values, dtype=np.float32)
    if v.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    vmax = v.max() if np.isfinite(v).all() else 1.0
    if vmax <= 1e-12:
        norm = np.zeros_like(v)
    else:
        norm = np.clip(v / (vmax + 1e-12), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm)[:, :3].astype(np.float32)
    return colors

def make_keypoint_spheres(keypoints, radius=0.02, color=(1.0, 0.2, 0.2)):
    spheres = []
    for p in keypoints:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.compute_vertex_normals()
        s.paint_uniform_color(color)
        s.translate(p.astype(np.float64))
        spheres.append(s)
    return spheres


def make_arrow(p0, p1, radius=0.01, head_ratio=0.2, color=(1.0, 0.2, 0.2)):
    """
    在 Open3D 中生成从 p0 指向 p1 的箭头 TriangleMesh。
    - p0, p1: (3,) 可迭代
    - radius: 箭杆半径
    - head_ratio: 箭头部分占总长度比例（0~1）
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-8:
        raise ValueError("p0 与 p1 太近，无法生成箭头")

    dir_ = v / L
    cyl_h = L * (1.0 - head_ratio)
    cone_h = L * head_ratio

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.5,
        cylinder_height=float(cyl_h),
        cone_height=float(cone_h),
        resolution=20,
        cylinder_split=4,
        cone_split=1,
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color)

    # 将默认 +Z 方向旋转到 dir_
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = np.clip(z.dot(dir_), -1.0, 1.0)
    if np.allclose(dir_, z):
        R = np.eye(3)
    elif np.allclose(dir_, -z):
        # 180° 翻转，绕任意与 z 垂直的轴旋转 π
        axis = np.array([1.0, 0.0, 0.0])
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
    else:
        axis = np.cross(z, dir_)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(p0)  # 底端放到 p0
    return arrow


# ------------------------------
# Main
# ------------------------------
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate data
    pc_np = sample_cube_surface(n_points=512, size=0.4, center=(0, 0, 0))
    kp_np = small_cube_vertices(size=0.2, center=(0.3, 0.3, 0.3))

    pc = torch.from_numpy(pc_np).to(device)
    kp = torch.from_numpy(kp_np).to(device)

    L, M = kp.shape[0], pc.shape[0]

    # Init manager
    mgr = CuriosityRewardManager(num_keypoints=L, num_object_points=M, device=device)

    # Step 1: update state & heatmap using distance-only contact (no force)
    dist_threshold = 0.05

    # Optionally compute both rewards just to exercise code paths
    total_r, info = mgr.compute_reward(
        reward_types=["reaching", "curiosity"],
        object_pointcloud=pc,
        keypoint_positions=kp,
        contact_force=None,
        dist_threshold=dist_threshold,
        k=32,
    )
    print(f"[Info] total_reward={total_r.item():.6f} | keys in info: {list(info.keys())[:6]}...")

    # Grab tensors for visualization
    H = mgr.contact_heatmap.detach().to("cpu").numpy()  # (L, M)
    D = mgr.distance_matrix.detach().to("cpu").numpy()  # (L, M)

    # ---- Open3D visualization ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np.astype(np.float64))

    # default colors = aggregate (max over keypoints)
    # h_max = H.max(axis=0)  # (M,)
    selected_idx = 0
    h_selected = H[selected_idx]  # (M,)
    h_selected[info["reaching_nbrs"][selected_idx].cpu().numpy()] = h_selected[info["reaching_j_peak"][selected_idx].cpu().numpy()]
    pcd.colors = o3d.utility.Vector3dVector(colorize_scalar(h_selected))

    spheres = make_keypoint_spheres(kp_np, radius=0.02, color=(1.0, 0.2, 0.2))
    geoms = [pcd] + spheres
    
    p0 = pc_np[info["reaching_j_peak"][selected_idx].cpu().numpy()]
    p1 = p0 + info["reaching_G_dir"][selected_idx, info["reaching_j_peak"][selected_idx]].cpu().numpy()
    arrow = make_arrow(p0, p1, radius=0.01, head_ratio=0.2, color=(0.2, 1.0, 0.2))
    geoms.append(arrow)
    
    real_dist = D[selected_idx]
    pcd_D = o3d.geometry.PointCloud()
    pcd_D.points = o3d.utility.Vector3dVector(pc_np.astype(np.float64))
    pcd_D.colors = o3d.utility.Vector3dVector(colorize_scalar(real_dist))
    
    curiosity_weight = info["reaching_w_all"][selected_idx].cpu().numpy()
    reweighted_dist = curiosity_weight * real_dist
    pcd_w_all = o3d.geometry.PointCloud()
    pcd_w_all.points = o3d.utility.Vector3dVector(pc_np.astype(np.float64))
    pcd_w_all.colors = o3d.utility.Vector3dVector(colorize_scalar(reweighted_dist))
    
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    w1 = o3d.visualization.O3DVisualizer("Win 1", 800, 600)
    for i, geom in enumerate(geoms):
        w1.add_geometry(f"geom_{i}", geom)
    app.add_window(w1)

    w2 = o3d.visualization.O3DVisualizer("Win 2", 800, 600)
    w2.add_geometry("pcd_D", pcd_D)
    app.add_window(w2)
    
    w3 = o3d.visualization.O3DVisualizer("Win 3", 800, 600)
    w3.add_geometry("pcd_w_all", pcd_w_all)
    app.add_window(w3)
    
    app.run()

if __name__ == "__main__":
    main()
