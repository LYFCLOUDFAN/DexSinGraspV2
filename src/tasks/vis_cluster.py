import torch
import open3d as o3d
import numpy as np
from typing import Tuple, Optional


def _create_box_pointclouds(pcl_num=1024, box_width=1.0, box_height=1.0, box_depth=1.0) -> torch.Tensor:
    """Create pointclouds for box objects by sampling points on the 6 faces."""
    # Sample random points uniformly across all objects
    points = torch.rand(pcl_num, 3)
    
    # Scale to box dimensions and center at origin
    points[..., 0] = points[..., 0] * box_width - box_width / 2    # x
    points[..., 1] = points[..., 1] * box_depth - box_depth / 2    # y
    points[..., 2] = points[..., 2] * box_height - box_height / 2  # z

    n_per_face = pcl_num // 6

    # z boundaries
    points[:n_per_face, 2] = box_height / 2
    points[n_per_face:2*n_per_face, 2] = -box_height / 2

    # y boundaries
    points[2*n_per_face:3*n_per_face, 1] = box_depth / 2
    points[3*n_per_face:4*n_per_face, 1] = -box_depth / 2

    # x boundaries
    points[4*n_per_face:5*n_per_face, 0] = box_width / 2
    points[5*n_per_face:, 0] = -box_width / 2

    return points


@torch.no_grad()
def _perform_clustering(pointcloud: torch.Tensor, cluster_k=10, max_clustering_iters=10) -> torch.Tensor:
    M, _ = pointcloud.shape
    points = pointcloud.to(torch.float32)

    indices = torch.randperm(M)[:cluster_k]
    centers = points[indices].clone()  # (cluster_k, 3)

    for _ in range(max_clustering_iters):
        distances = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # (M, cluster_k)
        labels = torch.argmin(distances, dim=1)  # (M,)
        new_centers = torch.zeros_like(centers)
        for i in range(cluster_k):
            mask = (labels == i)
            if mask.any():
                new_centers[i] = points[mask].mean(dim=0)
            else:
                new_centers[i] = centers[i]
        centers = new_centers

    return labels, centers


def _labels_to_colors(labels: np.ndarray, k: int) -> np.ndarray:
    """给每个簇一个固定颜色（用 RNG 固定种子避免每次颜色变）。"""
    rng = np.random.default_rng(0)
    palette = rng.uniform(0.1, 0.95, size=(k, 3))
    palette[palette.argmax(axis=1) == 0] = [0.9, 0.1, 0.1]  # 轻微打散，非必须
    return palette[labels]


def visualize_clusters_open3d(points_t: torch.Tensor, labels_t: torch.Tensor, centers_t: torch.Tensor = None,
                              sphere_radius: float = 0.01, window_name: str = "K-means clusters"):
    """用 Open3D 可视化：点按簇上色，中心画小球，附带坐标轴与包围盒。"""
    points = points_t.detach().cpu().numpy()
    labels = labels_t.detach().cpu().numpy().astype(np.int64)
    k = int(labels.max()) + 1 if labels.size > 0 else 0

    # 点云
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if k > 0:
        colors = _labels_to_colors(labels, k)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd]

    # 坐标轴 & 包围盒
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(extent * 0.4, 1e-3))
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (0, 0, 0)
    geoms.extend([axis, aabb])

    # 画中心
    if centers_t is not None:
        centers = centers_t.detach().cpu().numpy()
        for c in centers:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sph.paint_uniform_color([0, 0, 0])  # 黑色中心
            sph.translate(c.astype(float))
            geoms.append(sph)

    # o3d.visualization.draw_geometries(
    #     geoms, window_name=window_name, width=1024, height=768, mesh_show_back_face=True
    # )

    # 打印每簇大小辅助检查
    uniq, cnt = np.unique(labels, return_counts=True)
    print("Cluster sizes:", dict(zip(uniq.tolist(), cnt.tolist())))
    return geoms
    
    
@torch.no_grad()
def fps(points: torch.Tensor, k: int, start_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Farthest Point Sampling for a single point cloud.
    Args:
        points: (N, D) float tensor
        k:      #centers
        start_idx: 首个中心索引（可选），None 则随机
    Returns:
        idx:     (k,) long，被选点的索引（按采样顺序）
        centers: (k, D) float，被选中心坐标
    """
    assert points.dim() == 2, "points should be (N, D)"
    N, D = points.shape
    k = min(k, N)
    pts = points.to(dtype=torch.float32)
    device = pts.device

    dists = torch.full((N,), float('inf'), device=device)
    idx = torch.empty((k,), dtype=torch.long, device=device)

    farthest = torch.randint(0, N, (1,), device=device).item() if start_idx is None else int(start_idx) % N
    for i in range(k):
        idx[i] = farthest
        center = pts[farthest].view(1, D)
        dist2 = ((pts - center) ** 2).sum(dim=1)
        dists = torch.minimum(dists, dist2)
        farthest = torch.argmax(dists).item()

    centers = pts[idx]
    return idx, centers


@torch.no_grad()
def _assign_labels_by_nn(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    用平方欧氏距离把每个点分配到最近中心。
    points:  (M, D)
    centers: (k, D)
    return:  labels (M,)
    """
    x2 = (points**2).sum(dim=1, keepdim=True)         # (M,1)
    c2 = (centers**2).sum(dim=1).unsqueeze(0)         # (1,k)
    dist2 = x2 - 2 * (points @ centers.T) + c2        # (M,k)
    labels = dist2.argmin(dim=1)
    return labels


@torch.no_grad()
def cluster_with_fps(pointcloud: torch.Tensor, k: int, refine_iters: int = 0):
    """
    先用 FPS 选 k 个中心，再按最近中心聚类。
    可选：refine_iters > 0 时进行少量 K-means 式的均值微调（把中心仍约束在点云上可改为“投影回最近点”）。

    Returns:
        labels:  (M,) long
        centers: (k, D) float
    """
    M, _ = pointcloud.shape
    k = min(k, M)
    pts = pointcloud.to(torch.float32)

    _, centers = fps(pts, k)
    labels = _assign_labels_by_nn(pts, centers)

    # 可选：小步微调中心（默认 0 轮）
    for _ in range(refine_iters):
        counts = torch.bincount(labels, minlength=k).clamp_min(1).to(pts.dtype)  # (k,)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, labels, pts)
        new_centers = new_centers / counts.unsqueeze(1)
        centers = new_centers
        labels = _assign_labels_by_nn(pts, centers)

    return labels, centers


if __name__ == "__main__":
    pcl_num = 1024
    box_width = 0.08
    box_height = 0.16
    box_depth = 0.04
    
    # knn
    pcl = _create_box_pointclouds(pcl_num=pcl_num, box_width=box_width, box_height=box_height, box_depth=box_depth)
    print(pcl.shape)
    labels, centers = _perform_clustering(pcl, cluster_k=64, max_clustering_iters=10)
    print(labels)
    
    geoms_kmeans = visualize_clusters_open3d(pcl, labels, centers, sphere_radius=0.004, window_name="Box point cloud - K-means")
    
    # fps + nn
    pcl = _create_box_pointclouds(pcl_num=pcl_num, box_width=box_width, box_height=box_height, box_depth=box_depth)
    print(pcl.shape)
    labels, centers = cluster_with_fps(pcl, k=64, refine_iters=0)
    print(labels)
    geoms_fps = visualize_clusters_open3d(pcl, labels, centers, sphere_radius=0.004, window_name="Box point cloud - FPS + NN")
    
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    w1 = o3d.visualization.O3DVisualizer("Win 1", 800, 600)
    for i, geom in enumerate(geoms_kmeans):
        w1.add_geometry(f"geom_{i}", geom)
    app.add_window(w1)

    w2 = o3d.visualization.O3DVisualizer("Win 2", 800, 600)
    for i, geom in enumerate(geoms_fps):
        w2.add_geometry(f"geom_{i}", geom)
    app.add_window(w2)
    
    app.run()