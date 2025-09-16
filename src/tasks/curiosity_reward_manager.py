import torch
from typing import Optional, Dict, Tuple


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


class CuriosityRewardManager:

    def __init__(
        self,
        num_keypoints: Optional[int] = None,
        num_object_points: Optional[int] = None,
        canonical_pointcloud: Optional[torch.Tensor] = None,
        *,
        device: Optional[torch.device] = None,
        k: int = 8,  # number of KNN for generating expected position
        contact_bonus: float = 0.0,
        per_contact: bool = False,
        eps: float = 1e-8,
        potential_sigma: float = 0.05,  # empirical value
        cluster_k: int = 64,  # number of clusters for object point cloud
        max_clustering_iters: int = 10,  # max number of iterations for K-Means
        multiplier_min: float = 0.0,
        threshold_ratio: float = 0.2,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canonical_pointcloud = canonical_pointcloud
        # --- keep: point-level heatmap (for gradient computation) ---
        self.contact_heatmap: Optional[torch.Tensor] = None  # long, (L, M)

        # runtime cache
        self.prev_potential: Optional[torch.Tensor] = None  # (N,) previous potential
        self.potential_per_kp_max: Optional[torch.Tensor] = None  # (N, L) previous potential per keypoint

        self.L = num_keypoints
        self.M = num_object_points

        self.default_k = k
        self.default_contact_bonus = contact_bonus
        self.default_per_contact = per_contact
        self.eps = eps

        # potential function parameter
        self.potential_sigma = potential_sigma

        # cache for gradient computation
        self.smoothed_heatmap: Optional[torch.Tensor] = None  # (N, L, M)
        self.gradient_direction: Optional[torch.Tensor] = None  # (N, L, M, 3)
        self.gradient_magnitude: Optional[torch.Tensor] = None  # (N, L, M)
        
        self.multiplier_min = multiplier_min
        self.threshold_ratio = threshold_ratio

        # cache KNN indices
        self._knn_cache: Optional[torch.Tensor] = None  # (N, M, k)

        # --- new: cluster related (only for contact reward) ---
        self.cluster_k = cluster_k
        self.max_clustering_iters = max_clustering_iters
        self._point_to_cluster: Optional[torch.Tensor] = None  # (M,) cluster id for each point in object point cloud
        # for each fingertip, maintain a separate counter (L, cluster_k)
        self._per_fingertip_cluster_counts: Optional[torch.Tensor] = None
        
        self._point_to_cluster = self._perform_clustering_w_fps(self.canonical_pointcloud) 
        self._per_fingertip_cluster_counts = torch.zeros((self.L, self.cluster_k), dtype=torch.long, device=self.device)


    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.LongTensor] = None, reset_counters: bool = False):
        if reset_counters:
            if self.contact_heatmap is not None:
                self.contact_heatmap.zero_()
            if self._per_fingertip_cluster_counts is not None:
                self._per_fingertip_cluster_counts.zero_()
        if self.prev_potential is not None:
            if env_ids is None:
                self.prev_potential = None
            else:
                self.prev_potential[env_ids] = 0.0

    @torch.no_grad()
    def reset_counters(self):
        """zero out all counters"""
        if self.contact_heatmap is not None:
            self.contact_heatmap.zero_()
        if self._per_fingertip_cluster_counts is not None:
            self._per_fingertip_cluster_counts.zero_()

    @torch.no_grad()
    def update_contact_heatmap(self, contact_indices: torch.Tensor, contact_mask: torch.Tensor, num_object_points: Optional[int] = None):
        """
        update global contact heatmap (across environments)
        - only update point-level heatmap `self.contact_heatmap`, for gradient computation
        """
        assert contact_indices.dtype in (torch.int32, torch.int64)
        N, L = contact_indices.shape

        if self.L is None:
            self.L = L
        else:
            assert self.L == L, f"L mismatch: {self.L} != {L}"

        if num_object_points is not None:
            M = int(num_object_points)
            if self.M is None:
                self.M = M
            else:
                assert self.M == M, f"M mismatch: {self.M} != {M}"
        else:
            assert self.M is not None, "num_object_points is required at first update"

        if self.contact_heatmap is None:
            self.contact_heatmap = torch.zeros((self.L, self.M), dtype=torch.long, device=self.device)

        has = contact_mask
        if not has.any():
            return

        env_idx, kp_idx = torch.nonzero(has, as_tuple=True)
        j_idx = contact_indices[env_idx, kp_idx].clamp(min=0, max=self.M - 1)

        lin = kp_idx.to(torch.long) * self.M + j_idx.to(torch.long)
        counts = torch.bincount(lin, minlength=self.L * self.M)
        counts = counts.view(self.L, self.M).to(self.contact_heatmap.dtype).to(self.contact_heatmap.device)

        self.contact_heatmap.add_(counts)

    @torch.no_grad()
    def _perform_clustering(self, pointcloud: torch.Tensor):
        M, _ = pointcloud.shape
        points = pointcloud.to(torch.float32)

        indices = torch.randperm(M, device=self.device)[:self.cluster_k]
        centers = points[indices].clone()  # (cluster_k, 3)

        for _ in range(self.max_clustering_iters):
            distances = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # (M, cluster_k)
            labels = torch.argmin(distances, dim=1)  # (M,)
            new_centers = torch.zeros_like(centers)
            for i in range(self.cluster_k):
                mask = (labels == i)
                if mask.any():
                    new_centers[i] = points[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]
            centers = new_centers

        return labels
    
    def _perform_clustering_w_fps(self, pointcloud: torch.Tensor):
        """
        先用 FPS 选 k 个中心，再按最近中心聚类。
        可选：refine_iters > 0 时进行少量 K-means 式的均值微调（把中心仍约束在点云上可改为“投影回最近点”）。

        Returns:
            labels:  (M,) long
            centers: (k, D) float
        """
        M, _ = pointcloud.shape
        k = min(self.cluster_k, M)
        pts = pointcloud.to(torch.float32)

        _, centers = fps(pts, k)
        labels = _assign_labels_by_nn(pts, centers)

        # 可选：小步微调中心（默认 0 轮）
        for _ in range(self.max_clustering_iters):
            counts = torch.bincount(labels, minlength=k).clamp_min(1).to(pts.dtype)  # (k,)
            new_centers = torch.zeros_like(centers)
            new_centers.index_add_(0, labels, pts)
            new_centers = new_centers / counts.unsqueeze(1)
            centers = new_centers
            labels = _assign_labels_by_nn(pts, centers)

        return labels

    @torch.no_grad()
    def compute_reward(
        self,
        tau: float,
        object_pointclouds: torch.Tensor,   # (N, M, 3)
        keypoint_positions: torch.Tensor,   # (N, L, 3) — fingertip positions
        contact_indices: Optional[torch.Tensor] = None,  # (N, L) long, [0..M-1]
        contact_mask: Optional[torch.Tensor] = None,     # (N, L) bool
        *,
        k: Optional[int] = None,
        contact_bonus: Optional[float] = None,
        per_contact: Optional[bool] = None,
        object_normals: Optional[torch.Tensor] = None,   # (N, M, 3) optional, 本方案未使用
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        * Legacy version of reward computation
            1. 执行聚类 (如果需要) 并更新聚类计数器。
            2. 更新点级热图 (用于梯度计算)。
            3. 计算全局热图的平滑版本和梯度方向 (基于点级热图)。
            4. 对于每个指尖，找到其**接触次数最多**的物体点 P_c (如果无接触，则用最近点)。
            5. 在 P_c 处，用 -∇H 与 KNN 邻居向量的余弦相似度加权，生成一个“期望探索位置” P_target。
            6. 计算每个指尖到其 P_target 的距离。
            7. **奖励 = exp(-当前距离 / sigma) - exp(-上一步距离 / sigma)** (势能差分)。
            8. 叠加基于聚类的新颖度接触奖励。
        """
        device = object_pointclouds.device
        N, M, _ = object_pointclouds.shape
        _, L, _ = keypoint_positions.shape

        k = int(k if k is not None else self.default_k)
        contact_bonus = float(contact_bonus if contact_bonus is not None else self.default_contact_bonus)
        per_contact = bool(self.default_per_contact if per_contact is None else per_contact)

        # --- Step 1: 执行聚类 (如果需要) 并更新聚类计数器 ---
        if self._point_to_cluster is None or self._point_to_cluster.shape[0] != M:
            sample_pointcloud = object_pointclouds[0]  # (M, 3)
            self._point_to_cluster = self._perform_clustering(sample_pointcloud)  # (M,)
            # 初始化计数器
            self._per_fingertip_cluster_counts = torch.zeros(
                (self.L, self.cluster_k), dtype=torch.long, device=self.device
            )

        # 更新聚类计数器 (仅用于接触奖励)
        cm = contact_mask.to(torch.bool) if contact_mask is not None else torch.zeros((N, L), dtype=torch.bool, device=device)
        if (contact_indices is not None) and cm.any():
            has = cm
            ei, ki = torch.nonzero(has, as_tuple=True)  # (num_contacts,)
            pj = contact_indices[ei, ki].clamp(0, M - 1)  # (num_contacts,)
            # 获取接触点所属的簇ID
            cluster_ids = self._point_to_cluster[pj]  # (num_contacts,)
            # 更新计数器: lin_idx = ki * cluster_k + cluster_id
            lin_idx = ki * self.cluster_k + cluster_ids
            counts = torch.bincount(lin_idx, minlength=self.L * self.cluster_k)
            counts = counts.view(self.L, self.cluster_k).to(self._per_fingertip_cluster_counts.dtype)
            self._per_fingertip_cluster_counts.add_(counts)

        # --- Step 2: 更新点级热图 (用于梯度计算) ---
        self.update_contact_heatmap(contact_indices.to(device), contact_mask.to(device), num_object_points=M)

        # --- Step 3: 基于点级热图计算梯度 (保持不变) ---
        contact_heatmap = self.contact_heatmap.to(torch.float32)  # (L, M)
        H = contact_heatmap  # 直接使用点级计数作为H

        # 获取 KNN 索引
        k_eff = min(k, max(1, M))
        if self._knn_cache is None or self._knn_cache.shape != (N, M, k_eff):
            self._knn_cache = self._knn_indices(object_pointclouds, k=k_eff)  # (N, M, k)
        knn_idx = self._knn_cache

        # 计算梯度
        H_expanded = H.unsqueeze(0).expand(N, -1, -1)  # (N, L, M)
        smoothed_heatmap, gradient_direction, gradient_magnitude = self._smooth_and_gradient(
            object_pointclouds, H_expanded, knn_idx
        )
        self.smoothed_heatmap = smoothed_heatmap
        self.gradient_direction = gradient_direction  # (N, L, M, 3)
        self.gradient_magnitude = gradient_magnitude

        # --- Step 4: 为每个指尖生成“期望探索位置” P_target ---
        # 4.1 找到每个指尖**接触次数最多**的物体点索引
        # contact_heatmap: (L, M) - 每个指尖对每个点的累计接触次数
        # 对于从未接触过的指尖，其热图全为0，argmax会返回0号点，这可能不合理。
        # 因此，我们先检查是否有接触，如果没有，则回退到使用几何最近点。

        # 计算几何最近点索引 (备用方案)
        D_geom = torch.cdist(keypoint_positions, object_pointclouds, p=2)  # (N, L, M)
        closest_point_idx_geom = D_geom.argmin(dim=-1)  # (N, L)

        # 初始化 P_c 的索引
        closest_point_idx = torch.zeros((N, L), dtype=torch.long, device=device)

        for l in range(L):
            # 获取第 l 个指尖的接触热图 (M,)
            heatmap_l = contact_heatmap[l, :]
            # 检查该指尖是否有过接触
            if heatmap_l.sum() > 0:
                # 有接触：选择接触次数最多的点
                most_contacted_idx = heatmap_l.argmax()  # 标量
                closest_point_idx[:, l] = most_contacted_idx
            else:
                # 无接触：回退到几何最近点
                closest_point_idx[:, l] = closest_point_idx_geom[:, l]

        # 4.2 为每个指尖的 P_c 点，获取其 KNN 邻居 (保持不变)
        closest_knn_idx = torch.gather(
            knn_idx.unsqueeze(1).expand(-1, L, -1, -1),  # (N, L, M, k)
            dim=2,
            index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, k_eff)  # (N, L, 1, k)
        ).squeeze(2)  # (N, L, k)

        # 4.3 获取 P_c 和其邻居 P_i 的坐标 (保持不变)
        P_c = torch.gather(
            object_pointclouds.unsqueeze(1).expand(-1, L, -1, -1),  # (N, L, M, 3)
            dim=2,
            index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)  # (N, L, 1, 3)
        ).squeeze(2)  # (N, L, 3)

        P_neighbors = torch.gather(
            object_pointclouds.unsqueeze(1).expand(-1, L, -1, -1),  # (N, L, M, 3)
            dim=2,
            index=closest_knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (N, L, k, 3)
        )  # (N, L, k, 3)

        # 4.4 计算从 P_c 指向每个邻居的向量 V_i (保持不变)
        V_i = P_neighbors - P_c.unsqueeze(2)  # (N, L, k, 3)
        V_i_norm = V_i / (V_i.norm(dim=-1, keepdim=True).clamp_min(self.eps))  # (N, L, k, 3)

        # 4.5 获取在 P_c 处的梯度方向 -∇H (保持不变)
        neg_grad_at_Pc = torch.gather(
            gradient_direction,  # (N, L, M, 3)
            dim=2,
            index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)  # (N, L, 1, 3)
        ).squeeze(2)  # (N, L, 3)

        # 4.6 计算余弦相似度作为权重 w_i (保持不变)
        w_i = (neg_grad_at_Pc.unsqueeze(2) * V_i_norm).sum(dim=-1)  # (N, L, k)
        V_exp = (w_i.unsqueeze(-1) * V_i).sum(dim=2)  # (N, L, 3)
        P_target = P_c + V_exp  # (N, L, 3)

        # --- Step 5: 计算每个指尖到其 P_target 的距离 (保持不变) ---
        dist_to_target = torch.norm(keypoint_positions - P_target, dim=-1)  # (N, L)
        avg_dist_to_target = dist_to_target.mean(dim=1)  # (N,)

        # --- Step 6: 使用势能函数计算奖励 (保持不变) ---
        current_potential = torch.exp(-avg_dist_to_target / self.potential_sigma)  # (N,)

        if (self.prev_potential is None) or (self.prev_potential.shape != (N,)):
            self.prev_potential = current_potential.detach().clone()
        r_progress = current_potential - self.prev_potential
        self.prev_potential = current_potential.detach().clone()

        # --- Step 7: 计算基于聚类的新颖度接触奖励 (保持不变) ---
        contact_novelty_reward = torch.zeros((N, L), dtype=avg_dist_to_target.dtype, device=device)
        if (contact_indices is not None) and cm.any():
            has = cm
            ei, ki = torch.nonzero(has, as_tuple=True)  # (num_contacts,)
            pj = contact_indices[ei, ki].clamp(0, M - 1)  # (num_contacts,)
            cluster_ids = self._point_to_cluster[pj]  # (num_contacts,)
            counts = self._per_fingertip_cluster_counts[ki, cluster_ids]  # (num_contacts,)
            contact_novelty_reward[ei, ki] = 1.0 / torch.sqrt(1.0 + counts.float())

        bonus_term = contact_novelty_reward.mean(dim=1)  # (N,)
        reward = r_progress + contact_bonus * bonus_term

        self.last_P_target = P_target.detach().clone()  # (N, L, 3)

        info = {
            "avg_dist_to_target": avg_dist_to_target,  # (N,)
            "progress": r_progress,                    # (N,)
            "contact_count": cm.sum(dim=1),            # (N,) - 总接触次数
            "cluster_novelty_reward": bonus_term,      # (N,) - 基于聚类的新颖度奖励
            "current_potential": current_potential,    # (N,)
        }
        return reward, info



    @torch.no_grad()
    def _knn_indices(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """assume all env use the same point cloud"""
        N, M, _ = X.shape
        d_first = torch.cdist(X[0:1], X[0:1], p=2).squeeze(0)  # Shape: (M, M)
        # Get KNN indices for the first point cloud
        knn_idx_first = d_first.topk(k=k, dim=-1, largest=False).indices  # Shape: (M, k)
        # Expand the result to all N environments
        knn_idx = knn_idx_first.unsqueeze(0).expand(N, -1, -1)  # Shape: (N, M, k)
        return knn_idx

    @torch.no_grad()
    def _smooth_and_gradient(
        self,
        pc: torch.Tensor,                # (N, M, 3) or (M, 3)
        H: torch.Tensor,                 # (N, L, M) or (L, M)
        knn_idx: torch.Tensor,           # (N, M, k) or (M, k)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        has_batch = (pc.dim() == 3)
        if not has_batch:
            # to with bath
            pc = pc.unsqueeze(0)          # (1, M, 3)
            H = H.unsqueeze(0)            # (1, L, M)
            knn_idx = knn_idx.unsqueeze(0)  # (1, M, k)

        N, M, _ = pc.shape
        _, L, M2 = H.shape
        assert M == M2
        k = knn_idx.size(-1)
        device = pc.device

        Hf = H.to(torch.float32)  # (N, L, M)

        H_nb = torch.gather(
            Hf.unsqueeze(-1).expand(-1, -1, -1, k),  # (N, L, M, k)
            dim=2,
            index=knn_idx.unsqueeze(1).expand(-1, L, -1, -1)  # (N, L, M, k)
        )  # (N, L, M, k)

        smoothed_heatmap = 0.5 * Hf + 0.5 * H_nb.mean(dim=-1)  # (N, L, M)

        knn_idx_reshaped = knn_idx.view(N, M*k)  # (N, M*k)
        X_nb_flat = torch.gather(
            pc,  # (N, M, 3)
            dim=1,
            index=knn_idx_reshaped.unsqueeze(-1).expand(N, M*k, 3)  # (N, M*k, 3)
        )  # (N, M*k, 3)
        X_nb_base = X_nb_flat.view(N, M, k, 3)  # (N, M, k, 3)
        
        X_nb = X_nb_base.unsqueeze(1).expand(-1, L, -1, -1, -1)  # (N, L, M, k, 3)
        
        X = pc.unsqueeze(1).expand(-1, L, -1, -1)  # (N, L, M, 3)

        dX = X_nb - X.unsqueeze(-2)                 # (N, L, M, k, 3)
        dH = H_nb - smoothed_heatmap.unsqueeze(-1)  # (N, L, M, k)

        denom = (dX.pow(2).sum(dim=-1) + self.eps)  # (N, L, M, k)
        G = (dH.unsqueeze(-1) * dX / denom.unsqueeze(-1)).sum(dim=-2)  # (N, L, M, 3)

        G_mag = G.norm(dim=-1)        # (N, L, M)
        G_dir = torch.zeros_like(G)   # (N, L, M, 3)
        nonzero = (G_mag > 1e-8)
        if nonzero.any():
            G_dir[nonzero] = (-G[nonzero] / G_mag[nonzero].unsqueeze(-1)) # -∇H

        if not has_batch:
            return smoothed_heatmap.squeeze(0), G_dir.squeeze(0), G_mag.squeeze(0)

        return smoothed_heatmap, G_dir, G_mag

    @torch.no_grad()
    def compute_reward_from_canonical(
        self,
        *,
        object_positions: torch.Tensor,        # (N, 3)
        object_orientations: torch.Tensor,     # (N, 4) [x,y,z,w]
        keypoint_positions_world: torch.Tensor,# (N, L, 3)
        contact_indices: Optional[torch.Tensor] = None,  # (N, L)
        contact_mask: Optional[torch.Tensor] = None,     # (N, L)
        k: Optional[int] = None,
        contact_bonus: Optional[float] = None,
        per_contact: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        compute reward based on canonical point cloud
        - use canonical point cloud to compute gradient
        - compute P_target in object local coordinates
        - compute potential-based reaching reward
        - compute novelty-based contact reward
        - return reward and info
        """
        device = keypoint_positions_world.device
        N, L, _ = keypoint_positions_world.shape
        M = self.canonical_pointcloud.shape[0]

        k = int(k if k is not None else self.default_k)
        contact_bonus = float(contact_bonus if contact_bonus is not None else self.default_contact_bonus)
        per_contact = bool(self.default_per_contact if per_contact is None else per_contact)

        # Step 1: cluster (only for contact reward)
        cm = contact_mask.to(torch.bool) if contact_mask is not None else torch.zeros((N, L), dtype=torch.bool, device=device)
        if (contact_indices is not None) and cm.any():
            has = cm
            ei, ki = torch.nonzero(has, as_tuple=True)
            pj = contact_indices[ei, ki].clamp(0, M - 1)
            cluster_ids = self._point_to_cluster[pj]
            lin_idx = ki * self.cluster_k + cluster_ids
            counts = torch.bincount(lin_idx, minlength=L * self.cluster_k)
            counts = counts.view(L, self.cluster_k).to(self._per_fingertip_cluster_counts.dtype)
            self._per_fingertip_cluster_counts.add_(counts)

        # Step 2: update global point-level heatmap, for gradient computation
        if contact_indices is None:
            contact_indices = torch.zeros((N, L), dtype=torch.long, device=device)
        if contact_mask is None:
            contact_mask = torch.zeros((N, L), dtype=torch.bool, device=device)
        self.update_contact_heatmap(contact_indices.to(device), contact_mask.to(device), num_object_points=M)

        # Step 3: compute gradient based on canonical point cloud (only N=1)
        contact_heatmap = self.contact_heatmap.to(torch.float32)  # (L, M)
        H = contact_heatmap
        k_eff = min(k, max(1, M))
        # only use the first (canonical) point cloud to compute KNN, and reuse for all envs
        X = self.canonical_pointcloud.unsqueeze(0)  # (1, M, 3)
        if self._knn_cache is None or self._knn_cache.shape != (1, M, k_eff):
            self._knn_cache = self._knn_indices(X, k=k_eff)  # (1, M, k)
        knn_idx = self._knn_cache  # (1, M, k)

        H_expanded = H.unsqueeze(0)  # (1, L, M)
        smoothed_heatmap, gradient_direction, gradient_magnitude = self._smooth_and_gradient(
            X, H_expanded, knn_idx
        )

        self.smoothed_heatmap = smoothed_heatmap
        self.gradient_direction = gradient_direction  # (1, L, M, 3)
        self.gradient_magnitude = gradient_magnitude

        # Step 4: compute P_target in object local coordinates
        # convert fingertip positions from world to local
        # p_local = R^T (p_world - t)
        from .torch_utils import quat_conjugate, quat_apply
        q = object_orientations
        t = object_positions
        q_conj = quat_conjugate(q)
        kp_local = quat_apply(q_conj.unsqueeze(1).expand(-1, L, -1), keypoint_positions_world - t.unsqueeze(1))  # (N,L,3)

        # Ruoyi: previous strategy
        # d_local = torch.cdist(kp_local, self.canonical_pointcloud.unsqueeze(0).expand(N, -1, -1), p=2)  # (N,L,M)
        # closest_point_idx = d_local.argmin(dim=-1)  # (N,L)
        
        # closest_point_idx = torch.where(
        #     self.contact_heatmap.sum(dim=-1) > 0,
        #     self.contact_heatmap.argmax(dim=-1),
        #     closest_point_idx
        # )
        
        # closest_knn_idx = torch.gather(
        #     knn_idx.unsqueeze(1).expand(N, L, -1, -1),  # (N,L,M,k)
        #     dim=2,
        #     index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, k_eff)
        # ).squeeze(2)  # (N,L,k)

        # P_c = self.canonical_pointcloud.unsqueeze(0).unsqueeze(1).expand(N, L, -1, -1)
        # P_c = torch.gather(P_c, dim=2, index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)  # (N,L,3)

        # P_neighbors = self.canonical_pointcloud.unsqueeze(0).unsqueeze(1).expand(N, L, -1, -1)
        # P_neighbors = torch.gather(P_neighbors, dim=2, index=closest_knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # (N,L,k,3)

        # V_i = P_neighbors - P_c.unsqueeze(2)  # (N,L,k,3)
        # V_i_norm = V_i / (V_i.norm(dim=-1, keepdim=True).clamp_min(self.eps))

        # neg_grad_at_Pc = torch.gather(
        #     gradient_direction.expand(N, -1, -1, -1),  # (N,L,M,3)
        #     dim=2,
        #     index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        # ).squeeze(2)  # (N,L,3)

        # w_i = (neg_grad_at_Pc.unsqueeze(2) * V_i_norm).sum(dim=-1)  # (N,L,k)
        # V_exp = (w_i.unsqueeze(-1) * V_i).sum(dim=2)  # (N,L,3)
        # P_target_local = P_c + V_exp  # (N,L,3)
        

        anchor_idx = torch.argmax(self.contact_heatmap, dim=-1)  # (L,) 
        anchor_cluster = self._point_to_cluster[anchor_idx]       # (L,)

        # 构造锚点所在簇的mask: (N, L, M)
        point_clusters = self._point_to_cluster  # (M,)
        inside_cluster_mask = (point_clusters.unsqueeze(0).expand(self.L, -1) == anchor_cluster.unsqueeze(1))  # (L,M)
        inside_cluster_mask = inside_cluster_mask.unsqueeze(0).expand(N, -1, -1)  # (N,L,M)

        # 取锚点位置 X_anchor: (L,3) → (N,L,1,3)
        X_anchor = self.canonical_pointcloud[anchor_idx]  # (L,3)
        X_anchor = X_anchor.unsqueeze(0).unsqueeze(2).expand(N, L, 1, 3)  # (N,L,1,3)

        # 取锚点方向 u_anchor_dir: gradient_direction[0, L, M, 3] → (L,3) → (N,L,1,3)
        gd = gradient_direction.squeeze(0)  # (L,M,3)
        idx_gd = anchor_idx.view(self.L, 1, 1).expand(self.L, 1, 3)
        u_anchor_dir = torch.gather(gd, dim=1, index=idx_gd).squeeze(1)  # (L,3)
        # 归一化，防止零向量
        u_norm = torch.norm(u_anchor_dir, dim=-1, keepdim=True).clamp_min(self.eps)
        u_anchor_dir = u_anchor_dir / u_norm
        u_anchor_dir = u_anchor_dir.unsqueeze(0).unsqueeze(2).expand(N, L, 1, 3)  # (N,L,1,3)

        # 全点位置与锚点差向量 v = X_j - X_anchor
        Xj = self.canonical_pointcloud.view(1, 1, M, 3).expand(N, L, -1, -1)  # (N,L,M,3)
        v = Xj - X_anchor  # (N,L,M,3)
        v_norm = v / (v.norm(dim=-1, keepdim=True).clamp_min(self.eps))  # (N,L,M,3)

        # cosine sim：s = dot(u_anchor_dir, v_norm) ∈ [-1, 1]
        s = (u_anchor_dir * v_norm).sum(dim=-1)  # (N,L,M)


        d_local = torch.norm(kp_local.unsqueeze(2) - Xj, dim=-1)  # (N,L,M)

        # multiplier∈[multiplier_min, 1]，仅在锚点簇内
        # s = dot(u_anchor_dir, v)  # (N,L,M), 未归一化投影
        s = (u_anchor_dir * v).sum(dim=-1)

        # 簇内最大投影 s_max，用 -inf 屏蔽簇外
        s_masked = torch.where(inside_cluster_mask, s, torch.full_like(s, float('-inf')))
        s_max = s_masked.max(dim=-1, keepdim=True).values  # (N,L,1)

        # 阈值 & 选择强同向点
        threshold = s_max * self.threshold_ratio
        pos = (s > threshold) & inside_cluster_mask

        # 线性映射 [threshold, s_max] → [0,1]
        scale = torch.zeros_like(s)
        valid = pos & (s_max > threshold)
        scale[valid] = (s - threshold)[valid] / (s_max - threshold).expand_as(s)[valid].clamp_min(self.eps)

        # 仅对强同向簇内点降低 multiplier；其它保持 1
        multiplier = torch.ones_like(d_local)
        multiplier[pos] = self.multiplier_min + (1.0 - self.multiplier_min) * (1.0 - scale[pos])

        score = d_local * multiplier
        closest_point_idx = score.argmin(dim=-1)
        P_c = torch.gather(Xj, dim=2, index=closest_point_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)  # (N,L,3)
        P_target_local = P_c
        
        original_closet_idx = d_local.argmin(dim=-1)
        original_P_c = torch.gather(Xj, dim=2, index=original_closet_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)  # (N,L,3)
        original_P_target_local = original_P_c

        # Step 5: local to world
        P_target_world = quat_apply(q.unsqueeze(1).expand(-1, L, -1), P_target_local) + t.unsqueeze(1)  # (N,L,3)
        original_P_target_world = quat_apply(q.unsqueeze(1).expand(-1, L, -1), original_P_target_local) + t.unsqueeze(1)  # (N,L,3)

        # Step 6: potential-based reaching reward
        dist_to_target = torch.norm(keypoint_positions_world - P_target_world, dim=-1)  # (N,L)
        avg_dist_to_target = dist_to_target.mean(dim=1)  # (N,)
        
        # aggregate first, then exp
        current_potential = torch.exp(-avg_dist_to_target / self.potential_sigma)
        if (self.prev_potential is None) or (self.prev_potential.shape != (N,)):
            self.prev_potential = current_potential.detach().clone()
        r_progress = current_potential - self.prev_potential
        self.prev_potential = current_potential.detach().clone()
        
        # delayed aggregation: per-link energy first, then average over links
        # current_potential_per_kp = torch.exp(-dist_to_target / self.potential_sigma)  # (N,L)
        # current_potential = current_potential_per_kp.mean(dim=-1)  # (N,)

        # if (self.prev_potential is None) or (self.prev_potential.shape != (N,)):
        #     self.prev_potential = current_potential.detach().clone()
        # r_progress = current_potential - self.prev_potential
        # self.prev_potential = current_potential.detach().clone()
            
        # current_potential_per_kp = torch.exp(-dist_to_target / self.potential_sigma)
        # if self.potential_per_kp_max is None or (self.potential_per_kp_max.shape != (N, L)):
        #     self.potential_per_kp_max = torch.zeros_like(current_potential_per_kp) # E_d = 0 -> dist -> inf
        # current_potential = self.potential_per_kp_max.mean(dim=-1)
        # r_progress = torch.clip(current_potential_per_kp - self.potential_per_kp_max, min=0).mean(dim=-1)
        # self.potential_per_kp_max = torch.max(self.potential_per_kp_max, current_potential_per_kp)

        # Step 7: novelty-based contact reward
        #Ruoyi: not tested/used for now
        contact_novelty_reward = torch.zeros((N, L), dtype=avg_dist_to_target.dtype, device=device)
        if contact_bonus > 0 and(contact_indices is not None) and cm.any():
            has = cm
            ei, ki = torch.nonzero(has, as_tuple=True)
            pj = contact_indices[ei, ki].clamp(0, M - 1)
            cluster_ids = self._point_to_cluster[pj]
            counts = self._per_fingertip_cluster_counts[ki, cluster_ids]
            contact_novelty_reward[ei, ki] = 1.0 / torch.sqrt(1.0 + counts.float())
        bonus_term = contact_novelty_reward.mean(dim=1)

        reward = r_progress + contact_bonus * bonus_term
        # reward = contact_bonus * bonus_term

        # viz
        self.last_P_target = P_target_world.detach().clone()
        self.last_original_P_target = original_P_target_world.detach().clone()
        
        anchor_world = quat_apply(q.unsqueeze(1).expand(-1, L, -1), X_anchor.squeeze(2)) + t.unsqueeze(1)
        self.last_anchor = anchor_world.detach().clone()

        info = {
            "avg_dist_to_target": avg_dist_to_target,
            "progress": r_progress,
            "contact_count": cm.sum(dim=1),
            "cluster_novelty_reward": bonus_term,
            "current_potential": current_potential,
        }
        return reward, info