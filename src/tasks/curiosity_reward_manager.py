import torch
from typing import Optional, Dict, Tuple

class CuriosityRewardManager:
    """
    - Maintains a global (cross-environment) contact count heatmap, either per-point or per-cluster.
    - Core reward: for each fingertip, compute the novelty-weighted average distance within a local reachable neighborhood to the object point cloud; use the forward difference (previous - current) as progress reward.
    - Reachability constraint: only points within reachability_threshold are considered to keep exploration local and along the surface.
    - Decay: apply exponential decay to the heatmap at each step so previously explored regions cool down over time.
    - An optional contact bonus can be added and scaled by novelty at actual contacts.
    """

    def __init__(
        self,
        num_keypoints: Optional[int] = None,
        num_object_points: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        # default hyper-parameters
        decay_factor: float = 0.999,  # decay per step for the heatmap
        contact_bonus: float = 0.0,
        per_contact: bool = False,
        eps: float = 1e-8,
        # Reachability threshold (key parameter)
        reachability_threshold: float = 0.03,  # 3 cm;
        # Clustering parameters
        use_clustering: bool = True,
        cluster_k: int = 16,
        max_clustering_iters: int = 10,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global heatmap: (L, M) - per point or (L, cluster_k) - per cluster
        self.contact_heatmap: Optional[torch.Tensor] = None
        # If using clustering, this stores the cluster ID for each point in the point cloud
        self._point_to_cluster: Optional[torch.Tensor] = None  # (M,)
        # If using clustering, this stores the novelty score for each cluster (L, cluster_k)
        self._cluster_novelty: Optional[torch.Tensor] = None

        # Runtime cache
        self.prev_expected_distance_mean: Optional[torch.Tensor] = None  # (N,)

        self.L = num_keypoints
        self.M = num_object_points

        self.default_contact_bonus = contact_bonus
        self.default_per_contact = per_contact
        self.eps = eps

        # Reachability & Decay
        self.decay_factor = decay_factor
        self.reachability_threshold = reachability_threshold

        # Clustering config
        self.use_clustering = use_clustering
        self.cluster_k = cluster_k
        self.max_clustering_iters = max_clustering_iters

        # Cache
        self.novelty_score: Optional[torch.Tensor] = None  # (L, M) or (L, cluster_k)


    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.LongTensor] = None, reset_counters: bool = False):
        if reset_counters and self.contact_heatmap is not None:
            self.contact_heatmap.zero_()
        if self.prev_expected_distance_mean is not None:
            if env_ids is None:
                self.prev_expected_distance_mean = None
            else:
                self.prev_expected_distance_mean[env_ids] = 0.0

    @torch.no_grad()
    def reset_counters(self):
        if self.contact_heatmap is not None:
            self.contact_heatmap.zero_()

    @torch.no_grad()
    def _perform_clustering(self, pointcloud: torch.Tensor):
        """
        Args:
            pointcloud: (M, 3)
        Returns:
            cluster_labels: (M,)
        """
        M, _ = pointcloud.shape
        points = pointcloud.to(torch.float32)

        # random initialize cluster centers
        indices = torch.randperm(M, device=self.device)[:self.cluster_k]
        centers = points[indices].clone()  # (cluster_k, 3)

        for _ in range(self.max_clustering_iters):
            # compute distance from each point to each center
            distances = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # (M, cluster_k)
            # assign points to the nearest center
            labels = torch.argmin(distances, dim=1)  # (M,)
            # recompute centers
            new_centers = torch.zeros_like(centers)
            for i in range(self.cluster_k):
                mask = (labels == i)
                if mask.any():
                    new_centers[i] = points[mask].mean(dim=0)
                else:
                    # if a cluster has no points, keep the old center
                    new_centers[i] = centers[i]
            centers = new_centers

        return labels

    @torch.no_grad()
    def update_contact_heatmap(self, contact_indices: torch.Tensor, contact_mask: torch.Tensor, num_object_points: Optional[int] = None):
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
            if self.use_clustering:
                self.contact_heatmap = torch.zeros((self.L, self.cluster_k), dtype=torch.float32, device=self.device)
            else:
                self.contact_heatmap = torch.zeros((self.L, self.M), dtype=torch.float32, device=self.device)

        # Apply decay
        self.contact_heatmap.mul_(self.decay_factor)
        
        has = contact_mask
        if not has.any():
            return

        env_idx, kp_idx = torch.nonzero(has, as_tuple=True)
        j_idx = contact_indices[env_idx, kp_idx].clamp(min=0, max=self.M - 1)

        if self.use_clustering:
            # if clustering mode, need to convert point index j_idx to cluster index
            if self._point_to_cluster is None:
                # Lazy clustering: perform clustering when first needed
                sample_pc = contact_indices.new_zeros(self.M, 3) # Dummy, will be covered by compute_reward

                raise RuntimeError("Clustering not initialized. Please call compute_reward first.")

            cluster_idx = self._point_to_cluster[j_idx]  # (num_contacts,)
            # Merge env dim for cluster indices
            lin = kp_idx.to(torch.long) * self.cluster_k + cluster_idx.to(torch.long)
            counts = torch.bincount(lin, minlength=self.L * self.cluster_k)
            counts = counts.view(self.L, self.cluster_k).to(self.contact_heatmap.dtype).to(self.contact_heatmap.device)
        else:
            # per-point mode
            lin = kp_idx.to(torch.long) * self.M + j_idx.to(torch.long)
            counts = torch.bincount(lin, minlength=self.L * self.M)
            counts = counts.view(self.L, self.M).to(self.contact_heatmap.dtype).to(self.contact_heatmap.device)

        self.contact_heatmap.add_(counts)

    @torch.no_grad()
    def compute_reward(
        self,
        tau: float,
        object_pointclouds: torch.Tensor,   # (N, M, 3)
        keypoint_positions: torch.Tensor,   # (N, L, 3)
        contact_indices: Optional[torch.Tensor] = None,
        contact_mask: Optional[torch.Tensor] = None,
        *,
        contact_bonus: Optional[float] = None,
        per_contact: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device = object_pointclouds.device
        N, M, _ = object_pointclouds.shape
        _, L, _ = keypoint_positions.shape

        contact_bonus = float(contact_bonus if contact_bonus is not None else self.default_contact_bonus)
        per_contact = bool(self.default_per_contact if per_contact is None else per_contact)

        # perform clustering if enabled and not initialized
        if self.use_clustering and (self._point_to_cluster is None or self._point_to_cluster.shape[0] != M):
            # use first env's point cloud for clustering
            sample_pointcloud = object_pointclouds[0]  # (M, 3)
            self._point_to_cluster = self._perform_clustering(sample_pointcloud)  # (M,)
            # initialize or reset cluster-level heatmap
            if self.contact_heatmap is None or self.contact_heatmap.shape[1] != self.cluster_k:
                self.contact_heatmap = torch.zeros((self.L, self.cluster_k), dtype=torch.float32, device=self.device)

        # Update heatmap
        self.update_contact_heatmap(contact_indices.to(device), contact_mask.to(device), num_object_points=M)

        # compute novelty score
        if self.use_clustering:
            # compute novelty for each cluster
            cluster_novelty = 1.0 / (1.0 + self.contact_heatmap)  # (L, cluster_k)
            self._cluster_novelty = cluster_novelty
            # broadcast cluster novelty to each point
            point_novelty = cluster_novelty[:, self._point_to_cluster]  # (L, M)
            self.novelty_score = point_novelty
        else:
            # per-point 
            point_novelty = 1.0 / (1.0 + self.contact_heatmap)  # (L, M)
            self.novelty_score = point_novelty

        # distances fingertip-to-all points
        D = torch.cdist(keypoint_positions, object_pointclouds, p=2)     # (N, L, M)

        # weights: novelty_score
        weights = point_novelty.unsqueeze(0)  # (1, L, M) -> (N, L, M)

        # Apply reachability constraint
        reachable = (D < self.reachability_threshold)
        weights = weights * reachable.to(weights.dtype)

        # Compute weighted average distance
        weighted_D = (weights * D).sum(dim=-1) / (weights.sum(dim=-1) + self.eps)  # (N, L)
        avg_weighted_D = weighted_D.mean(dim=1)                                    # (N,)

        # progress reward
        if (self.prev_expected_distance_mean is None) or (self.prev_expected_distance_mean.shape != (N,)):
            self.prev_expected_distance_mean = avg_weighted_D.detach().clone()
        r_progress = self.prev_expected_distance_mean - avg_weighted_D
        self.prev_expected_distance_mean = avg_weighted_D.detach().clone()

        # contact bonus (no used for now)
        cm = contact_mask.to(torch.bool) if contact_mask is not None else torch.zeros((N, L), dtype=torch.bool, device=device)
        if per_contact:
            bonus_term = cm.sum(dim=1).to(avg_weighted_D.dtype)
        else:
            bonus_term = cm.any(dim=1).to(avg_weighted_D.dtype)

        novelty_hit = torch.zeros((N,), dtype=avg_weighted_D.dtype, device=device)
        if (contact_indices is not None) and (contact_mask is not None):
            novelty_on_hits = torch.zeros((N, L), dtype=avg_weighted_D.dtype, device=device)
            has = cm
            if has.any():
                ei, ki = torch.nonzero(has, as_tuple=True)
                pj = contact_indices[ei, ki].clamp(0, M - 1)
                if self.use_clustering:
                    cluster_id_for_hits = self._point_to_cluster[pj]
                    novelty_on_hits[ei, ki] = self._cluster_novelty[ki, cluster_id_for_hits]
                else:
                    novelty_on_hits[ei, ki] = point_novelty[ki, pj]
            novelty_hit = novelty_on_hits.mean(dim=1)
            bonus_term = bonus_term * (1.0 + novelty_hit)

        reward = r_progress + contact_bonus * bonus_term

        info = {
            "avg_weighted_distance": avg_weighted_D,
            "progress": r_progress,
            "contact_count": cm.sum(dim=1),
            "novelty_hit_mean": novelty_hit,
            "clustering_enabled": torch.tensor([self.use_clustering], device=device), # for logging
        }
        return reward, info