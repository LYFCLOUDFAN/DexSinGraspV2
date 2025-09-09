import torch
from typing import Optional, Dict, Tuple


class CuriosityRewardManager:
    """
    CuriosityRewardManager
    - 全局（跨环境）维护接触计数热图 contact_heatmap: (L, M)
    - 基于点云构建 kNN 图平滑 contact_heatmap → smoothed_heatmap，并计算 -∇Ĥ 方向
    - 用新奇度与方向对齐度做 soft-attend 得到“期望距离”，以前后差分为主奖励
    - 可叠加 contact bonus，并用新奇度增强
    """

    def __init__(
        self,
        num_keypoints: Optional[int] = None,
        num_object_points: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        # 默认超参
        k: int = 16,
        temp: float = 0.02,
        w_curv: float = 0.5,
        w_align: float = 0.5,
        contact_bonus: float = 0.0,
        per_contact: bool = False,
        eps: float = 1e-8,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 全局（跨环境）接触计数热图: (L, M); 延迟初始化
        self.contact_heatmap: Optional[torch.Tensor] = None  # long, (L, M)

        # 运行时缓存
        self.prev_expected_distance_mean: Optional[torch.Tensor] = None  # (N,)

        self.L = num_keypoints
        self.M = num_object_points

        self.default_k = k
        self.default_temp = temp
        self.default_w_curv = w_curv
        self.default_w_align = w_align
        self.default_contact_bonus = contact_bonus
        self.default_per_contact = per_contact
        self.eps = eps

        self.smoothed_heatmap: Optional[torch.Tensor] = None           # Ĥ
        self.heatmap_gradient: Optional[torch.Tensor] = None           # ∇H（内部未直接暴露）
        self.gradient_direction: Optional[torch.Tensor] = None         # G_dir
        self.gradient_magnitude: Optional[torch.Tensor] = None         # G_mag
        self.novelty_score: Optional[torch.Tensor] = None              # Nvl

    # ---------- 公共接口 ----------

    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.LongTensor] = None, reset_counters: bool = False):
        """
        - env_ids: 仅用于重置进度基线（prev_expected_distance_mean）的对应环境位置
        - reset_counters=True: 清零全局 contact_heatmap
        """
        if reset_counters and self.contact_heatmap is not None:
            self.contact_heatmap.zero_()

        # 重置进度基线
        if self.prev_expected_distance_mean is not None:
            if env_ids is None:
                self.prev_expected_distance_mean = None
            else:
                self.prev_expected_distance_mean[env_ids] = 0.0

    @torch.no_grad()
    def reset_counters(self):
        """清零全局接触热图计数。"""
        if self.contact_heatmap is not None:
            self.contact_heatmap.zero_()


    @torch.no_grad()
    def update_contact_heatmap(self, contact_indices: torch.Tensor, contact_mask: torch.Tensor, num_object_points: Optional[int] = None):
        """
        累计全局接触计数热图（跨环境）。
        - contact_indices: (N, L) long, 取值范围 [0, M-1]
        - contact_mask: (N, L) bool/byte, 1=有接触, 0=无接触
        - 全局 self.contact_heatmap 形状: (L, M)
        """
        assert contact_indices.dtype in (torch.int32, torch.int64)
        N, L = contact_indices.shape

        if self.L is None:
            self.L = L
        else:
            assert self.L == L, f"L mismatch: {self.L} != {L}"

        # 推断/校验 M
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
        j_idx = contact_indices[env_idx, kp_idx].clamp(min=0, max=self.M - 1)  # Map kp_idx to pd_idx

        # Merge env dim
        lin = kp_idx.to(torch.long) * self.M + j_idx.to(torch.long) # kp * M
        counts = torch.bincount(lin, minlength=self.L * self.M)
        counts = counts.view(self.L, self.M).to(self.contact_heatmap.dtype).to(self.contact_heatmap.device)

        self.contact_heatmap.add_(counts)

    @torch.no_grad()
    def compute_reward(
        self,
        tau: float,
        object_pointclouds: torch.Tensor,   # (N, M, 3)
        keypoint_positions: torch.Tensor,   # (N, L, 3)  — fingertip positions (no offset)
        contact_indices: Optional[torch.Tensor] = None,  # (N, L) long, [0..M-1]
        contact_mask: Optional[torch.Tensor] = None,     # (N, L) bool
        *,
        k: Optional[int] = None,
        temp: Optional[float] = None,
        w_curv: Optional[float] = None,
        w_align: Optional[float] = None,
        contact_bonus: Optional[float] = None,
        per_contact: Optional[bool] = None,
        normalize: str = "tau",
        object_normals: Optional[torch.Tensor] = None,   # (N, M, 3) optional surface normals
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return:
          - reward: (N,)
          - info: Dict[str, Tensor]
        """
        device = object_pointclouds.device
        N, M, _ = object_pointclouds.shape
        _, L, _ = keypoint_positions.shape

        # params
        k = int(k if k is not None else self.default_k)
        temp = float(temp if temp is not None else self.default_temp)
        w_curv = float(w_curv if w_curv is not None else self.default_w_curv)
        w_align = float(w_align if w_align is not None else self.default_w_align)
        contact_bonus = float(contact_bonus if contact_bonus is not None else self.default_contact_bonus)
        per_contact = bool(self.default_per_contact if per_contact is None else per_contact)

        # 初始化全局计数热图 (L, M) 并累计本步接触
        self.update_contact_heatmap(contact_indices.to(device), contact_mask.to(device), num_object_points=M)

        contact_heatmap = self.contact_heatmap.to(torch.float32)
        novelty_score = 1.0 / (1.0 + contact_heatmap)  # (L, M)
        self.novelty_score = novelty_score
        
        # get k-near point idx for each point
        k_eff = min(k, max(1, M))
        knn_idx = self._knn_indices(object_pointclouds, k=k_eff)  # (N, M, k)

        # use per-env graph to smooth global heatmap
        H = contact_heatmap.unsqueeze(0).expand(N, -1, -1)  # (N, L, M)
        smoothed_heatmap, gradient_direction, gradient_magnitude = self._smooth_and_gradient(
            object_pointclouds, H, knn_idx
        )
        self.smoothed_heatmap = smoothed_heatmap
        self.gradient_direction = gradient_direction
        self.gradient_magnitude = gradient_magnitude

        # simplified novelty: drop min-max normalization and global mean term
        novelty_smoothed = novelty_score                               # (L, M)

        # distances for reachability and logits
        D = torch.cdist(keypoint_positions, object_pointclouds, p=2)   # (N, L, M)

        # pick the best unexplored-but-reachable point
        reach_thr = tau * 1.5
        reachable = (D < reach_thr)                                    # (N, L, M)
        novelty_map = novelty_smoothed.unsqueeze(0)                    # (1, L, M) -> broadcast
        novelty_reach_score = novelty_map * (1.0 / (D + self.eps))     # (N, L, M)
        novelty_reach_score = torch.where(reachable, novelty_reach_score, torch.zeros_like(novelty_reach_score))

        j_explore = novelty_reach_score.argmax(dim=-1)                 # (N, L)

        # guidance direction at the exploration points: use -∇H
        d_i = torch.gather(
            gradient_direction, dim=2, index=j_explore.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        ).squeeze(2)  # (N, L, 3)
        d_i = d_i / (d_i.norm(dim=-1, keepdim=True).clamp_min(self.eps))

        # alignment with direction (positive only)
        v = object_pointclouds.unsqueeze(1) - keypoint_positions.unsqueeze(2)  # (N, L, M, 3)
        v_hat = v / v.norm(dim=-1, keepdim=True).clamp_min(self.eps)           # (N, L, M, 3)
        align = (v_hat * d_i.unsqueeze(2)).sum(dim=-1)                         # (N, L, M)
        align_pos = torch.clamp(align, min=0.0)

        # optional surface-normal constraint to keep motion along/into surface
        if object_normals is not None:
            normal_align = (v_hat * object_normals.unsqueeze(1)).sum(dim=-1)   # (N, L, M)
            normal_align_pos = torch.clamp(normal_align, min=0.0)
            align_pos = align_pos * normal_align_pos

        # soft attention fusion: distance + novelty + alignment
        logits = (- D / max(temp, self.eps)) \
                 + w_curv * novelty_smoothed.unsqueeze(0) \
                 + w_align * align_pos                                         # (N, L, M)
        attn = torch.softmax(logits, dim=-1)                                   # (N, L, M)

        # expected distance (no tau normalization)
        expected_distance = (attn * D).sum(dim=-1)                             # (N, L)
        E_mean = expected_distance.mean(dim=1)                                  # (N,)

        # progress reward (prev - current)
        if (self.prev_expected_distance_mean is None) or (self.prev_expected_distance_mean.shape != (N,)):
            self.prev_expected_distance_mean = E_mean.detach().clone()
        r_progress = self.prev_expected_distance_mean - E_mean

        # contact bonus (mask-driven)
        cm = contact_mask.to(torch.bool)
        if per_contact:
            bonus_term = cm.sum(dim=1).to(E_mean.dtype)
        else:
            bonus_term = cm.any(dim=1).to(E_mean.dtype)

        if contact_indices is not None and contact_mask is not None:
            # novelty on actual hits (use global novelty map)
            novelty_on_hits = torch.zeros((N, L), dtype=E_mean.dtype, device=device)
            has = cm
            if has.any():
                ei, ki = torch.nonzero(has, as_tuple=True)
                pj = contact_indices[ei, ki].clamp(0, M - 1)
                novelty_on_hits[ei, ki] = novelty_smoothed[ki, pj]
            novelty_hit = novelty_on_hits.mean(dim=1)  # (N,)
            bonus_term = bonus_term * (1.0 + novelty_hit)

        reward = r_progress + contact_bonus * bonus_term

        # update baseline
        self.prev_expected_distance_mean = E_mean.detach().clone()

        info = {
            "E_mean": E_mean,                           # (N,)
            "progress": r_progress,                     # (N,)
            "contact_count": cm.sum(dim=1),             # (N,)
        }
        return reward, info

    # ---------- 私有方法 ----------

    @torch.no_grad()
    def _knn_indices(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """
        X: (N, M, 3)
        返回 knn_idx: (N, M, k)
        """
        d = torch.cdist(X, X, p=2)  # (N, M, M)
        knn_idx = d.topk(k=k, dim=-1, largest=False).indices
        return knn_idx

    @torch.no_grad()
    def _smooth_and_gradient(
        self,
        pc: torch.Tensor,                # (N, M, 3)
        H: torch.Tensor,                 # (N, L, M) - 由全局 (L, M) 扩展
        knn_idx: torch.Tensor,           # (N, M, k)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
        smoothed_heatmap:   (N, L, M)
        gradient_direction: (N, L, M, 3) 指向未探索（-∇H）的单位向量
        gradient_magnitude: (N, L, M)
        """
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
        
        # Then expand to include L dimension
        X_nb = X_nb_base.unsqueeze(1).expand(-1, L, -1, -1, -1)  # (N, L, M, k, 3)
        
        # X is (N, L, M, 3)
        X = pc.unsqueeze(1).expand(-1, L, -1, -1)  # (N, L, M, 3)

        dX = X_nb - X.unsqueeze(-2)                 # (N, L, M, k, 3)
        dH = H_nb - smoothed_heatmap.unsqueeze(-1)  # (N, L, M, k)

        denom = (dX.pow(2).sum(dim=-1) + self.eps)  # (N, L, M, k)
        G = (dH.unsqueeze(-1) * dX / denom.unsqueeze(-1)).sum(dim=-2)  # (N, L, M, 3)

        G_mag = G.norm(dim=-1)        # (N, L, M)
        G_dir = torch.zeros_like(G)   # (N, L, M, 3)
        nonzero = (G_mag > 1e-8)
        if nonzero.any():
            # Use -∇H to point toward LOWER heat (unexplored regions)
            G_dir[nonzero] = (-G[nonzero] / G_mag[nonzero].unsqueeze(-1))

        return smoothed_heatmap, G_dir, G_mag