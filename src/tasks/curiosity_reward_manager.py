import torch
from typing import Optional, Dict, Tuple
import math


def perturb_unit_direction(u, angle_std_deg=2.0, eps=1e-8):
    """
    u: (..., 3) 已归一化的单位向量
    返回: (..., 3) 扰动后的单位向量，且与原向量夹角 ≤ angle_max_deg（或~N(0, angle_std_deg)）
    """
    assert u.shape[-1] == 3
    device = u.device
    # 随机向量并投影到切平面
    r = torch.randn_like(u)
    v = r - (r * u).sum(dim=-1, keepdim=True) * u                      # 去掉平行分量
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)                # 归一化为切向


    # N(0, σ) 的小角扰动（可与 3σ≈angle_max 对齐）
    theta = torch.randn(u.shape[:-1] + (1,), device=device) * math.radians(angle_std_deg)

    # 旋转到新方向（罗德里格斯在该构造下简化为 cos/sin 组合）
    u_new = u * torch.cos(theta) + v * torch.sin(theta)
    # 数值保险：再归一化一次
    u_new = u_new / (u_new.norm(dim=-1, keepdim=True) + eps)
    return u_new


class CuriosityRewardManager:

    def __init__(
        self,
        num_keypoints: Optional[int] = None,
        num_object_points: Optional[int] = None,
        device: Optional[torch.device] = None,
        curiosity_reward_scale: float = 1.0,
        reaching_reward_scale: float = 1.0,
        eps: float = 1e-6,
        gradient_noise_std_deg: Optional[float] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L = num_keypoints
        self.M = num_object_points
        self.contact_heatmap = torch.zeros((self.L, self.M), dtype=torch.long, device=self.device)
        self.object_pointcloud = torch.zeros((self.M, 3), dtype=torch.float32, device=self.device)
        self.keypoint_positions = torch.zeros((self.L, 3), dtype=torch.float32, device=self.device)
        self.contact_mask = torch.zeros((self.L,), dtype=torch.bool, device=self.device)
        self.nearest_indices = torch.zeros((self.L,), dtype=torch.long, device=self.device)
        self.dist_min = torch.zeros((self.L,), dtype=torch.float32, device=self.device)
        
        self.curiosity_reward_scale = curiosity_reward_scale
        self.reaching_reward_scale = reaching_reward_scale
        self.eps = eps
        
        self.gradient_noise_std_deg = gradient_noise_std_deg

    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.LongTensor] = None, reset_counters: bool = False):
        if reset_counters:
            self.contact_heatmap.zero_()

    @torch.no_grad()
    def reset_counters(self):
        """zero out all counters"""
        self.contact_heatmap.zero_()
            
    @torch.no_grad()
    def _knn_indices(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """
        X: (M, 3)
        Returns knn_idx: (M, k)
        """
        d = torch.cdist(X, X, p=2)  # (M, M)
        knn_idx = d.topk(k=k, dim=-1, largest=False).indices
        return knn_idx

    @torch.no_grad()
    def _smooth_and_gradient(
        self,
        pc: torch.Tensor,        # (M, 3)
        H: torch.Tensor,         # (L, M)
        knn_idx: torch.Tensor,   # (M, k)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          smoothed_heatmap:   (L, M)
          gradient_direction: (L, M, 3)  unit vectors of -∇H
          gradient_magnitude: (L, M)
        """
        Hf = H.to(torch.float32).unsqueeze(-1).expand(-1, -1, knn_idx.shape[-1])  # (L, M, k)

        # neighbor gather: (L, M, k)
        H_nb = Hf.gather(dim=1, index=knn_idx.unsqueeze(0).expand(self.L, -1, -1)) # (L, M, k)
        Hf = 0.5 * Hf + 0.5 * H_nb  # (L, M, k)

        # neighbor positions: (M, k, 3)
        X_nb = pc[knn_idx]                      # (M, k, 3)
        X = pc.unsqueeze(1)                     # (M, 1, 3)
        dX = X_nb - X                           # (M, k, 3)

        # expand H terms to (L, M, k)
        dH = H_nb - Hf  # (L, M, k)

        denom = (dX.pow(2).sum(dim=-1) + self.eps)  # (M, k)
        # broadcast denom to (L, M, k, 1) via (1, M, k, 1)
        G = (dH.unsqueeze(-1) * dX.unsqueeze(0) / denom.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)  # (L, M, 3)

        G_mag = G.norm(dim=-1)  # (L, M)
        G_dir = -G / (G_mag.unsqueeze(-1) + self.eps)  # (L, M, 3)
        return Hf, G_dir, G_mag
    
    @torch.no_grad()
    def update_contact_heatmap(
        self,
        contact_indices: torch.Tensor,  # (L,) long in [0, M-1]
        contact_mask: torch.Tensor,     # (L,) bool
    ):
        """
        Accumulate global contact counts (cross-env).
        Global self.contact_heatmap shape: (L, M)
        """
        has = contact_mask.to(torch.bool)
        if not has.any():
            return

        kp_idx = torch.nonzero(has, as_tuple=True)[0]
        j_idx = contact_indices[kp_idx].clamp(min=0, max=self.M - 1)

        # accumulate
        print("before heatmap add", self.contact_heatmap.sum().item())
        self.contact_heatmap[kp_idx, j_idx] += 1
        print("after heatmap add", self.contact_heatmap.sum().item())
        print(f"Updated heatmap at {len(kp_idx)} keypoints.")

    @torch.no_grad()
    def update_contact_state(
        self,
        object_pointcloud: torch.Tensor,    # (M, 3)
        keypoint_positions: torch.Tensor,   # (L, 3)
        contact_force: Optional[torch.Tensor] = None,  # (L, 3) or None
        dist_threshold: float = 0.01,              # 距离阈值 (判定接触)
        force_threshold: Optional[float] = None,  # 力阈值 (可选；仅当提供 contact_force 才生效)
    ):
        """
        统一更新本步的几何与接触状态，并回写全局 contact_heatmap：
        - 计算距离矩阵 D、最近点 j_nearest
        - dist_mask = (D_min < dist_threshold)
        - 若提供 contact_force：force_mask = (||F|| >= force_threshold)
        - contact_mask = dist_mask & force_mask (若未提供力，则 contact_mask = dist_mask)
        - 用 contact_mask 与 j_nearest 调用 update_contact_heatmap
        额外：将 D, j_nearest, contact_mask 存为类属性供后续奖励函数复用。
        """
        assert self.device == object_pointcloud.device == keypoint_positions.device, f"device mismatch {self.device} vs {object_pointcloud.device} vs {keypoint_positions.device}"
        assert self.M == object_pointcloud.shape[0], f"object point count mismatch {self.M} vs {object_pointcloud.shape[0]}"
        assert self.L == keypoint_positions.shape[0], f"keypoint count mismatch {self.L} vs {keypoint_positions.shape[0]}"

        # 距离矩阵与最近点
        D = torch.cdist(keypoint_positions, object_pointcloud, p=2)  # (L, M)
        j_nearest = torch.argmin(D, dim=-1)                           # (L,)
        d_min = D[torch.arange(self.L, device=self.device), j_nearest]          # (L,)

        # 距离掩码
        dist_mask = d_min < float(dist_threshold)

        # 力掩码（可选）
        if contact_force is not None:
            assert force_threshold is not None, "提供 contact_force 时需给 force_threshold"
            f_norm = contact_force.norm(dim=-1)                       # (L,)
            force_mask = f_norm >= float(force_threshold)
            contact_mask = dist_mask & force_mask
        else:
            contact_mask = dist_mask
        
        # 回写计数（把“确实接触”的 (k, j_nearest[k]) 计入全局热图）
        if contact_mask.any():
            k_idx = torch.nonzero(contact_mask, as_tuple=True)[0]     # (Kc,)
            j_idx = j_nearest[k_idx].clamp(0, self.M - 1)
            self.update_contact_heatmap(j_idx, torch.ones_like(k_idx, dtype=torch.bool))
            # 说明：这里的 update_contact_heatmap 期望 (L,) 形状；为了最少改动，上面传入的是
            # 仅包含接触 keypoints 的索引。若你保留的是 (L,) 版本，请替换为：
            #   - contact_indices_full = j_nearest
            #   - contact_mask_full = contact_mask
            #   - self.update_contact_heatmap(contact_indices_full, contact_mask_full, num_object_points=M)
        else:
            print("No contacts detected this step; heatmap unchanged.")

        # 存到类属性，供奖励函数直接使用
        self.distance_matrix = D
        self.object_pointcloud = object_pointcloud
        self.keypoint_positions = keypoint_positions
        self.contact_mask = contact_mask
        self.nearest_indices = j_nearest
        self.dist_min = d_min

    @torch.no_grad()
    def compute_reaching_reward(
        self,
        k: int = 16,
    ):
        """
        使用“最高热点邻域 + 梯度方向余弦权重”的 reaching 奖励。
        仅在被调用时计算梯度/邻域，避免无用算子。
        """
        H = self.contact_heatmap.to(torch.float32)
        D = self.distance_matrix
        L, M = D.shape
        pc = self.object_pointcloud

        # 找热图最高点（全局）
        j_peak = torch.argmax(H, dim=-1)  # (L,)
        j_peak_rows = torch.arange(L, device=pc.device)

        # 梯度方向（-∇H）
        knn_idx = self._knn_indices(pc, k=min(k, M))  # (M, k)
        _, G_dir, _ = self._smooth_and_gradient(pc, H, knn_idx)  # (L, M, 3)

        g_col = G_dir[j_peak_rows, j_peak, :]     # (L, 3)
        g_peak = g_col.mean(dim=0)
        g_peak = g_peak / (g_peak.norm() + self.eps)
        
        if self.gradient_noise_std_deg is not None:
            g_peak = perturb_unit_direction(g_peak, angle_std_deg=self.gradient_noise_std_deg, eps=self.eps)
            
        # 邻域（包含中心点更稳）
        nbrs = knn_idx[j_peak]  # (L, k)

        v = pc[nbrs] - pc[j_peak.unsqueeze(1)]
        v_hat = v / (v.norm(dim=-1, keepdim=True) + self.eps)
        cos_sim = (v_hat * g_peak.unsqueeze(0)).sum(dim=-1).clamp(-1, 1)
        w_local = (1 - cos_sim) * 0.5

        w_all = torch.ones(L, M, device=pc.device)
        rows = torch.arange(L, device=pc.device).unsqueeze(1)
        w_all[rows, nbrs] = w_local

        # 按权重选点，但奖励用真实距离
        D_w = D * w_all
        j_star = torch.argmin(D_w, dim=-1)
        d_star = D[rows, j_star]

        r = - d_star.mean()

        return r, {"j_peak": j_peak, "j_star": j_star, "d_star": d_star, "nbrs": nbrs, "G_dir": G_dir, "w_all": w_all, "D": D}

    @torch.no_grad()
    def compute_curiosity_reward(
        self,
    ):
        """
        在“确实接触”的 keypoint 上，用全局计数的 1/(1+count) 给奖励。
        这里的“确实接触”复用 update_contact_state 中的最近点与距离判定。
        """
        H = self.contact_heatmap.to(torch.float32)
        L = self.L

        j_star = self.nearest_indices                 # (L,)
        d_star = self.dist_min                        # (L,)
        contact_mask = self.contact_mask              # (L,)
        
        counts = H[torch.arange(L, device=H.device), j_star.to(H.device)]
        novelty = 1.0 / (1.0 + counts)
        novelty_on_contacts = torch.where(contact_mask, novelty, torch.zeros_like(novelty))

        r = novelty_on_contacts.sum()

        return r, {"j_star": j_star, "d_star": d_star, "novelty": novelty_on_contacts}

    @torch.no_grad()
    def compute_reward(
        self,
        reward_types: list,                      # 例如 ["reaching","curiosity"]
        object_pointcloud: torch.Tensor,         # (M, 3)
        keypoint_positions: torch.Tensor,        # (L, 3)
        contact_force: Optional[torch.Tensor] = None,  # (L, 3) or None
        dist_threshold: float = 0.01,
        force_threshold: Optional[float] = None,
        # reaching 超参
        k: int = 16,
    ):
        """
        统一入口：
        1) 用距离阈值 + （可选）力阈值在本步判定接触并更新全局计数；
        2) 按 reward_types 选择性计算并线性加权。
        """
        # 1) 更新状态 & 全局计数
        self.update_contact_state(
            object_pointcloud,
            keypoint_positions,
            contact_force,
            dist_threshold=dist_threshold,
            force_threshold=force_threshold,
        )

        # 2) 逐项计算
        total = torch.tensor(0.0, device=object_pointcloud.device)
        info = {}

        if "reaching" in reward_types:
            r_r, i_r = self.compute_reaching_reward(k=k)
            total += self.reaching_reward_scale * r_r
            info.update({f"reaching_{k}": v for k, v in i_r.items()})
            info["reaching_weight"] = torch.tensor(self.reaching_reward_scale, device=total.device)

        if "curiosity" in reward_types:
            r_c, i_c = self.compute_curiosity_reward()
            total += self.curiosity_reward_scale * r_c
            info.update({f"curiosity_{k}": v for k, v in i_c.items()})
            info["curiosity_weight"] = torch.tensor(self.curiosity_reward_scale, device=total.device)

        info["total_reward"] = total
        return total, info
