import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from termcolor import colored
import math

class DiscretizationNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DiscretizationNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for layer_index in range(len(hidden_sizes)):
            if layer_index == len(hidden_sizes) - 1:
                layers.append(nn.Linear(hidden_sizes[layer_index], output_size))
            else:
                layers.append(nn.Linear(hidden_sizes[layer_index], hidden_sizes[layer_index + 1]))
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        x = self.mlp(x)
        return (x>0.)
    
class NeuralHashCuriosity:
    def __init__(self, cfg, device, num_envs):
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        
        # Neural hash network
        self.hashnn = DiscretizationNN(
            cfg["obs_dim"], 
            cfg["hidden_sizes_hash"], 
            cfg["pred_dim"]
        ).to(device)
        
        # Bin counter for hash buckets
        self.bin_cnt = torch.zeros(
            2**cfg["pred_dim"], 
            dtype=torch.long, 
            device=device, 
            requires_grad=False
        )
        
        # self.obs_lb = torch.tensor(cfg.obs_lb, device=device).unsqueeze(0)
        # self.obs_ub = torch.tensor(cfg.obs_ub, device=device).unsqueeze(0)
        
        # self.obs_lb = torch.cat([
        #     torch.ones(1, 48, device=device) * -0.2,
        #     torch.ones(1, 32, device=device) *  0.0,
        # ], dim=-1)

        # self.obs_ub = torch.cat([
        #     torch.ones(1, 48, device=device) * 0.2,
        #     torch.ones(1, 32, device=device) * 0.4,
        # ], dim=-1)
        
        repr_type = cfg.get("repr", "nearest_surface")  # "basic" | "nearest_surface" | "surface_all_fingertips"
        if "obs_lb" in cfg and "obs_ub" in cfg:
            self.obs_lb = torch.tensor(cfg["obs_lb"], device=device).unsqueeze(0)
            self.obs_ub = torch.tensor(cfg["obs_ub"], device=device).unsqueeze(0)
        elif repr_type == "nearest_surface":
            self.obs_lb = torch.tensor([[
                -1.0, -1.0, -1.0, 0.0,
                # -1.0, -1.0, -1.0,
                0.0,
                0.0,
                -0.5, -0.5, -0.5, 0.7
                ]], device=device)
            self.obs_ub = torch.tensor([[ 
                1.0,  1.0,  1.0, 1.0,
                # 1.0, 1.0, 1.0,
                0.2,
                0.2,
                0.5, 0.5, 0.5, 1.0
                ]], device=device)
        elif repr_type == "surface_all_fingertips":
            # 4 fingertips × [u_hat(3 in [-1,1]), r_log_norm(1 in [0,1])]
            lb_unit = torch.cat([       
                torch.tensor([-1.0, -1.0, -1.0, 0.0], device=device).repeat(4),
                torch.tensor([0.0], device=device).repeat(16)
            ], dim=-1)
            ub_unit = torch.cat([
                torch.tensor([1.0, 1.0, 1.0, 1.0], device=device).repeat(4),
                torch.tensor([0.2], device=device).repeat(16)
            ], dim=-1)
            self.obs_lb = lb_unit.view(1, 32)
            self.obs_ub = ub_unit.view(1, 32)
        else:
            self.obs_lb = torch.zeros(1, cfg["obs_dim"], device=device)
            self.obs_ub = torch.ones(1, cfg["obs_dim"], device=device)

        self.obs_lb = torch.tensor([[
            -0.15, -0.15, -0.15, -0.15,
            -0.15, -0.15, -0.15, -0.15,
            -0.15, -0.15, -0.15, -0.15,
            ]], device=device)
        self.obs_ub = torch.tensor([[ 
            0.15, 0.15, 0.15, 0.15,
            0.15, 0.15, 0.15, 0.15,
            0.15, 0.15, 0.15, 0.15,
            ]], device=device)
    def bin2int(self, bins):
        base_ = 2 ** torch.arange(end=self.cfg["pred_dim"], device=self.device).unsqueeze(0)
        ints = torch.sum(bins * base_, dim=-1)
        ints = torch.clamp(ints, min=0, max=2**self.cfg["pred_dim"]-1)
        return ints
    
    def normalize(self, obs):
        """Normalize observations using trigonometric encoding"""
        obs_ = torch.clamp(obs, min=self.obs_lb, max=self.obs_ub)
        obs_ = (obs_ - self.obs_lb) / (self.obs_ub - self.obs_lb) * math.pi
        obs_ = torch.cat([torch.cos(obs_), torch.sin(obs_)], dim=-1)
        return obs_
    
    def update_curiosity(self, obs, exploration_bonus_scale=1.0):
        """Update curiosity and return exploration bonus"""
        

        obs_norm = self.normalize(obs)
        hash_codes = self.hashnn(obs_norm) # boolean (N, pred_dim)
        bin_indices = self.bin2int(hash_codes)
        
        counts = torch.bincount(bin_indices, minlength=self.bin_cnt.numel())
        self.bin_cnt += counts
        
        curiosity_reward = 1.0 / torch.sqrt(1.0 + self.bin_cnt[bin_indices])
        
        return curiosity_reward.detach()

class FibonacciCuriosity:
    def __init__(self, cfg, device, num_envs, num_fib_points=128):
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.num_fib_points = int(num_fib_points)

        # equal-area Fibonacci lattice on S^2
        self.fib_points = self._generate_optimal_fibonacci_lattice(self.num_fib_points).to(device)

        self.per_finger = bool(cfg.get("per_finger", True))
        self.num_fingers = int(cfg.get("num_fingers", 4))
        if self.per_finger:
            self.bin_cnt = torch.zeros(self.num_fingers, self.num_fib_points, dtype=torch.long, device=device, requires_grad=False)
        else:
            self.bin_cnt = torch.zeros(self.num_fib_points, dtype=torch.long, device=device, requires_grad=False)

        self.obs_lb = torch.tensor([[
            -1.0, -1.0, -1.0,  # normal dir
        ]], device=device)
        self.obs_ub = torch.tensor([[
             1.0,  1.0,  1.0,
        ]], device=device)

    def _generate_optimal_fibonacci_lattice(self, n: int) -> torch.Tensor:
        golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
        i = torch.arange(0, n, dtype=torch.float32)

        # epsilon per the article (piecewise)
        if n >= 600000:
            epsilon = 214.0
        elif n >= 400000:
            epsilon = 75.0
        elif n >= 11000:
            epsilon = 27.0
        elif n >= 890:
            epsilon = 10.0
        elif n >= 177:
            epsilon = 3.33
        elif n >= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33

        theta = 2.0 * torch.pi * i / golden_ratio
        phi = torch.acos(1.0 - 2.0 * (i + epsilon) / (n - 1.0 + 2.0 * epsilon))

        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)
        return torch.stack([x, y, z], dim=1)  # (n,3)

    def _closest_indices(self, normals: torch.Tensor) -> torch.Tensor:
        # normals: (B,3) → unit
        normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
        sims = normals @ self.fib_points.T  # (B,N)
        return torch.argmax(sims, dim=-1)    # (B,)

    @torch.no_grad()
    def update_curiosity(
        self,
        obs: torch.Tensor,                         # (N, F, 3)
        exploration_bonus_scale: float = 1.0,
        contact_mask: Optional[torch.Tensor] = None # NOTE: zero obs works the same as no contact
    ) -> torch.Tensor:
        assert obs.dim() == 3 and obs.size(-1) == 3, f"obs must be (N, F, 3), got {tuple(obs.shape)}"
        N, F, _ = obs.shape

        norms = torch.norm(obs, dim=-1, keepdim=True)                     # (N, F, 1)
        nonzero_mask = (norms.squeeze(-1) > 1e-8)                         # (N, F)
        if contact_mask is None:
            valid = nonzero_mask
        else:
            valid = nonzero_mask & contact_mask.bool()                    # (N, F)

        if not valid.any():
            return torch.zeros(N, device=self.device)

        dirs = torch.zeros_like(obs)                                      # (N, F, 3)
        dirs[valid] = (obs[valid] / norms[valid])
        flat_dirs = dirs.view(-1, 3)                                      # (N*F, 3)
        flat_valid = valid.view(-1)                                       # (N*F,)
        sel_dirs = flat_dirs[flat_valid]                                  # (M, 3), M = number of valid contacts

        idx = self._closest_indices(sel_dirs)                             # (M,)

        # per-finger or global counting
        per_contact_counts = torch.zeros_like(idx, dtype=torch.long, device=self.device)  # (M,)

        if self.per_finger:
            # derive per-contact finger ids from column index
            finger_ids_full = torch.arange(F, device=self.device).unsqueeze(0).expand(N, F).contiguous().view(-1)
            sel_finger_ids = finger_ids_full[flat_valid]                  # (M,)

            # update per-finger bins and read counts back
            num_f_used = min(self.num_fingers, F)
            for f in range(num_f_used):
                mask_f = (sel_finger_ids == f)
                if mask_f.any():
                    idx_f = idx[mask_f]
                    binc = torch.bincount(idx_f, minlength=self.num_fib_points)
                    self.bin_cnt[f, :] += binc
                    per_contact_counts[mask_f] = self.bin_cnt[f, idx_f]
        else:
            # global update
            binc = torch.bincount(idx, minlength=self.num_fib_points)
            self.bin_cnt += binc
            per_contact_counts[:] = self.bin_cnt[idx]

        alpha, beta = 4.0, 4.0
        per_contact_rew = alpha / torch.sqrt(1.0 + beta * per_contact_counts.float())  # (M,)

        rewards_matrix = torch.zeros(N * F, device=self.device)
        rewards_matrix[flat_valid] = per_contact_rew
        rewards_matrix = rewards_matrix.view(N, F)

        per_env_reward = (rewards_matrix * valid.float()).sum(dim=1) / self.num_fingers

        return (exploration_bonus_scale * per_env_reward).detach()

def build_curiosity(cfg, device, num_envs):
    """
    Factory: cfg['mechanism']: 'fibonacci' | 'neural_hash' (default)
    - For fibonacci: cfg keys:
        - num_fib_points (int, default 128)
        - per_finger (bool, default False)
        - num_fingers (int, default 4)
    """
    mech = str(cfg.get("mechanism", "neural_hash")).lower()
    if mech == "fibonacci":
        return FibonacciCuriosity(cfg, device, num_envs, num_fib_points=cfg.get("num_fib_points", 128))
    return NeuralHashCuriosity(cfg, device, num_envs)