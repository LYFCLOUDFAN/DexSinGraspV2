import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
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
                # -1.0, -1.0, -1.0, 0.0,
                -1.0, -1.0, -1.0,
                0.0,
                0.0,
                -0.5, -0.5, -0.5, 0.7
                ]], device=device)
            self.obs_ub = torch.tensor([[ 
                # 1.0,  1.0,  1.0, 1.0,
                1.0, 1.0, 1.0,
                0.1,
                0.1,
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