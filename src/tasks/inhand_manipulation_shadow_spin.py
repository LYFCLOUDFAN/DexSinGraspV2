# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from .inhand_manipulation_shadow import InhandManipulationShadow

import math
from .torch_utils import *
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import json
from .isaacgym_utils import (
    ObservationSpec,
    ActionSpec,
)


class InhandManipulationShadowSpin(InhandManipulationShadow):
    _asset_root: str = "/home/nus/IsaacGymEnvs/assets"
    
    _observation_specs: Sequence[ObservationSpec] = []
    _action_specs: Sequence[ActionSpec] = []
    _shadow_hand_center_prim: str = "palm_link"
    _fingertips: List[str] = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:thdistal"] # index, middle, ring, thumb
    _keypoints: List[str] = [
        # 'robot0:palm', 
        # 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
        # 'robot0:mfproximal', 'robot0:mfmiddle', 'robot0:mfdistal', 
        # 'robot0:rfproximal', 'robot0:rfmiddle', 'robot0:rfdistal', 
        # 'robot0:lfmetacarpal', 'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal', 
        # 'robot0:thproximal', 'robot0:thmiddle', 'robot0:thdistal'
        'robot0:ffproximal', 'robot0:ffmiddle',
        'robot0:mfproximal', 'robot0:mfmiddle',
        'robot0:rfproximal', 'robot0:rfmiddle',
        'robot0:thproximal', 'robot0:thmiddle'
        
    ]
    _keypoints_info_path: str = "assets/urdf/robot_shadow_hand_keypoints.json" # XXX: one keypointper link for now.
    _keypoints_info: Dict[str, List[List[float]]] = json.load(open(_keypoints_info_path, "r"))

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        # Addition to InhandManipulationShadow
        self.goal_states[:, 3] = -0.8
        self.goal_init_state = self.goal_states.clone()     
    
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        
        self.task_reward = self.rew_buf.clone()
        self.extras["task_reward"] = self.task_reward.clone()
        self.compute_reach_reward_keypoints(); 
        self.reach_rew_scaled = self.reach_rew_scaled_keypoints.clone()
        self.rew_buf[:] += self.reach_rew_scaled
        
        self.curiosity_reward = self.compute_curiosity_reward()
        self.rew_buf[:] += self.curiosity_reward

        # self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
        
        self.extras["success_num"] = torch.sum(self.successes>0).unsqueeze(-1).clone()

    def reset_target_pose(self, env_ids, apply_reset=False):
        current_rotations = self.goal_states[env_ids, 3:7] # Replace this line with the actual tensor

        angle_degree = 10
        angle_radian = np.deg2rad(angle_degree)

        # Assuming you have as many rotations as len(env_ids)
        z_unit_tensor = torch.tensor([[0.0, 1.0, 0.0]] * len(env_ids), device=current_rotations.device)
        rotation_increment = quat_from_angle_axis(torch.tensor([angle_radian]*len(env_ids), device=current_rotations.device), z_unit_tensor)

        # Update the rotations
        new_rot = quat_mul(current_rotations, rotation_increment)

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot.float()
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

 
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

#####################################################################
###=========================jit functions=========================###
#####################################################################

import math
@torch.jit.script
def flip_orientation(current_rotations: torch.Tensor):
    # Flips the pen upside down first, then applies the current rotation
    num_envs = current_rotations.shape[0]
    z_unit_tensor = torch.tensor([[0.0, 1.0, 0.0]] * num_envs, device=current_rotations.device)
    flip_rot = quat_from_angle_axis(torch.tensor([math.pi] * num_envs, device=current_rotations.device), z_unit_tensor)
    return quat_mul(current_rotations, flip_rot)

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance
    
    # Orientation alignment 
    # Modified so pen is symmetrical; since we only rotate around the z axis,
    quat_diff_1 = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist_1 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff_1[:, 0:3], p=2, dim=-1), max=1.0))
    quat_diff_2 = quat_mul(object_rot, quat_conjugate(flip_orientation(target_rot)))
    rot_dist_2 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff_2[:, 0:3], p=2, dim=-1), max=1.0))
    rot_dist = torch.min(rot_dist_1, rot_dist_2)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    # reward = cons_successes
    return reward, resets, goal_resets, progress_buf, successes, cons_successes
