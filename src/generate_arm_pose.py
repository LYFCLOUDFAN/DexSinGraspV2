# must be the first lines
import sys, argparse
_known = argparse.ArgumentParser(add_help=False)
_known.add_argument("--num_poses", type=int, default=1000)
_known.add_argument("--output", type=str, default="data/precomputed_arm_poses.pt")
_known_args, _remaining = _known.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

import os
import sys
from pathlib import Path

# Isaac Gym must be imported before torch
import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch  # noqa: F401

import torch

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from tasks import load_isaacgym_env
from utils.config import get_args, load_cfg


def _ensure_task_overrides(args):
    # Ensure task name override exists
    if not any([ov.startswith("task=") for ov in args.overrides]):
        args.overrides.append("task=XArmAllegroHandFunctionalManipulationUnderarm")
    # Default headless for speed if not set
    if not any([ov.startswith("headless=") for ov in args.overrides]):
        args.overrides.append("headless=True")


def _wire_obs_act_spaces(args, cfg_train):
    # Mirror train.py logic for obs/action spaces; only pgm + XArm branch is needed here
    if "env_mode=pgm" in args.overrides and "task=XArmAllegroHandFunctionalManipulationUnderarm" in " ".join(args.overrides):
        obs_space = [
            "xarm_endeffector_position",
            "xarm_endeffector_orientation",
            "xarm_endeffector_linear_velocity",
            "xarm_endeffector_angular_velocity",
            "allegro_hand_dof_position",
            "allegro_hand_dof_velocity",
            "fingertip_position_wrt_palm",
            "fingertip_orientation_wrt_palm",
            "fingertip_linear_velocity",
            "fingertip_angular_velocity",
            "object_position_wrt_palm",
            "object_orientation_wrt_palm",
            "object_position",
            "object_orientation",
            "object_linear_velocity",
            "object_angular_velocity",
            "nearest_non_target_object_position",
            "nearest_non_target_object_orientation",
            "object_bbox",
            "tactile",
        ]
        action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
    else:
        # Fallback: keep small action space to move wrist only
        obs_space = ["xarm_endeffector_position", "xarm_endeffector_orientation", "allegro_hand_dof_position"]
        action_space = ["wrist_translation", "wrist_rotation"]

    # Device overrides
    sim_device = next((ov.split("=", 1)[1] for ov in args.overrides if ov.startswith("sim_device=")), "cuda:0")
    rl_device = next((ov.split("=", 1)[1] for ov in args.overrides if ov.startswith("rl_device=")), "cuda:0")

    args.overrides.append(f"sim_device={sim_device}")
    args.overrides.append(f"rl_device={rl_device}")
    args.overrides.append(f"obs_space={obs_space}")
    args.overrides.append(f"action_space={action_space}")


def main(num_poses: int, output_path: str):
    # Hydra-style args and cfg (same as train.py)
    args = get_args()
    cfg_train, _ = load_cfg(args)

    _ensure_task_overrides(args)
    _wire_obs_act_spaces(args, cfg_train)

    # Build env via the same loader used in train.py
    env, _ = load_isaacgym_env(task_name="", args=args)

    # How many envs per reset
    num_envs = getattr(env, "num_envs", 1)
    device = env.device if hasattr(env, "device") else "cuda:0"

    # Preallocate storage
    poses = []
    metas = []

    needed = num_poses
    print(f"Collecting {num_poses} poses in batches of {num_envs}...")

    while needed > 0:
        env.reset()
        # sample_initial_pose is called inside reset when randomInitAroundObject is enabled.
        # If you prefer forcing it: env.sample_initial_pose(torch.arange(min(num_envs, needed), device=device))

        batch_n = min(num_envs, needed)
        env_ids = torch.arange(batch_n, device=device)

        # Read joint positions
        joint_q = env.allegro_hand_dof_positions[env_ids].detach().cpu()  # (B, num_joints)
        poses.append(joint_q)

        # Optional metadata
        metas.append({
            "object_center": env.init_scene_object_root_positions.mean(dim=1)[env_ids].detach().cpu(),
            "palm_position": env.allegro_hand_center_positions[env_ids].detach().cpu(),
            "eef_position": env.endeffector_positions[env_ids].detach().cpu(),
            "eef_orientation": env.endeffector_orientations[env_ids].detach().cpu(),
        })

        needed -= batch_n

    poses = torch.cat(poses, dim=0)[:num_poses]
    # Flatten simple metadata for the first N entries
    meta_list = []
    filled = 0
    for m in metas:
        b = m["object_center"].shape[0]
        for i in range(b):
            if filled >= num_poses:
                break
            meta_list.append({
                "object_center": m["object_center"][i],
                "palm_position": m["palm_position"][i],
                "eef_position": m["eef_position"][i],
                "eef_orientation": m["eef_orientation"][i],
            })
            filled += 1
        if filled >= num_poses:
            break

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "poses": poses,  # (N, num_joints)
        "metadata": meta_list,  # list of dicts with tensors
        "num_poses": poses.shape[0],
    }, output_path)

    print(f"Saved {poses.shape[0]} poses to {output_path}")


if __name__ == "__main__":
    main(_known_args.num_poses, _known_args.output)