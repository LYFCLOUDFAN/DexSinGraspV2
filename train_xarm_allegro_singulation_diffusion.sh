#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task with Diffusion
# This script runs diffusion training for the singulation and grasp task

python src/run.py \
    headless=False \
    env_mode=pgm \
    env_info=False \
    num_envs=6968 \
    num_objects=25 \
    num_objects_per_env=25 \
    graphics_device_id=-1 \
    split="train" \
    task=XArmAllegroHandFunctionalManipulationUnderarm \
    train=XArmAllegroHandFunctionalManipulationUnderarmDiffusion \
    train.policy.network.n_layer=4 \
    train.policy.network.encode_state_type="arm+dof+obj2palmpose+target" \
    train.learn.dataset.data_dir="data/expert_dataset_pose_level_specialist/memmap" \
    +train.learn.dataset.max_num_trajectories_per_target=20 \
    +train.learn.dataset.num_repeats=1 \
    train.learn.dataloader.batch_size=1024 \
    train.policy.optimizer.learning_rate=0.0002 \
    train.learn.num_epochs=501 \
    stack_frame_number=2 \
    --algorithm="diffusion" \
    --mode="train" \
    --exp_name="xarm_allegro_singulation_diffusion" \
    --device_id=0 \
    --num_evaluation_rounds=1 \
    --print_freq=20 