#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

python src/train.py \
    headless=False \
    env_mode=pgm \
    env_info=True \
    num_envs=4096 \
    num_objects=1 \
    num_objects_per_env=1 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=XArmAllegroHandCubeInBox \
    train=XArmAllegroHandCubeInBoxPPO \
    task.env.enableContactSensors=False \
    reward_type=succrew+tilt+slide+neighbor+actionpen \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    --cfg_train=XArmAllegroHandCubeInBoxPPO \
    --seed=0 \
    --exp_name='PPO' \
    --logdir='cube_in_box_singulation_ppo_use_random_pre_pose_w_curiosity' \
    --run_device_id=0 \
    --run_device_id=0 \
    --web_visualizer_port=-1
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # reward_type=succrew+tilt+slide+neighbor+stability+actionpen \
