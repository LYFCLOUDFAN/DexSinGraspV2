#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

python src/train.py \
    headless=False \
    env_mode=pgm \
    env_info=True \
    num_envs=4096 \
    num_objects=5 \
    num_objects_per_env=5 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=XArmAllegroHandFunctionalManipulationUnderarm \
    train=XArmAllegroHandFunctionalManipulationUnderarmPPO \
    task.env.enableContactSensors=False \
    reward_type=succrew+tilt+slide+neighbor+actionpen \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    --seed=42 \
    --exp_name='PPO' \
    --logdir='xarm_allegro_singulation_ppo' \
    --model_dir='logs/PPO/07-24-03-48_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed0/model_18000.pt' \
    --run_device_id=0 \
    --run_device_id=0 \
    --web_visualizer_port=-1
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # reward_type=succrew+tilt+slide+neighbor+stability+actionpen \
