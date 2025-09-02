#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

python src/train.py \
    headless=False \
    env_mode=pgm \
    env_info=True \
    num_envs=24 \
    num_objects=1 \
    num_objects_per_env=1 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=XArmAllegroHandCubeInBox \
    train=XArmAllegroHandCubeInBoxPPO \
    task.env.enableContactSensors=False \
    reward_type=succrew+tilt+slide+neighbor+actionpen \
    sim_device=cpu \
    rl_device=cpu \
    physics_engine=physx \
    --seed=42 \
    --exp_name='PPO' \
    --logdir='' \
    --run_device_id=-1 \
    --web_visualizer_port=-1 \
    --print_log=True 
    # --resume_iter=1000 \
    # --model_dir="logs/PPO/08-16-18-05_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42"
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # reward_type=succrew+tilt+slide+neighbor+stability+actionpen \
