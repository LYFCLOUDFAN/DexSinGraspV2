#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

python src/train.py \
    headless=False \
    env_mode=pgm \
    env_info=True \
    num_envs=24 \
    num_objects=5 \
    num_objects_per_env=5 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=XArmAllegroHandFunctionalManipulationUnderarm \
    train=XArmAllegroHandFunctionalManipulationUnderarmPPO \
    task.env.enableContactSensors=False \
    reward_type=succrew+tilt+slide+neighbor+actionpen+contact_curiosity \
    sim_device=cpu \
    rl_device=cpu \
    physics_engine=physx \
    task.env.pclObs=True \
    --seed=0 \
    --exp_name='PPO' \
    --logdir='' \
    --run_device_id=-1 \
    --web_visualizer_port=-1 \
    --print_log=True
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # reward_type=succrew+tilt+slide+neighbor+stability+actionpen \
