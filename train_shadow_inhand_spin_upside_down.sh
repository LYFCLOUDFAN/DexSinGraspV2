#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

python src/train.py \
    headless=False \
    env_mode=pgm \
    env_info=True \
    num_envs=4096 \
    num_objects=5 \
    num_objects_per_env=1 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=InhandManipulationShadowSpinUpsideDown \
    train=InhandManipulationShadowSpinUpsideDownPPO \
    reward_type=task+curiosity \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    --cfg_train=InhandManipulationShadowSpinUpsideDownPPO \
    --seed=42 \
    --exp_name='PPO' \
    --logdir='shadow_inhand_spin_upside_down_ppo' \
    --run_device_id=0 \
    --web_visualizer_port=-1 \
    # --con \
    # --model_dir=/home/nus/DexSinGraspV2/logs/PPO/08-24-21-31_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:2048_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42 \
    # --model_dir=/home/nus/DexSinGraspV2/logs/PPO/08-12-17-05_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:2048_rewtype:succrew+tilt+slide+neighbor+actionpen+curiosity_seed42 \
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # reward_type=succrew+tilt+slide+neighbor+stability+actionpen \
