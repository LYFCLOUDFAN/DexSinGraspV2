#!/bin/bash

python src/train.py \
    headless=False \
    env_mode=pgm \
    mode=eval \
    env_info=True \
    num_envs=16 \
    num_objects=5 \
    num_objects_per_env=5 \
    graphics_device_id=0 \
    split='test' \
    cluster=$CLUSTER \
    task=XArmAllegroHandFunctionalManipulationUnderarm \
    train=XArmAllegroHandFunctionalManipulationUnderarmPPO \
    task.env.enableContactSensors=False \
    task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
    task.env.datasetPoseLevelSampling=True \
    sim_device=cpu \
    rl_device=cpu \
    physics_engine=physx \
    --vis_env_num=2 \
    --seed=0 \
    --exp_name='PPO' \
    --test \
    --eval_times=5 \
    --logdir="logs/PPO/07-15-21-03_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:16_rewtype:succrew+rotrew+actionpen+mutual+fjcontact_seed0/" \
    --run_device_id=-1 \
    --web_visualizer_port=-1 \
    --model_dir="/home/ruoyi/Work/UniDexFPM/logs/PPO/07-22-23-28_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed0/model_4000.pt"
