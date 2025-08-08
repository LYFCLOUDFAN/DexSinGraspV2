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
    reward_type=succrew+tilt+slide+neighbor+actionpen \
    physics_engine=physx \
    --vis_env_num=2 \
    --seed=42 \
    --exp_name='PPO' \
    --test \
    --eval_times=5 \
    --logdir="" \
    --run_device_id=-1 \
    --web_visualizer_port=-1 \
    --resume_iter=29000 \
    --model_dir="/home/ruoyi/Work/UniDexFPM/logs/PPO/08-07-04-07_xarm_allegro_singulation_ppo_safety_finetune_5e-1_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42"
