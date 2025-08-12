#!/bin/bash

python src/train.py \
    headless=False \
    env_mode=pgm \
    mode=eval \
    env_info=True \
    num_envs=4096 \
    num_objects=5 \
    num_objects_per_env=5 \
    graphics_device_id=0 \
    split='train' \
    cluster=$CLUSTER \
    task=XArmAllegroHandFunctionalManipulationUnderarm \
    train=XArmAllegroHandFunctionalManipulationUnderarmPPO \
    task.env.enableContactSensors=False \
    task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
    task.env.datasetPoseLevelSampling=True \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    reward_type=succrew+tilt+slide+neighbor+actionpen \
    physics_engine=physx \
    --seed=42 \
    --exp_name='PPO' \
    --test \
    --eval_times=2 \
    --logdir="" \
    --run_device_id=-1 \
    --web_visualizer_port=-1 \
    --resume_iter=1000 \
    --model_dir="logs/PPO/08-13-00-38_xarm_allegro_singulation_ppo_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42"
