# train distillation
python run_offline.py --config universal_policy_vision_based.yaml --device cuda:0

# test distillation
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 \
    --num_envs 10 --config universal_policy_vision_based.yaml --test --test_iteration 10 \
    --model_dir distill_student_seed0_0405 --save_camera True --table_dim_z 0.6 --expert_id 3 --surrounding_obj_num 8