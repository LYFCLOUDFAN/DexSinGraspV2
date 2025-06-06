# D-8 Expert Collection
#! 200 episodes on 8 objects
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  \
 --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
 --save --save_train --save_render --model_dir path_to_D8_teacher.pt \
 --save_camera True --table_dim_z 0.6 --expert_id 2 --surrounding_obj_num 8

#! 200 episodes on 6 objects
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  \
 --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
 --save --save_train --save_render --model_dir path_to_D8_teacher.pt \
 --save_camera True --table_dim_z 0.6 --expert_id 2 --surrounding_obj_num 6

#! 100 episodes on 4 objects
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  \
 --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
 --save --save_train --save_render --model_dir path_to_D8_teacher.pt \
 --save_camera True --table_dim_z 0.6 --expert_id 2 --surrounding_obj_num 4

# R-8 Expert needs to collect randomized arrangements, set shuffle_object_arrangements to True in yaml
#! 200 episodes on 8 objects
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 \
  --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
   --save --save_train --save_render --model_dir path_to_R8_teacher.pt \
    --save_camera True --table_dim_z 0.6 --expert_id 3 --surrounding_obj_num 8


#! 200 episodes on 6 objects
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 \
  --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
   --save --save_train --save_render --model_dir path_to_R8_teacher.pt \
    --save_camera True --table_dim_z 0.6 --expert_id 3 --surrounding_obj_num 6


#! 100 episodes on 4 objects
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 \
  --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
   --save --save_train --save_render --model_dir path_to_R8_teacher.pt \
    --save_camera True --table_dim_z 0.6 --expert_id 3 --surrounding_obj_num 4
