# train from scratch
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations  10000 --config dedicated_policy.yaml 
# train from checkpoint
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations  10000 --config dedicated_policy.yaml  --model_dir xxx 

# test checpoint
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 50 --max_iterations  10000 --config dedicated_policy.yaml --test --model_dir xxx

# save trajectory and vision pointcloud (100 envs)
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 100 --max_iterations  10000 --config dedicated_policy.yaml --test --test_iteration 10 --model_dir /data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_0219_stage_5_eight_dense_0.08_succ/model_10000.pt --save --save_train --save_render

python run_online.py --task StateBasedGraspLeapDummy --algo ppo --seed 0 --rl_device cuda:0  --num_envs 100 --max_iterations  10000 --config dedicated_policy_leap_dummy.yaml --test --test_iteration 10 --model_dir '/data/UniGraspTransformer/Logs/Results/results_train/0525_seed0_dummy_pos_succ_0.075_dist_best/model_10000.pt' --save --save_train --save_render
