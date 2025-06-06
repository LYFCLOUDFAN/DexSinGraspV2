import numpy as np

import argparse
import sys
import os
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import process_ppo
import torch
import time
def run_inference(model_path,seed):
    """
    Run inference with a pretrained model
    Args:
        model_path: Path to the pretrained model checkpoint
        goal_position: [x, y, z] coordinates for goal position. If None, uses random goals
    """
    # Initialize args by passing our model path to sys.argv
    sys.argv = [
        'inference.py',
        '--task', 'StateBasedGrasp',
        '--algo', 'ppo',
        '--seed', str(seed),
        '--rl_device', 'cuda:0',
        '--num_envs', '1',  # Reduced for inference
        '--config', 'dedicated_policy.yaml',
        # '--headless',
        '--model_dir', model_path,
        '--task_type', 'Python',
    ]
    
    args, _ = get_args()
    args.test = True  # Set to test mode
    
    # Load configs
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    
    # Create environment and load policy
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]])
    
    args.model_dir = '/data/UniGraspTransformer/Logs/Results/results_train/model_3800_pure_singulation_ver2.pt'
    # '/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_expert2_obj8_wo_curriculum_failed/model_10000.pt'
    # 
    # '/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_0223_pure_singulation_3000_better/model_3000.pt'
    #'/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_0219_stage_0_pure_singulation/model_3300.pt'

    print(f"Singulation: Loading checkpoint from {args.model_dir}")
    ppo = process_ppo(args, env, cfg_train, logdir)


    args.model_dir = '/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_expert2_obj0/model_5000.pt'
    #'/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_expert2_obj0_2000/model_2000.pt' 
    # '/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_expert3_obj8_wo_curriculum/model_10000.pt'

    # 
    
    #'/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_0219_stage0_pure_grasping_0_3300/model_1300.pt'

    print(f"Lift: Loading checkpoint from {args.model_dir}")    
    ppo_lift = process_ppo(args, env, cfg_train, logdir)
    
    
    # Convert goal position to tensor if provided
    # goal pos
    # goal_position = np.array([0.4, 0.4, 0.0]) # relative to the center of the table
    # if goal_position is not None:
    #     # exit()
    #     # Calculate displacement from default position to desired goal
    #     # default_position = np.array([0.0, 0.0, 0.605])  # Default goal height
    #     # displacement = np.array(goal_position) - default_position
    #     # Set the goal displacement in the task

    #     task.set_goal_displacement(np.array(goal_position))
    
    # Run episodes
    num_episodes = 10
    successes = 0
    
    idx = 0
    iteration = 0
    prev_avg_distance = 0
    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset()
        
        # Run episode
        done = False
        episode_reward = 0
        
        while True:
            # Get action from policy
            with torch.no_grad():
                object_distances = task.get_object_distance()
                # Calculate average distance excluding zeros
                non_zero_distances = object_distances[object_distances > 0]
                avg_distance = non_zero_distances.mean() if len(non_zero_distances) > 0 else 0
                # print(f"Average non-zero distance: {avg_distance:.4f}")
                # time.sleep(0.1)
                
                # Check if all object distances are > 0.15
                # all_objects_separated = torch.all(object_distances > 0.15, dim=0)
                iteration += 1
                if avg_distance >= 0.16 and prev_avg_distance < 0.16:
                    print(f"Transition at iteration {iteration}: distance {avg_distance:.3f}")
                prev_avg_distance = avg_distance
                if avg_distance < 0.16:
                    actions, _ = ppo.actor_critic.act_inference(obs)
                else:
                    actions, _ = ppo_lift.actor_critic.act_inference(obs)
            idx += 1
            if idx == 300:
                print("successes",task.successes)    
            if idx == 301:
                print("steps",task.current_avg_steps) # being calculated after reset in the env
                iteration = 0
                # return task.successes.sum().item()
            # import pdb; pdb.set_trace()
            # Step environment
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
            
            # Check for success
            if 'successes' in info:
                successes += info['successes'].sum().item()
        
        print(f"Episode {episode + 1} reward: {episode_reward.mean().item():.2f}")
    
    success_rate = successes / (num_episodes * env.num_envs)
    print(f"\nOverall success rate: {success_rate:.2%}")

def train(args):
    """Run training with specified goal position and checkpoint initialization"""
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    # Load configs and setup environment
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    
    # Create environment and load policy
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    
    # Set goal position if provided
    # if goal_position is not None:
    #     task.set_goal_displacement(np.array(goal_position))
        
    # Initialize PPO trainer
    ppo = process_ppo(args, env, cfg_train, logdir)
    
    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}")
        ppo.load(args.checkpoint)
        
        # Extract starting iteration from checkpoint filename
        # Assumes filename format like "model_10000.pt"
        try:
            start_iteration = int(os.path.basename(args.checkpoint).split('_')[1].split('.')[0])
            print(f"Resuming from iteration {start_iteration}")
        except:
            print("Could not parse iteration number from checkpoint filename")
            start_iteration = 0
    else:
        start_iteration = 0
        
    # Set number of training iterations
    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations
        
    # Adjust remaining iterations if resuming from checkpoint
    remaining_iterations = iterations - start_iteration
        
    # Run training
    ppo.run(num_learning_iterations=remaining_iterations,
            log_interval=cfg_train["learn"]["save_interval"])


if __name__ == "__main__":
    # Create custom argument parser for our script
    custom_parser = argparse.ArgumentParser(description='Run inference with custom goal positions')
    custom_parser.add_argument("--model_dir", type=str, 
                             default="/data/UniGraspTransformer/Logs/Results/results_train/0525_seed0/model_6000.pt",
                             help="Path to pretrained model")
    custom_parser.add_argument("--task", type=str, default="StateBasedGrasp", help="Task name")
    custom_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    custom_args = custom_parser.parse_args()
    

    run_inference(custom_args.model_dir,seed=custom_args.seed)
