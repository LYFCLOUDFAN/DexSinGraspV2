# <p align="center"> DexSinGrasp: Learning a Unified Policy for Dexterous Object Singulation and Grasping in Cluttered Environments </p>
### <p align="center"> [Arxiv](https://arxiv.org/abs/2504.04516) | [Website](https://nus-lins-lab.github.io/dexsingweb/) </p>

# I. Installation

Make a project directory, inside which clone this repo
```
git clone https://github.com/DavidLXu/DexSinGrasp.git
```
The project directory should have a structure like this
```
PROJECT
    └── Logs
        └── Results
            └── results_train
            └── results_distill
            └── results_trajectory
    └── Assets
        └── meshdatav3_pc_fps
        └── meshdatav3_scaled
        └── textures
        └── urdf
    └── DexSinGrasp
        └── results
        └── dexgrasp
```
Download [Assets.zip](https://drive.google.com/file/d/1wMcJFjTKa5bFGTdD4JgIJ9o8eB2sgSd8/view?usp=sharing) and unzip to `PROJECT/Assets`

Download [random_arrangements.zip](https://drive.google.com/file/d/1vFf_4PwfW6BUaI3Q71vEws2a-jQvYP0R/view?usp=sharing) and unzip to `PROJECT/DexSinGrasp/dexgrasp/random_arrangements`

(Optional) Download pretrained [teacher models](https://drive.google.com/drive/folders/18n6XpfUP6Z80IDsSK_2RslAjtRWX7_mm?usp=sharing) and [student model](https://drive.google.com/drive/folders/1zJpoEV0fOZG4m3I3d-U3wQ3YjBF4ZK71?usp=sharing) for testing.

## Install Environment:
Python 3.8 is required.

```
conda create -n dexgrasp python=3.8
conda activate dexgrasp
```

Install IsaacGym:
Download [isaacgym](https://developer.nvidia.com/isaac-gym) first.
```
cd path_to_issacgym/python
pip install -e .
```

Install DexSinGrasp:
```
cd PROJECT/DexSinGrasp
bash install.sh
```
when doing this step, you may encounter some issues with pytorch3d
```
The detected CUDA version (12.1) mismatches the version that was used to compile
PyTorch (11.7). Please make sure to use the same CUDA versions.
```
Install CUDA 11.7 in your custom dir and set the environment variable.


Install other dependencies:
```
cd pytorch_kinematics-default
pip install -e .
```


# II. Teacher policies
## Step1: Train Teacher Policy:
Enter the working directory
```
cd PROJECT/DexSinGrasp/dexgrasp/
```
Start by training a single object grasping task by setting `surrounding_obj_num` to `0`.
```
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 2 --surrounding_obj_num 0
```
The saved weights are in `PROJECT/Logs/Results/results_train/`

**Curriculums**

Based on the Single Object Grasping Policy, we train the curriculum in a sequence of `D-4,D-6,D-8` followed by `R-4,R-6,R-8`, where `D` and `R` stand for dense and random arrangements respectively, and the number stands for the quantity of surrounding objects.

For dense arrangement environment, we use `--expert_id 2`. 
```bash
# Training D-4 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 2 --surrounding_obj_num 4 --model_dir path_to_single_obj_ckpt.pt
```
```bash
# Training D-6 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 2 --surrounding_obj_num 6 --model_dir path_to_D4_ckpt.pt
```
```bash
# Training D-8 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 2 --surrounding_obj_num 8 --model_dir path_to_D6_ckpt.pt
```
The resulting D-8 expert is used for data collection in the distillation phase.

For random arrangement environment, we use `--expert_id 3`. (Note that `--expert_id 1` is deprecated.)
```bash
# Training R-4 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 3 --surrounding_obj_num 4 --model_dir path_to_D8_ckpt.pt
```
```bash
# Training R-6 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 3 --surrounding_obj_num 6 --model_dir path_to_R4_ckpt.pt
```
```bash
# Training R-8 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --expert_id 3 --surrounding_obj_num 8 --model_dir path_to_R6_ckpt.pt
```
The resulting R8 expert is used for data collection in the distillation phase.

For pretrained D8/R8 teacher policy, you can download [here](https://drive.google.com/drive/folders/18n6XpfUP6Z80IDsSK_2RslAjtRWX7_mm?usp=sharing).

## Step2: Test Teacher Policy
For the `D/R-n` policy, we test 10 envs for 10 episodes.

```bash
# Testing D-8 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 --config dedicated_policy.yaml --expert_id 2 --surrounding_obj_num 8 --num_envs 10  --test --test_iteration 10 --model_dir path_to_D8_ckpt.pt  
```
```bash
# Testing R-8 Policy
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 --config dedicated_policy.yaml --expert_id 3 --surrounding_obj_num 8 --num_envs 10  --test --test_iteration 10 --model_dir path_to_R8_ckpt.pt  
```

# III. Policy Distillation
## Step3: Data Collection for Distillation
For example, to collect 200 episodes of dense 8 surrounding objects
```bash
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0  \
 --num_envs 50 --max_iterations 10000 --config dedicated_policy.yaml --test --test_iteration 4  \
 --save --save_train --save_render --model_dir path_to_D8_teacher.pt \
 --save_camera True --table_dim_z 0.6 --expert_id 2 --surrounding_obj_num 8
```
Refer to `run_collection.sh` for details to collect trajectories in various environments. You can download collected data [here]() to skip this part.

NOTE: 
1. DO NOT pause the visualization of isaacgym during data collection, otherwise the collected pointcloud will be static.
2. We set `--table_dim_z` to be 0.6 to avoid a bug casuing PCA errors. In this case, raw pointclouds are collected with lowest height 0.6 (normalized during training). Other states in the observation space are invariant with table heights.

Dynamic visualization of pointclouds
```
python dexgrasp/pointcloud_vis_pkl_dyn.py
```


## Step4: Vision-based Distillation
Train distilled vision-based policy
```bash
python run_offline.py --config universal_policy_vision_based.yaml --device cuda:0
```
The checkpoints are saved into `PROJECT/Logs/Results/results_distill/random/universal_policy/distill_student_model`

You can modify `config['Offlines']['train_epochs']` and `config['Offlines']['train_batchs']` in `run_offline.py` to change the epochs and batch size.

You can also download the pretrained student model [here](https://drive.google.com/drive/folders/1zJpoEV0fOZG4m3I3d-U3wQ3YjBF4ZK71?usp=sharing) and save it as `PROJECT/Logs/Results/results_distill/random/universal_policy/distill_student_model/model_best.pt`

Test distilled vision-based policy on dense arrangements.
```bash
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 --num_envs 10 --config universal_policy_vision_based.yaml --test --test_iteration 10 --model_dir distill_student_model --save_camera True --table_dim_z 0.6 --expert_id 2 --surrounding_obj_num 8
```
Test distilled vision-based policy on random arrangements.
```bash
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 --num_envs 10 --config universal_policy_vision_based.yaml --test --test_iteration 10 --model_dir distill_student_model --save_camera True --table_dim_z 0.6 --expert_id 3 --surrounding_obj_num 8
```

## Extra: Define new task environment
1. In `dexgrasp/state_based_grasp_customed.py` write `class StateBasedGraspCustomed`
2. In `dexgrasp/utils/config.py`
```
def retrieve_cfg(args, use_rlg_config=False):
    # add
    elif args.task == "StateBasedGraspCustomed":
        return os.path.join(args.logdir, "state_based_grasp_customed/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/state_based_grasp_customed.yaml"
```
3. Add `dexgrasp/cfg/state_based_grasp_customed.yaml`, copy from existing file, but some content is not used and will be overwritten by other codes for now. Remember to change the `env_name` to python task file name.
4. Add in `dexgrasp/utils/parse_task.py`
```
from tasks.state_based_grasp_customed import StateBasedGraspCustomed
```
5. Call the task class from here
```
elif args.task_type == "Python":
    try:
        # ... previous if and elif ...
        elif cfg['env']['env_name'] == "state_based_grasp_customed":
            task = StateBasedGraspCustomed(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                is_multi_agent=False)
```

6. Train the customed task:
```
python run_online.py --task StateBasedGraspCustomed --algo ppo --seed 0 --rl_device cuda:0 --num_envs 1000 --max_iterations 10000 --config dedicated_policy_customed.yaml
```

# Acknowledgements

This project builds upon and extends the work from [UniGraspTransformer](https://dexhand.github.io/UniGraspTransformer/) and [UniDexGrasp++](https://github.com/PKU-EPIC/UniDexGrasp2). We gratefully acknowledge the contributions of the researchers and developers behind these projects.