Adapted from [UniGraspTransformer](https://dexhand.github.io/UniGraspTransformer/). 




# <p align="center"> DexSingulation </p>



# I. Get Started

make a PROJECT folder outside this repo.
```
PROJECT
    └── Logs
        └── Results
            └── results_train
            └── results_distill
            └── results_trajectory
    └── Assets
        └── datasetv4.1_posedata.npy
        └── meshdatav3_pc_feat
        └── meshdatav3_scaled
        └── meshdatav3_init
        └── textures
        └── mjcf
        └── urdf
    └── isaacgym3
    └── isaacgym4
    └── DexSingulation
        └── results
        └── dexgrasp
```

## Install Environment:
Python 3.8 is required.

```
conda create -n dexgrasp python=3.8
conda activate dexgrasp
```

Install IsaacGym:
```
cd issacgym/python
pip install -e .
```

Install DexSingulation:
```
cd DexSingulation/dexgrasp
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

## Download Assets:
Download [meshdatav3_pc_feat.zip](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/meshdatav3_pc_feat.zip)
Download [meshdatav3_pc_fps.zip]()
Download simplified version of [meshdatav3_scaled.zip]()
Download simplified version of [meshdatav3_v3_init.zip]()
Download [datasetv4.1_posedata.npy](https://drive.google.com/file/d/1DajtOFyTPC5YhsO-Fd3Gv17x7eAysI1b/view)
Download [xarm6.zip](https://drive.google.com/file/d/1YmR7fin26oFazvx6lWe8H2jdxyrS77US/view?usp=sharing)



# II. Training
## Step1: Train Teacher Policy:

```
cd PROJECT/DexSingulation/dexgrasp/
```

```
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0     --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml
```

The saved weights are in `PROJECT/Logs/Results/results_train/`

## Curriculums
```
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0     --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --model_dir xxxxx.pt
```

## Step2: Inference:

```
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0     --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --model_dir xxxxx.pt --test --test_iteration 1
```

Test with different object sizes, numbers and arrangements



<!-- You can download the pretrained weights of StateBasedGraspLeapNoArm from [here](https://drive.google.com/file/d/1JweyfIJVXKxUMD485JSQiwsAdWUvx3jb/view?usp=drive_link), and save it as `PROJECT/Logs/Results/results_train/0525_seed0/model_30000.pt`. -->

<!-- ```
cd PROJECT/DexSingulation/dexgrasp/
python inference.py --task StateBasedGraspLeapNoArm --model_dir PROJECT/Logs/Results/results_train/0525_seed0/model_30000.pt 
``` -->

## Step3: Gather Trajectories
`StateBasedGrasp` env as example. Need to train an expert policy first.
Then, use ppo based rl expert policy to gather trajectories, including the vision features


```
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0       --num_envs 20  --config dedicated_policy.yaml --expert_id 2 --surrounding_obj_num 8  --model_dir '/data/UniGraspTransformer/Logs/Results/results_train/0000_seed0_expert2_obj8_0.06_succ0.75_for_distillation/model_5000.pt'  --test  --test_iteration 50  --save --save_train --save_render --save_camera True --table_dim_z 0.6
```
NOTE: DO NOTE pause the visualization of isaacgym during data collection, otherwise the collected pointcloud will be static.

Dynamic visualization of pointclouds
```
python dexgrasp/pointcloud_vis_pkl_dyn.py
```

## Existing bugs: 

1. --num_envs 10 greater than 10 will have the pointcloud trajectory static.
2. during training objects and hand would occasionally penetrate into tables
3. when gather vision trajectories with table_dim.z < 0.5 will have PCA errors

Solutions:
1. --num_envs 10 gather for 100 iterations = 1000 trajectories
2. when training, set table_dim.z 0.01
3. when testing, set table_dim.z back to 0.6

The pointclouds are collected with lowest height 0.6. other states are invariant with table sizes.

<!-- Remember to force set `self.use_dynamic_visual_feats` to `True` in `state_based_grasp.py`. Otherwise, the collected vision features will be zeros. -->

<!-- The collected trajectories are for both state-based and vision-based distillation. -->

<!-- ## Step3.1: State-based Distillation
```
python run_offline.py --start 525 --finish 525 --config universal_policy_state_based.yaml --object train_set_results.yaml --device cuda:0 -->
<!-- ```
You can modify `config['Offlines']['train_epochs']` and `config['Offlines']['train_batchs']` in `run_offline.py` to change the epochs and batch size.

## Step3.2: Test distilled state-based policy
```
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 --num_envs 10 --max_iterations 10000 --config universal_policy_state_based.yaml --test --test_iteration 1 --model_dir distill_0525_0525 --object_scale_file train_set_results.yaml --start_line 525 --end_line 526
``` -->

## Step4: Vision-based Distillation
Train distilled vision-based policy
```
python run_offline.py --config universal_policy_vision_based.yaml --object train_set_results.yaml --device cuda:0 --expert_id 2 --surrounding_obj_num 8
```
You can modify `config['Offlines']['train_epochs']` and `config['Offlines']['train_batchs']` in `run_offline.py` to change the epochs and batch size.

Test distilled vision-based policy
```
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 --num_envs 10 --config universal_policy_vision_based.yaml --test --test_iteration 1 --model_dir distill_expert2_obj4 --save_camera True --table_dim_z 0.6 --expert_id 2 --surrounding_obj_num 8
```

The policy finds and uses `model_best.pt` in the trained directory.






# Extra: Define new task environment if needed:
1. In `state_based_grasp_customed.py` write `class StateBasedGraspCustomed`
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
python run_online.py --task StateBasedGraspCustomed --algo ppo --seed 0 --rl_device cuda:0 --num_envs 1000 --max_iterations 30000 --config dedicated_policy_customed.yaml --object_scale_file train_set_results.yaml --start_line 525 --end_line 526
```