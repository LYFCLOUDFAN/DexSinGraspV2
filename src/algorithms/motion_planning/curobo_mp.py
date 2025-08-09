from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from typing import Optional
import torch
from ..algo import Algorithm


class CuroboMotionPlanning(Algorithm):
    def __init__(self, env, motion_gen_config_path: str, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        # 加载 cuRobo 的 motion planner
        self.motion_gen = MotionGen(
            MotionGenConfig.load_from_robot_config(
                motion_gen_config_path,
                world_cfg=None,  # 可选：可以自定义障碍物信息
                interpolation_dt=0.01,
            )
        )
        self.motion_gen.warmup()

    def eval(self, iteration: int, num_rounds: Optional[int] = None) -> None:
        self.env.eval()
        num_envs = self.env.num_envs
        num_rounds = num_rounds or self.num_evaluation_rounds

        success_count = 0

        for _ in range(num_rounds):
            obs = self.env.reset()
            start_joint_state = JointState.from_position(
                self.env.robot_dof[:, :6].to(self.device),  # UR5e DOFs
                joint_names=[
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                ]
            )

            # 获取目标末端位姿（以目标物体为例，可按需调整）
            target_obj_pos = self.env.object_targets[:, :3]  # [num_envs, 3]
            target_obj_rot = self.env.object_targets[:, 3:]  # [num_envs, 4]

            goal_poses = [
                Pose.from_list(torch.cat([target_obj_pos[i], target_obj_rot[i]]).tolist())
                for i in range(num_envs)
            ]

            for env_id in range(num_envs):
                result = self.motion_gen.plan_single(
                    start_joint_state[env_id:env_id+1],
                    goal_poses[env_id],
                    MotionGenPlanConfig(max_attempts=1)
                )

                if result.success:
                    trajectory = result.get_interpolated_plan()  # [T, 6]
                    print(f"[env {env_id}] Motion plan successful. Steps: {trajectory.shape[0]}")

                    # 可以在仿真中执行轨迹（这里只演示，未封装执行逻辑）
                    for q in trajectory:
                        action = q.to(self.device).unsqueeze(0)  # [1, 6]
                        # 如果你需要混合 ShadowHand，可考虑组合 6+24 维动作
                        self.env.step(action)

                    success_count += 1
                else:
                    print(f"[env {env_id}] Motion plan failed")

        success_rate = success_count / (num_envs * num_rounds)
        print(f"[cuRobo Evaluation] Success rate: {success_rate * 100:.2f}%")
