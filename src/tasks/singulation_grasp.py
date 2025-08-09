import enum
import math
import os
import random
import warnings
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import cv2
import numpy as np
import omegaconf
import open3d as o3d
import pandas as pd
import pytorch3d
import torch
import trimesh
from dotenv import find_dotenv
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_quaternion
from torch import LongTensor, Tensor

from .dataset import OakInkDataset, point_to_mesh_distance
from .isaacgym_utils import (
    ActionSpec,
    ObservationSpec,
    draw_axes,
    draw_boxes,
    get_action_indices,
    ik,
    orientation,
    position,
    print_action_space,
    print_asset_options,
    print_dof_properties,
    print_links_and_dofs,
    print_observation_space,
    random_orientation_within_angle,
    to_torch,
)
from .torch_utils import *

# for debug
test_ik = False
test_sim = False
test_rel = False
test_pcl = False
fix_wrist = False
wrist_zero_action = False
test = False
local_test = False

success_tolerance = 0.1
height_success_tolerance = 0.1
negative_part_reward = False
trans_scale = 10
hand_pcl_num = 1024
batch_size = 1000
high_thumb_reward = False
set_arm_pose_according_to_object = False

video_pose = [0.0, -0.3, -0.3]

add_noise = False
STATIC_TARGET = False


class XArmAllegroHandUnderarmDimensions(enum.Enum):
    """Dimension constants for Isaac Gym with xArm6 + Allegro Hand."""

    # general state
    # cartesian position (3) + quaternion orientation (4)
    POSE_DIM = 7
    # linear velocity (3) + angular velocity (3)
    VELOCITY_DIM = 6
    # pose (7) + velocity (6)
    STATE_DIM = 13
    # force (3) + torque (3)
    WRENCH_DIM = 6

    NUM_FINGERTIPS = 4  # Allegro hand has 4 fingertips
    NUM_DOFS = 22  # xArm6 (6 DOF) + Allegro hand (16 DOF)

    WRIST_TRAN = 3
    WRIST_ROT = 3

    # Allegro hand actuated dimensions
    HAND_ACTUATED_DIM = 16


class ForceSensorSpec:
    name: str
    index: int
    rigid_body_name: str
    rigid_body_index: int
    pose: gymapi.Transform

    def __init__(
        self,
        name: str,
        rigid_body_name: str,
        translation: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        *,
        index: int = -1,
        rigid_body_index: int = -1,
        pose: Optional[gymapi.Transform] = None,
    ) -> None:
        assert not (((translation is not None) or (rotation is not None)) and (pose is not None))
        if pose is not None:
            pass
        elif (translation is not None) or (rotation is not None):
            pose = gymapi.Transform()
            if translation is not None:
                assert len(translation) == 3
                pose.p = gymapi.Vec3(*translation)
            if rotation is not None:
                assert len(rotation) == 4
                pose.r = gymapi.Quat(*rotation)
        else:
            pose = gymapi.Transform()

        self.name = name
        self.index = index
        self.rigid_body_name = rigid_body_name
        self.rigid_body_index = rigid_body_index
        self.pose = pose
        self.translation, self.rotation = position(pose), orientation(pose)


class AggregateTracker:
    aggregate_bodies: int
    aggregate_shapes: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.aggregate_bodies = 0
        self.aggregate_shapes = 0

    def update(self, bodies: int, shapes: int):
        self.aggregate_bodies += bodies
        self.aggregate_shapes += shapes


class XArmAllegroHandFunctionalManipulationUnderarm(VecTask):
    # constants
    _asset_root: os.PathLike = os.path.join(os.path.dirname(find_dotenv()), "assets/urdf")
    _data_root: os.PathLike = os.path.join(os.path.dirname(find_dotenv()), "data")
    _allegro_hand_right_asset_file: os.PathLike = os.path.join("hands", "allegro_hand", "allegro_hand_right.urdf")
    _allegro_hand_left_asset_file: os.PathLike = os.path.join("hands", "allegro_hand", "allegro_hand_left.urdf")
    _xarm_allegro_hand_right_asset_file: str = "xarm6_allegro_right.urdf"
    _xarm_allegro_hand_left_asset_file: str = "xarm6_allegro_left.urdf"

    # fmt: off
    _xarm_dof_names: List[str] = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    # Allegro hand DOF names (16 DOF total)
    _allegro_hand_dof_names: List[str] = [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",  # finger 0 (index)
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",  # finger 1 (middle)
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",  # finger 2 (ring)
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",  # thumb
    ]

    # Group allegro hand DOF names by finger
    _allegro_finger0_dof_names: List[str] = ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"]
    _allegro_finger1_dof_names: List[str] = ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"]
    _allegro_finger2_dof_names: List[str] = ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"]
    _allegro_thumb_dof_names: List[str] = ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]

    _allegro_fingers_dof_names: List[str] = (
        _allegro_finger0_dof_names + _allegro_finger1_dof_names + _allegro_finger2_dof_names
    )
    _allegro_digits_dof_names: List[str] = _allegro_fingers_dof_names + _allegro_thumb_dof_names
    # fmt: on

    _arm_links: List[str] = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]
    _hand_links: List[str] = [
        "base_link", "palm", "wrist",
        "link_0.0", "link_1.0", "link_2.0", "link_3.0",
        "link_4.0", "link_5.0", "link_6.0", "link_7.0",
        "link_8.0", "link_9.0", "link_10.0", "link_11.0",
        "link_12.0", "link_13.0", "link_14.0", "link_15.0",
    ]
    _fingertips: List[str] = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"] # index, middle, ring, thumb
    _allegro_hand_center_prim: str = "base_link"
    _allegro_hand_palm_prim: str = "palm"
    # fmt: off
    _keypoints: List[str] = [
        # "palm",
        "link_12.0", "link_13.0", "link_14.0", "link_15.0_tip",  # thumb
        "link_0.0", "link_1.0", "link_2.0", "link_3.0_tip",     # finger 0 (index)
        "link_4.0", "link_5.0", "link_6.0", "link_7.0_tip",     # finger 1 (middle)
        "link_8.0", "link_9.0", "link_10.0", "link_11.0_tip",   # finger 2 (ring)
    ]
    # fmt: on

    _xarm_right_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    _xarm_left_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    allegro_hand_init_dof_positions: Dict[str, float] = {
        "joint_0.0": -0.14,
        "joint_1.0": 1.0,
        "joint_2.0": 1.8,
        "joint_3.0": 0.9,
        "joint_4.0": 0.0,
        "joint_5.0": 0.0,
        "joint_6.0": 0.15,
        "joint_7.0": 0.29,
        "joint_8.0": 0.38,
        "joint_9.0": 1.3,
        "joint_10.0": 1.4,
        "joint_11.0": 1.0,
        "joint_12.0": 0.46,
        "joint_13.0": 0.08,
        "joint_14.0": 1.0,
        "joint_15.0": 0.53,
    }
    allegro_hand_init_dof_positions: Dict[str, float] = {
        "joint_0.0": 0,
        "joint_1.0": 0.0,
        "joint_2.0": 0,
        "joint_3.0": 0,
        "joint_4.0": 0.0,
        "joint_5.0": 0.0,
        "joint_6.0": 0,
        "joint_7.0": 0,
        "joint_8.0": 0,
        "joint_9.0": 0,
        "joint_10.0": 0,
        "joint_11.0": 0,
        "joint_12.0": 0,
        "joint_13.0": 0.0,
        "joint_14.0": 0,
        "joint_15.0": 0.0,
    }

    _xarm_right_init_position = [0.00, 0.65, 0.00]
    _xarm_right_init_orientation = [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)]
    _allegro_hand_predef_qpos = [0] * 16  # reset if use predef qpos
    _target_hand_palm_pose = [-0.4, 0.053, 0.810, 0.0, -0.707, 0.707, 0.0]
    _current_hand_palm_pose = [0.021, 0.052, 0.608, 0.0, -0.707, 0.707, 0.0]
    _hand_geo_center = [0.0, 0.0, 0.0]
    _object_z = 0.5
    _object_nominal_orientation = [0.0, 0.0, 1.0, 0.0]
    _table_x_length = 0.5
    _table_y_length = 0.5
    _table_thickness = 0.02
    _table_pose = [0.0, 0.0, 0.4]

    _max_xarm_endeffector_pos_vel = 1.0
    _max_xarm_endeffector_rot_vel = torch.pi

    _palm2forearm_quat = [0.0, 0.0, 0.0, 1.0]
    _palm2forearm_pos = [0.0, -0.01, 0.247]

    _dims = XArmAllegroHandUnderarmDimensions
    _observation_specs: Sequence[ObservationSpec] = []
    _action_specs: Sequence[ActionSpec] = []
    _force_sensor_specs: Sequence[ForceSensorSpec] = [
        ForceSensorSpec("link_3.0_tip", "link_3.0_tip"),
        ForceSensorSpec("link_7.0_tip", "link_7.0_tip"),
        ForceSensorSpec("link_11.0_tip", "link_11.0_tip"),
        ForceSensorSpec("link_15.0_tip", "link_15.0_tip"),
    ]

    # TODO: add description about tensor shapes
    allegro_hand_index: int

    allegro_hand_dof_lower_limits: Tensor
    allegro_hand_dof_upper_limits: Tensor
    allegro_hand_dof_init_positions: Tensor
    allegro_hand_dof_init_velocities: Tensor

    allegro_hand_dof_start: int
    allegro_hand_dof_end: int
    target_allegro_hand_dof_start: int
    target_allegro_hand_dof_end: int

    # buffers to hold intermediate results
    root_states: Tensor
    root_positions: Tensor
    root_orientations: Tensor
    root_linear_velocities: Tensor
    root_angular_velocities: Tensor

    allegro_hand_root_states: Tensor
    allegro_hand_root_positions: Tensor
    allegro_hand_root_orientations: Tensor
    allegro_hand_root_linear_velocities: Tensor
    allegro_hand_root_angular_velocities: Tensor

    scene_object_root_states: Tensor
    scene_object_root_positions: Tensor
    scene_object_root_orientations: Tensor
    scene_object_root_linear_velocities: Tensor
    scene_object_root_angular_velocities: Tensor

    allegro_hand_dof_positions: Tensor
    allegro_hand_dof_velocities: Tensor

    target_allegro_hand_dof_positions: Tensor
    target_allegro_hand_dof_velocities: Tensor

    # tensors need to be refreshed manually
    fingertip_states: Tensor
    fingertip_positions: Tensor
    fingertip_orientations: Tensor
    fingertip_positions_wrt_palm: Tensor
    fingertip_orientations_wrt_palm: Tensor
    fingertip_linear_velocities: Tensor
    fingertip_angular_velocities: Tensor

    object_root_states: Tensor
    object_root_positions: Tensor
    object_root_orientations: Tensor
    object_positions_wrt_palm: Tensor
    object_orientations_wrt_palm: Tensor

    prev_targets: Tensor
    curr_targets: Tensor

    successes: Tensor
    consecutive_successes: Tensor

    object_spacing: float
    num_objects_per_env: int

    _obj_width: float = 0.04
    _obj_depth: float = 0.16
    _obj_height: float = 0.24
    _grid_rows: int = 1
    _grid_cols: int = 5
    _grid_layers: int = 1
    _obj_spacing: float = 0.005
    VISUAL_TARGET_COLLISION_FILTER = 0x7FFFFFFF
    

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        seed = cfg["env"]["seed"]
        torch.manual_seed(seed)  # cpu
        random.seed(seed)
        np.random.seed(seed)

        self.cfg = cfg

        self.method = self.cfg["env"]["method"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.use_predef_hand_pose = self.cfg["env"]["usePredefHandPose"]

        self.sub_steps = self.cfg["sim"]["substeps"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.enable_contact_sensors = self.cfg["env"]["enableContactSensors"] or self.cfg["env"]["tactileObs"]
        self.contact_sensor_fingertip_only = self.cfg["env"]["contactSensorFingertipOnly"]
        self.contact_sensor_fingertip_from_all = self.cfg["env"]["contactSensorFingertipFromAll"]
        self.contact_sensor_threshold = self.cfg["env"]["contactSensorThreshold"]

        # Section for functional grasping dataset
        # self.dataset_dir = self.cfg["env"]["datasetDir"]
        # self.dataset_metainfo_path = self.cfg["env"]["datasetMetainfoPath"]
        # self.dataset_skipcode_path = self.cfg["env"]["datasetSkipcodePath"]
        # self.dataset_pose_level_sampling = self.cfg["env"]["datasetPoseLevelSampling"]
        # self.dataset_queries = self.cfg["env"]["datasetQueries"]

        self.object_spacing = self.cfg["env"]["objectSpacing"]
        self.num_objects = self.cfg["env"]["numObjects"]
        self.num_objects_per_env = self.cfg["env"]["numObjectsPerEnv"]

        self.reset_obj_ori_noise = self.cfg["env"]["resetObjOriNoise"]

        self.velocity_observation_scale = self.cfg["env"]["velocityObservationScale"]
        self.reward_type = self.cfg["env"]["rewardType"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.tran_reward_scale = self.cfg["env"]["tranRewardScale"]
        self.contact_reward_scale = self.cfg["env"]["contactRewardScale"]
        # if "curr" in self.reward_type:
        #     self.tran_reward_scale = 1.0

        self.action_noise = self.cfg["env"]["actionNoise"]
        self.action_noise_level = self.cfg["env"]["actionNoiseLevel"]
        self.action_noise_ratio = self.cfg["env"]["actionNoiseRatio"]
        self.action_noise_sigma = self.cfg["env"]["actionNoiseSigma"]
        self.action_noise_max_times = self.cfg["env"]["actionNoiseMaxTimes"]
        assert self.action_noise_level in ["step", "value"]

        self.relative_part_reward = self.cfg["env"]["relativePartReward"]
        self.part_reward_scale = self.cfg["env"]["partRewardScale"]
        self.height_reward_scale = self.cfg["env"]["heightRewardScale"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.contact_eps = self.cfg["env"]["contactEps"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        if fix_wrist or wrist_zero_action:
            self.wrist_action_penalty_scale = 0
        else:
            self.wrist_action_penalty_scale = self.cfg["env"]["wristActionPenaltyScale"]
        self.arm_action_penalty_scale = self.cfg["env"]["armActionPenaltyScale"]
        self.similarity_reward_scale = self.cfg["env"]["similarityRewardScale"]
        self.similarity_reward_freq = self.cfg["env"]["similarityRewardFreq"]

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.height_scale = self.cfg["env"]["heightScale"]
        self.time_step_penatly = self.cfg["env"]["timeStepPenatly"]
        self.manipulability_penalty_scale = self.cfg["env"]["manipulabilityPenaltyScale"]

        # Singulation-specific reward parameters
        self.tilt_reward_scale = self.cfg["env"].get("tiltRewardScale", 1.0)
        self.slide_reward_scale = self.cfg["env"].get("slideRewardScale", 1.0)
        self.neighbor_stability_penalty_scale = self.cfg["env"].get("neighborStabilityPenaltyScale", -5.0)
        self.stability_penalty_scale = self.cfg["env"].get("stabilityPenaltyScale", -2.0)
        
        # self.neighbor_pos_penalty_scale = 25.0
        # self.neighbor_rot_penalty_scale = 10.0
        self.neighbor_pos_penalty_scale = .0
        self.neighbor_rot_penalty_scale = .0
        self.safety_distance_sigma = 0.15
        self.pos_disturbance_threshold = 0.002
        self.rot_disturbance_threshold = 0.05
        self.contact_penalty_scale = self.cfg["env"].get("contactPenaltyScale", 10.0)
        self.contact_force_threshold = self.cfg["env"].get("contactForceThreshold", 0.5)

        # Goal pose parameters for singulation task
        self.goal_translation_y = self.cfg["env"]["goalTranslationY"]
        self.goal_rotation_x = self.cfg["env"]["goalRotationX"]
        self.goal_tolerance_position = self.cfg["env"]["goalTolerancePosition"]
        self.goal_tolerance_rotation = self.cfg["env"]["goalToleranceRotation"]
        self.success_steps = self.cfg["env"]["successSteps"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.env_info_logging = self.cfg["logging"]["envInfo"]


        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        # Section for rendered point cloud observation
        self.real_pcl_obs = self.cfg["env"]["realPclObs"]
        self.enable_rendered_pointcloud_observation = (
            self.cfg["env"]["enableRenderedPointCloud"]
            or self.cfg["env"]["realPclObs"]
            or "rendered_pointcloud" in self.cfg["env"]["observationSpace"]
        )
        self.num_rendered_points = self.cfg["env"]["numRenderedPointCloudPoints"]
        self.rendered_pointcloud_multiplier = self.cfg["env"]["renderedPointCloudMultiplier"]
        self.rendered_pointcloud_sample_method = self.cfg["env"]["renderedPointCloudSampleMethod"]
        self.rendered_pointcloud_gaussian_noise = self.cfg["env"]["renderedPointCloudGaussianNoise"]
        self.rendered_pointcloud_gaussian_noise_sigma = self.cfg["env"]["renderedPointCloudGaussianNoiseSigma"]
        self.rendered_pointcloud_gaussian_noise_ratio = self.cfg["env"]["renderedPointCloudGaussianNoiseRatio"]
        assert self.rendered_pointcloud_sample_method in ["farthest", "random"]

        if self.enable_rendered_pointcloud_observation and not self.cfg["env"].get("enableCameraSensors", False):
            warnings.warn("enableRenderedPointCloud is set to True but enableCameraSensors is set to False.")
            warnings.warn("overriding enableCameraSensors to True.")
            self.cfg["env"]["enableCameraSensors"] = True

        self.vis_env_num = self.cfg["env"]["visEnvNum"]
        self.vis_image_size = self.cfg["env"]["visImageSize"]
        if self.vis_env_num > 0:
            self.cfg["env"]["enableCameraSensors"] = True
            self.save_video = True
        else:
            self.save_video = False

        self.img_pcl_obs = self.cfg["env"]["imgPclObs"]
        self.num_imagined_points = self.cfg["env"]["numImaginedPointCloudPoints"]
        self.enable_imagined_pointcloud_observation = (
            self.cfg["env"]["enableImaginedPointCloud"] or self.cfg["env"]["imgPclObs"]
        )

        self.num_object_points = self.cfg["env"]["numObjectPointCloudPoints"]

        self.num_nearest_non_targets = self.cfg['env']['observationSpecs']['__dim__']['num_nearest_non_targets']

        self.up_axis = "z"

        self.mode = self.cfg["env"]["mode"]
        # - orn: object orientation.
        # - relpose: relative pose.
        # - relposecontact: relative pose + finger pose
        self.env_mode = self.cfg["env"]["envMode"]
        self.curriculum_mode = self.cfg["env"]["curriculumMode"]

        self.render_target = self.cfg["env"].get("renderTarget", False)

        self.manipulated_object_codes = None
        self.resample_object = self.cfg["env"]["resampleObject"]


        self.aggregate_tracker = AggregateTracker()

        if self.env_mode == "orn":
            # self.cfg["env"]["actionSpace"] = ["hand_rotation", "wrist_3_joint"]
            self.object_targets = torch.zeros(self.cfg["env"]["numEnvs"], 4, device=sim_device)
            # if "wrist_3_joint" not in self.cfg["env"]["actionSpace"]:
            self._xarm_right_init_dof_positions = {
                "joint1": 0.0,
                "joint2":-1.0,
                "joint3":-0.5,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
            }
        elif self.env_mode == "relpose":
            # self.cfg["env"]["actionSpace"] = ["hand_rotation"]
            self.object_targets = torch.zeros(self.cfg["env"]["numEnvs"], 3 + 4, device=sim_device)
            # if "wrist_3_joint" not in self.cfg["env"]["actionSpace"]:
            self._xarm_right_init_dof_positions = {
                "joint1": 0.0,
                "joint2":-1.0,
                "joint3":-0.5,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
            }
        elif self.env_mode == "relposecontact":
            # self.cfg["env"]["actionSpace"] = ["hand_rotation"]
            self.object_targets = torch.zeros(self.cfg["env"]["numEnvs"], 3 + 4 + 18, device=sim_device)
            # if "wrist_3_joint" not in self.cfg["env"]["actionSpace"]:
            self._xarm_right_init_dof_positions = {
                "joint1": 0.0,
                "joint2":-1,
                "joint3":-0.5,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
            }
        elif self.env_mode == "pgm":
            self.object_targets = torch.zeros(self.cfg["env"]["numEnvs"], 3 + 4 + 18, device=sim_device)
            self._xarm_right_init_dof_positions = {
                "joint1": 0.0,
                "joint2":-1,
                "joint3":-0.5,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
            }
            self._hand_geo_center = [0, 0, 0]
            self._object_z = 0.01 + self._table_thickness / 2
            self._current_hand_palm_pose = [0.02, 0.3, 0.6, 0.707, 0.0, 0.0, 0.707]
            self.ur_control_type = "osc"

            if self.relative_part_reward:
                self.prev_pos_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1
                self.prev_rot_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1
                self.prev_contact_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1
                self.prev_nominal_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1

            self.curriculum_thres = 0.9
            if "stage" in self.curriculum_mode:
                self.height_scale = 0
                if self.relative_part_reward:
                    self.part_reward_scale = 1.0
                else:
                    self.part_reward_scale = 0.3
                self.nominal_env_ratio = 0.2
            elif "pose" in self.curriculum_mode:
                self.height_scale = 0
                if self.relative_part_reward:
                    self.part_reward_scale = 1.0
                else:
                    self.part_reward_scale = 0.3
                self.nominal_env_ratio = 1.0
            else:
                self.nominal_env_ratio = 0.2

        self.stack_frame_number = self.cfg["env"]["stackFrameNumber"]
        self.frames = deque([], maxlen=self.stack_frame_number)
        # self.goal_position = torch.tensor([0.0, 0.5, 0.75], device=sim_device, dtype=torch.float)
        self.goal_position = torch.tensor([0.0, 0.2, 1.0], device=sim_device, dtype=torch.float)
        self.goal_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim_device, dtype=torch.float)
        # non-target object
        self.max_non_targets = self.num_objects_per_env - 1  # Maximum possible non-target objects per env
        self.k_nearest = min(self.num_nearest_non_targets, self.max_non_targets)
        


        # TODO: define structure to hold all the indices
        # mapping from name to asset instance
        self.gym_assets = {}
        self.gym_assets["current"] = {}
        self.gym_assets["target"] = {}

        self.num_fingertips = len(self._fingertips)

        # self.__create_functional_grasping_dataset(device=sim_device)
        self.__create_box_grid_dataset(device=sim_device)
        self.__configure_mdp_spaces()

        super().__init__(  # create_sim
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        # reconfig viewer
        self.__configure_viewer()
        # HACK: not used
        # self.__reset_grasping_joint_indices()
        self.__reset_action_indices()

        # retrieve generic tensor descriptors for the simulation
        # - root_states: [num_envs * num_actors, 13]
        _root_states: torch.Tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # - dof_states: [num_envs * num_dofs, 2]
        _dof_states: torch.Tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # - dof_forces: [num_envs * num_dofs]
        _dof_forces: torch.Tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # - rigid_body_states: [num_envs * num_rigid_bodies, 13]
        _rigid_body_states: torch.Tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # - net_contact_forces: [num_envs * num_rigid_bodies, 3]
        _net_contact_forces: torch.Tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # - force_sensor_states: [num_envs * num_force_sensors, 6]
        _force_sensor_states: torch.Tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # - jacobians: [num_envs, num_prims - 1, 6, num_dofs]
        _jacobians: torch.Tensor = self.gym.acquire_jacobian_tensor(self.sim, "allegro_hand")

        if self.env_info_logging:
            print("root_states.shape: ", _root_states.shape)
            print("dof_states.shape: ", _dof_states.shape)
            print("rigid_body_states.shape: ", _rigid_body_states.shape)
            print("net_contact_forces.shape: ", _net_contact_forces.shape)
            print("force_sensor_states.shape: ", _force_sensor_states.shape)
            print("dof_forces.shape: ", _dof_forces.shape)
            print("jacobians.shape: ", _jacobians.shape)

        self.num_actors: int = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.num_dofs: int = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_force_sensors: int = self.gym.get_sim_force_sensor_count(self.sim) // self.num_envs
        self.num_rigid_bodies: int = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs

        if self.env_info_logging:
            print("num_actors: ", self.num_actors)
            print("num_dofs: ", self.num_dofs)
            print("num_force_sensors: ", self.num_force_sensors)
            print("num_rigid_bodies: ", self.num_rigid_bodies)

        # Wrap tensors with gymtorch
        self.root_states: torch.Tensor = gymtorch.wrap_tensor(_root_states)
        self.dof_states: torch.Tensor = gymtorch.wrap_tensor(_dof_states)
        self.dof_forces: torch.Tensor = gymtorch.wrap_tensor(_dof_forces)
        self.rigid_body_states: torch.Tensor = gymtorch.wrap_tensor(_rigid_body_states)
        self.net_contact_forces: torch.Tensor = gymtorch.wrap_tensor(_net_contact_forces)
        self.jacobians: torch.Tensor = gymtorch.wrap_tensor(_jacobians)

        if self.num_force_sensors > 0:
            self.force_sensor_states: torch.Tensor = gymtorch.wrap_tensor(_force_sensor_states)
        else:
            self.force_sensor_states: Optional[torch.Tensor] = None

        forearm_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], "link6")
        # jacobian entries corresponding to link6
        self.j_eef = self.jacobians[:, forearm_index - 1, :, :6]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create tensors to hold observations, actions, and rewards for each environment
        # only contiguous slices can be defined here
        # non-contiguous slices will be defined in `_refresh_sim_tensors`
        self.root_positions = self.root_states[:, 0:3]
        self.root_orientations = self.root_states[:, 3:7]
        self.root_linear_velocities = self.root_states[:, 7:10]
        self.root_angular_velocities = self.root_states[:, 10:13]

        root_states = self.root_states.view(self.num_envs, self.num_actors, 13)

        self.allegro_hand_root_states = root_states[:, self.allegro_hand_index, :]
        self.allegro_hand_root_positions = self.allegro_hand_root_states[:, 0:3]
        self.allegro_hand_root_orientations = self.allegro_hand_root_states[:, 3:7]
        self.allegro_hand_root_linear_velocities = self.allegro_hand_root_states[:, 7:10]
        self.allegro_hand_root_angular_velocities = self.allegro_hand_root_states[:, 10:13]
        
        self.surr_object_root_states = self.root_states[self.surr_object_indices]
        
        self.surr_object_root_positions = self.surr_object_root_states[..., 0:3].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_orientations = self.surr_object_root_states[..., 3:7].view(self.num_envs, self.max_non_targets, 4)
        self.surr_object_root_linear_velocities = self.surr_object_root_states[..., 7:10].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_angular_velocities = self.surr_object_root_states[..., 10:13].view(self.num_envs, self.max_non_targets, 3)
        self.prev_surr_object_root_positions = self.surr_object_root_positions
        self.prev_surr_object_root_orientations = self.surr_object_root_orientations
        self.prev_surr_object_root_linear_velocities = self.surr_object_root_linear_velocities
        self.prev_surr_object_root_angular_velocities = self.surr_object_root_angular_velocities
        

        self.scene_object_root_positions = self.root_positions[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_orientations = self.root_orientations[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 4)
        self.scene_object_root_linear_velocities = self.root_linear_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_angular_velocities = self.root_angular_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)

        self.init_scene_object_root_positions = self.scene_object_root_positions.clone()
        self.init_scene_object_root_orientations = self.scene_object_root_orientations.clone()

        dof_states = self.dof_states.view(self.num_envs, self.num_dofs, 2)

        self.allegro_hand_dof_positions = dof_states[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 0]
        self.allegro_hand_dof_velocities = dof_states[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 1]


        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_rigid_bodies, 13)

        self.allegro_hand_rigid_body_states = rigid_body_states[
            :, self.allegro_hand_rigid_body_start : self.allegro_hand_rigid_body_end, :
        ]
        self.allegro_hand_rigid_body_positions = self.allegro_hand_rigid_body_states[..., 0:3]
        self.allegro_hand_rigid_body_orientations = self.allegro_hand_rigid_body_states[..., 3:7]
        self.allegro_hand_rigid_body_linear_velocities = self.allegro_hand_rigid_body_states[..., 7:10]
        self.allegro_hand_rigid_body_angular_velocities = self.allegro_hand_rigid_body_states[..., 10:13]

        self.allegro_hand_center_states = self.allegro_hand_rigid_body_states[:, self.allegro_center_index, :]
        self.allegro_hand_center_positions = self.allegro_hand_center_states[:, 0:3]
        self.allegro_hand_center_orientations = self.allegro_hand_center_states[:, 3:7]


        endeffector_index = self.gym.find_asset_rigid_body_index(
            self.gym_assets["current"]["robot"]["asset"], "link6"
        )
        self.endeffector_states = self.allegro_hand_rigid_body_states[:, endeffector_index, :]
        self.endeffector_positions = self.allegro_hand_rigid_body_positions[:, endeffector_index, :]
        self.endeffector_orientations = self.allegro_hand_rigid_body_orientations[:, endeffector_index, :]
        self.endeffector_linear_velocities = self.allegro_hand_rigid_body_linear_velocities[:, endeffector_index, :]
        self.endeffector_angular_velocities = self.allegro_hand_rigid_body_angular_velocities[:, endeffector_index, :]

        self.nearest_non_target_object_positions = torch.zeros((self.num_envs, self.num_nearest_non_targets, 3), device=self.device)
        self.nearest_non_target_object_orientations = torch.zeros((self.num_envs, self.num_nearest_non_targets, 4), device=self.device)


        # Intermediate tensors for _refresh_sim_tensors
        self._target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self._gather_indices_pos = torch.zeros((self.num_envs, self.max_non_targets, 3), dtype=torch.long, device=self.device)
        self._gather_indices_ori = torch.zeros((self.num_envs, self.max_non_targets, 4), dtype=torch.long, device=self.device)
        self._non_target_positions = torch.zeros((self.num_envs, self.max_non_targets, 3), device=self.device)
        self._non_target_orientations = torch.zeros((self.num_envs, self.max_non_targets, 4), device=self.device)
        self._distances = torch.zeros((self.num_envs, self.max_non_targets), device=self.device)
        self._sorted_distances = torch.zeros((self.num_envs, self.max_non_targets), device=self.device)
        self._sorted_indices = torch.zeros((self.num_envs, self.max_non_targets), dtype=torch.long, device=self.device)
        self._nearest_indices = torch.zeros((self.num_envs, self.k_nearest), dtype=torch.long, device=self.device)
        self._batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.k_nearest)
        self._valid_mask = torch.zeros((self.num_envs, self.max_non_targets), dtype=torch.bool, device=self.device)
        self._valid_nearest = torch.zeros((self.num_envs, self.k_nearest), dtype=torch.bool, device=self.device)
        self._invalid_mask_pos = torch.zeros((self.num_envs, self.k_nearest, 3), dtype=torch.bool, device=self.device)
        self._invalid_mask_ori = torch.zeros((self.num_envs, self.k_nearest, 4), dtype=torch.bool, device=self.device)
        self._inf_tensor = torch.full((self.num_envs, self.max_non_targets), torch.inf, device=self.device)


        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)

        self.allegro_hand_net_contact_forces = net_contact_forces[
            :, self.allegro_hand_rigid_body_start : self.allegro_hand_rigid_body_end, :
        ]
        # self.surr_object_net_contact_forces = net_contact_forces[
        #     :, self.surr_object_indices, :
        # ]

        # allocate buffers to hold intermediate results

        # render_target - false mode
        kwargs = {"dtype": torch.float, "device": self.device}
        self._r_target_object_root_positions = torch.zeros((self.num_envs, 3), **kwargs)
        self._r_target_object_root_orientations = torch.zeros((self.num_envs, 4), **kwargs)
        self._r_target_allegro_dof_positions = torch.zeros((self.num_envs, 22), **kwargs)  # 6 arm + 16 hand
        self._r_target_allegro_digits_actuated_dof_positions = torch.zeros((self.num_envs, 16), **kwargs)  # 16 Allegro hand DOF
        self._r_target_allegro_fingers_actuated_dof_positions = torch.zeros((self.num_envs, 12), **kwargs)  # 3 fingers × 4 DOF
        self._r_target_allegro_thumb_actuated_dof_positions = torch.zeros((self.num_envs, 4), **kwargs)  # 4 thumb DOF
        self._r_target_object_positions_wrt_palm = torch.zeros((self.num_envs, 3), **kwargs)
        self._r_target_object_orientations_wrt_palm = torch.zeros((self.num_envs, 4), **kwargs)
        self._r_target_palm_positions_wrt_object = torch.zeros((self.num_envs, 3), **kwargs)
        self._r_target_palm_orientations_wrt_object = torch.zeros((self.num_envs, 4), **kwargs)

        self.prev_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), **kwargs)
        self.curr_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), **kwargs)

        # create slices from above buffer
        self.prev_targets = self.prev_targets_buffer[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end]
        self.curr_targets = self.curr_targets_buffer[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end]


        self.rb_forces = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), **kwargs)
        self.occupied_object_init_root_positions = self.root_positions[self.occupied_object_indices, :].view(self.num_envs, 3)
        self.occupied_object_init_root_orientations = self.root_orientations[self.occupied_object_indices, :].view(self.num_envs, 4)
        self.surr_object_init_root_positions = self.root_positions[self.surr_object_indices, :].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_init_root_orientations = self.root_orientations[self.surr_object_indices, :].view(self.num_envs, self.max_non_targets, 4)
        
        self.robot_init_dof = torch.zeros((self.num_envs, self._dims.NUM_DOFS.value), **kwargs)
        self._hand_geo_center = torch.tensor(self._hand_geo_center, **kwargs)
        self._table_pose_tensor = torch.tensor(self._table_pose, **kwargs)
        self._target_hand_palm_pose = torch.tensor(self._target_hand_palm_pose, **kwargs)
        self._current_hand_palm_pose = torch.tensor(self._current_hand_palm_pose, **kwargs)
        self._xarm_right_init_position = torch.tensor(self._xarm_right_init_position, **kwargs)
        self._xarm_right_init_orientation = torch.tensor(self._xarm_right_init_orientation, **kwargs)
        self._palm2forearm_quat = torch.tensor(self._palm2forearm_quat, **kwargs)
        self._palm2forearm_pos = torch.tensor(self._palm2forearm_pos, **kwargs)
        self._object_nominal_orientation = torch.tensor(self._object_nominal_orientation, **kwargs)

        if self.enable_full_pointcloud_observation:
            self.pointclouds = torch.zeros((self.num_envs, self.num_object_points, 3), **kwargs)
            self.pointclouds_wrt_palm = torch.zeros((self.num_envs, self.num_object_points, 3), **kwargs)

        self.__init_meta_data()
        self.preprocess_allegro_pointcloud()

        self.successes = torch.zeros(self.num_envs, **kwargs)
        self.done_successes = torch.zeros(self.num_envs, **kwargs)
        self.current_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes = torch.zeros(1, **kwargs)
        self.unused_object_init_root_positions = torch.stack(
            [position(pose, self.device) for pose in self.gym_assets["current"]["objects"]["poses"]], dim=0
        )
        self.picked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.picked_curr = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.near_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        
        
        self.obj_max_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if "gf" in self.observation_info:
            self.gf = torch.zeros((self.num_envs, self.observation_info["gf"]), **kwargs)
            self.action_gf = torch.zeros((self.num_envs, self.observation_info["gf"]), **kwargs)

        # for DPM
        self.object_bboxes = torch.zeros((self.num_envs, 6), **kwargs)
        self.object_categories = torch.zeros((self.num_envs, self.grasping_dataset._category_matrix.shape[1]), **kwargs)
        self.object_bboxes_wrt_world = torch.zeros((self.num_envs, 6), **kwargs)
        self.object_bboxes_wrt_palm = torch.zeros((self.num_envs, 6), **kwargs)

        self.training = True

        self.max_J = torch.ones(self.num_envs, device=self.device) * -torch.inf

        # for evaluation-only mode
        self.occupied_object_codes: np.ndarray = np.array(["" for _ in range(self.num_envs)])
        self.occupied_object_grasps: np.ndarray = np.array(["" for _ in range(self.num_envs)])
        self.occupied_object_cluster_ids: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # for action noise times tracking
        if self.action_noise and self.action_noise_level == "step" and self.action_noise_max_times > 0:
            self.action_noise_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


        # Initialize singulation reward variables
        self.tilt_reward_scaled = torch.zeros(self.num_envs, device=self.device)
        self.slide_reward_scaled = torch.zeros(self.num_envs, device=self.device)
        self.neighbor_stability_penalty_scaled = torch.zeros(self.num_envs, device=self.device)
        self.stability_penalty_scaled = torch.zeros(self.num_envs, device=self.device)

        # init state has collide with table, so we need to first reset to get robot to a valid pose, then continue simulation
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.reset_arm(first_time=True)

    def reset_arm(self, first_time=False):
        self.reset(first_time=first_time)
        for _ in range(10):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self.compute_observations()

    def step_simulation(self, step_time=1):
        for _ in range(step_time):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self.compute_observations()

    def destroy(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    # def test_pcl(self, env_ptr, env_id):
    #     object_asset_options = gymapi.AssetOptions()
    #     asset_sphere = self.gym.create_sphere(self.sim, 0.002, object_asset_options)
    #     pose = gymapi.Transform()
    #     pose.r = gymapi.Quat(0, 0, 0, 1)
    #     pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    #     pcl = torch.from_numpy(np.load("/home/thwu/Projects/func-mani/test.npy")).to("cuda:0")
    #     for (i, point) in enumerate(pcl):
    #         pose.p = gymapi.Vec3(point[0], point[1], point[2])
    #         capsule_handle = self.gym.create_actor(env_ptr, asset_sphere, pose, "actor{}", i+1000, 0)

    def __init_meta_data(self):
        self.observation_info = {}
        observation_space = self.cfg["env"]["observationSpace"]
        for name in observation_space:
            self.observation_info[name] = self._get_observation_dim(name)

        self.object_codes = []
        for object_codes_each_env in self.object_names:
            for object_code_each_env in object_codes_each_env:
                self.object_codes.append(object_code_each_env)
        self.object_codes = list(set(self.object_codes))
        self.object_cat = self.grasping_dataset.object_cat if self.grasping_dataset.object_cat is not None else "all"
        self.max_per_cat = self.grasping_dataset.max_per_cat if self.grasping_dataset.max_per_cat is not None else "all"
        self.object_geo_level = (
            self.grasping_dataset.object_geo_level if self.grasping_dataset.object_geo_level is not None else "all"
        )
        self.object_scale = (
            self.grasping_dataset.object_scale if self.grasping_dataset.object_scale is not None else "all"
        )
        self.label_paths = self.grasping_dataset.label_paths.copy()

    #######################
    # Imagined Pointcloud #
    #######################

    def preprocess_allegro_pointcloud(self):
        """Preprocess allegro-hand pointcloud.

        Load original allegro-hand pointcloud, apply farthest point sampling, store the result in `self._cached_pointclouds`.

            0.0-3.0 index finger
            4.0-7.0 middle finger
            8.0-11.0 ring finger
            12.0-15.0 thumb
        """


        original_mesh_dir = os.path.join(self._asset_root, "hands", "allegro_hand", "meshes", "visual")
        original_mesh_filepaths: OrderedDict = OrderedDict(
            [
                # Index finger (0.0-3.0)
                ("ffproximal", "link_0.0.glb"),
                ("ffmiddle", "link_1.0.glb"),
                ("ffdistal", "link_2.0.glb"),
                ("fftip", "link_tip.glb"),
                # Middle finger (4.0-7.0)
                ("mfproximal", "link_1.0.glb"),
                ("mfmiddle", "link_2.0.glb"),
                ("mfdistal", "link_3.0.glb"),
                ("mftip", "link_tip.glb"),
                # Ring finger (8.0-11.0)
                ("rfproximal", "link_1.0.glb"),
                ("rfmiddle", "link_2.0.glb"),
                ("rfdistal", "link_3.0.glb"),
                ("rftip", "link_tip.glb"),
                # Thumb (12.0-15.0)
                ("thproximal", "link_12.0_right.glb"),
                ("thmiddle", "link_13.0.glb"),
                ("thdistal", "link_14.0.glb"),
                ("thtip", "link_tip.glb"),
            ]
        )

        # load original mesh
        components = OrderedDict()
        for name, filepath in original_mesh_filepaths.items():
            # Map to actual link names used in Allegro hand
            if name.startswith("ff"):  # Index finger (finger 0)
                if "proximal" in name:
                    link_name = "link_0.0"
                elif "middle" in name:
                    link_name = "link_1.0"
                elif "distal" in name:
                    link_name = "link_2.0"
                else:  # tip
                    link_name = "link_3.0_tip"
            elif name.startswith("mf"):  # Middle finger (finger 1)
                if "proximal" in name:
                    link_name = "link_4.0"
                elif "middle" in name:
                    link_name = "link_5.0"
                elif "distal" in name:
                    link_name = "link_6.0"
                else:  # tip
                    link_name = "link_7.0_tip"
            elif name.startswith("rf"):  # Ring finger (finger 2)
                if "proximal" in name:
                    link_name = "link_8.0"
                elif "middle" in name:
                    link_name = "link_9.0"
                elif "distal" in name:
                    link_name = "link_10.0"
                else:  # tip
                    link_name = "link_11.0_tip"
            elif name.startswith("th"):  # Thumb
                if "proximal" in name:
                    link_name = "link_12.0"
                elif "middle" in name:
                    link_name = "link_13.0"
                elif "distal" in name:
                    link_name = "link_14.0"
                else:  # tip
                    link_name = "link_15.0_tip"
            else:
                link_name = name

            components[link_name] = {}
            components[link_name]["mesh"] = trimesh.load(
                os.path.join(original_mesh_dir, filepath), process=False, force="mesh"
            )

            area = components[link_name]["mesh"].area
            if "proximal" in name:
                area *= 0.3
            elif "middle" in name:
                area *= 0.6
            components[link_name]["area"] = area

        # compute number of samples for each component
        area = sum([item["area"] for item in components.values()])
        num_samples = self.num_imagined_points
        for name in components:
            components[name]["num_samples"] = int(round(components[name]["area"] / area * num_samples))
            area -= components[name]["area"]
            num_samples -= components[name]["num_samples"]
        assert sum([item["num_samples"] for item in components.values()]) == self.num_imagined_points

        # apply farthest point sampling
        pointclouds = {}
        for name in components:
            vertices = torch.tensor(components[name]["mesh"].vertices, dtype=torch.float, device=self.device)
            vertices *= 0.001  # convert to meter
            pcd = pytorch3d.ops.sample_farthest_points(vertices[None, ...], K=components[name]["num_samples"])[0][0]
            pointclouds[name] = pcd

            components[name]["pointcloud"] = pcd
            components[name]["contact"] = self.extract_contact_region(pcd)

        # find rigid body index for each component
        current_robot_asset = self.gym_assets["current"]["robot"]["asset"]
        # target_robot_asset = self.gym_assets["target"]["robot"]["asset"]

        if self.enable_contact_sensors:
            for name in components:
                components[name]["current_index"] = self.gym.find_asset_rigid_body_index(current_robot_asset, name)
                # components[name]["target_index"] = self.gym.find_asset_rigid_body_index(target_robot_asset, name)

                # For Allegro hand, map to force sensor names at fingertips
                if name in ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]:
                    sensor_name = f"sensor_{name}"
                    components[name]["sensor_index"] = (
                        self.force_sensor_names.index(sensor_name) if sensor_name in self.force_sensor_names else -1
                    )
                else:
                    components[name]["sensor_index"] = -1  # No sensor for non-tip links
                print(f"Link: {name}, Sensor: {components[name]['sensor_index']}")

        self._cached_pointclouds = pointclouds
        self.imagined_pointcloud_components = components
        # print(self.imagined_pointcloud_components)

    def extract_contact_region(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """Split the allegro-hand pointcloud to `front` and `back` side."""
        x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
        return (x.abs() < 0.9 * x.abs().max()) & (z.abs() < 0.9 * z.abs().max()) & (y < 0)

    def compute_imagined_pointclouds(
        self,
        stage: str,
        return_finger_index: bool = False,
        return_part_index: bool = False,
        return_binary_contact: bool = False,
    ) -> torch.Tensor:
        """Compute imagined pointclouds.

        Args:
            stage (str): "current" or "target"
            return_finger_index (bool, optional): _description_. Defaults to False.
            return_part_index (bool, optional): _description_. Defaults to False.
            return_binary_contact (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: imagined pointclouds w.r.t. world frame (num_envs, num_imagined_points, 3)
        """
        assert stage in ["current", "target"], "stage must be either `current` or `target`"

        if stage == "current":
            rigid_body_positions = self.allegro_hand_rigid_body_positions
            rigid_body_orientations = self.allegro_hand_rigid_body_orientations
        else:
            rigid_body_positions = self.target_allegro_hand_rigid_body_positions
            rigid_body_orientations = self.target_allegro_hand_rigid_body_orientations

        imagined_pointclouds = torch.zeros((self.num_envs, self.num_imagined_points, 3), device=self.device)
        cursor = 0
        for name in self.imagined_pointcloud_components:
            component = self.imagined_pointcloud_components[name]
            i = component["current_index"] if stage == "current" else component["target_index"]

            pcd = component["pointcloud"].clone()
            if pcd.size(0) == 0:
                continue

            position = rigid_body_positions[:, i]
            rotation = rigid_body_orientations[:, i]

            num_points = pcd.size(0)

            pcd = transformation_apply(rotation[:, None, :], position[:, None, :], pcd[None, :, :])
            imagined_pointclouds[:, cursor : cursor + num_points, :] = pcd

            if return_binary_contact:
                # create binary contact
                contact = torch.zeros((self.num_envs, num_points), device=self.device)
                mask = component["contact"]
                if component["sensor_index"] != -1:
                    contact[:] = mask[None, :] * self.contact_forces[:, component["sensor_index"]][:, None]

            if return_finger_index:
                # create finger index
                finger_indices = torch.zeros((self.num_envs, num_points), device=self.device)
                finger_names = ["_th", "_ff", "_mf", "_rf", "_lf"]
                for i, finger in enumerate(finger_names):
                    if finger in name:
                        finger_indices[:] = i
                        break
                else:
                    raise ValueError(f"Unknown finger name: {name}")

            if return_part_index:
                # create part index
                part_indices = torch.zeros((self.num_envs, num_points), device=self.device)
                part_names = ["proximal", "middle", "distal"]
                for i, part in enumerate(part_names):
                    if part in name:
                        part_indices[:] = i
                        break
                else:
                    raise ValueError(f"Unknown part name: {name}")

            cursor += num_points
        return imagined_pointclouds

    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False

    def __configure_viewer(self):
        """Viewer setup."""
        if self.viewer != None:
            # print("Viewer already exists, skipping viewer setup.")
            # pass
            cam_pos = gymapi.Vec3(1.0, 0.0, 1.2)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.2)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def compute_object_pointclouds(self, stage: str) -> torch.Tensor:
        """Compute the pointclouds of the objects w.r.t. the world frame.

        Args:
            stage (str): "current" or "target"

        Returns:
            torch.Tensor: pointclouds of the objects w.r.t. the world frame (num_envs, num_points, 3)
        """
        assert stage in ["current", "target"], "stage must be either `current` or `target`"

        if stage == "current":
            positions = self.object_root_positions
            orientations = self.object_root_orientations
        else:
            positions = self._r_target_object_root_positions
            orientations = self._r_target_object_root_orientations

        pcd = self.pointclouds.clone()
        pcd = transformation_apply(orientations[:, None, :], positions[:, None, :], pcd)
        return pcd

    def _refresh_sim_tensors(self) -> None:
        """Refresh the tensors for the simulation."""
        # TODO: only refresh tensors that are used in the task to save computation
        # TODO: only allocate once and reuse the tensors

        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)
        self.arm_contact_forces = net_contact_forces[:, self.arm_link_indices, :]
        self.hand_contact_forces = net_contact_forces[:, self.hand_link_indices, :]
        

        # Target object contact forces [num_envs, 3]
        self.target_object_contact_forces = self.net_contact_forces[self.target_object_rigid_body_indices, :].view(self.num_envs, 3)
        
        # Surrounding objects contact forces [num_envs, max_surr_objects, 3]
        self.surr_object_contact_forces = self.net_contact_forces[self.surr_object_rigid_body_indices.flatten(), :].view(self.num_envs, self.max_non_targets, 3)

        self.fingertip_states = self.allegro_hand_rigid_body_states[:, self.fingertip_indices, :]
        self.fingertip_positions = self.fingertip_states[..., 0:3]
        self.fingertip_orientations = self.fingertip_states[..., 3:7]
        self.fingertip_linear_velocities = self.fingertip_states[..., 7:10]
        self.fingertip_angular_velocities = self.fingertip_states[..., 10:13]
        
        self.index_fingertip_positions = self.fingertip_positions[:, 0, :]
        self.middle_fingertip_positions = self.fingertip_positions[:, 1, :]
        self.ring_fingertip_positions = self.fingertip_positions[:, 2, :]
        self.thumb_fingertip_positions = self.fingertip_positions[:, 3, :]

        self.keypoint_positions = self.allegro_hand_rigid_body_positions[:, self.keypoint_indices, :]


        self.object_root_states = self.root_states[self.occupied_object_indices]
        self.object_root_positions = self.object_root_states[..., 0:3]
        self.object_root_orientations = self.object_root_states[..., 3:7]
        self.object_root_linear_velocities = self.object_root_states[..., 7:10]
        self.object_root_angular_velocities = self.object_root_states[..., 10:13]
        
        # surrounding object
        self.surr_object_root_states = self.root_states[self.surr_object_indices]
        
        self.prev_surr_object_root_positions = self.surr_object_root_positions.clone()
        self.prev_surr_object_root_orientations = self.surr_object_root_orientations.clone()
        self.prev_surr_object_root_linear_velocities = self.surr_object_root_linear_velocities.clone()
        self.prev_surr_object_root_angular_velocities = self.surr_object_root_angular_velocities.clone()
        
        self.surr_object_root_positions = self.surr_object_root_states[..., 0:3].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_orientations = self.surr_object_root_states[..., 3:7].view(self.num_envs, self.max_non_targets, 4)
        self.surr_object_root_linear_velocities = self.surr_object_root_states[..., 7:10].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_angular_velocities = self.surr_object_root_states[..., 10:13].view(self.num_envs, self.max_non_targets, 3)

        # scene object
        self.scene_object_root_positions = self.root_positions[ self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_orientations = self.root_orientations[ self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 4)
        self.scene_object_root_linear_velocities = self.root_linear_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_angular_velocities = self.root_angular_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)

        # Compute nearest non-target objects for each environment
        self.nearest_non_target_object_positions.zero_()
        self.nearest_non_target_object_orientations.zero_()

        if self.non_occupied_object_indices.numel() > 0:
            # occupied_object_relative_indices shape: (num_envs,)
            # scene_object_root_positions shape: (num_envs, num_objects_per_env, 3)
            torch.gather(
                self.scene_object_root_positions,
                1,
                self.occupied_object_relative_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
                out=self._target_positions.unsqueeze(1)
            )
            self._target_positions.squeeze_(1)  # Shape: (num_envs, 3)

            # Create expanded indices for gathering (use pre-allocated tensors)
            self._gather_indices_pos[:] = self.non_occupied_object_indices.unsqueeze(-1).expand(-1, -1, 3)
            self._gather_indices_ori[:] = self.non_occupied_object_indices.unsqueeze(-1).expand(-1, -1, 4)

            # Gather non-target positions and orientations (use pre-allocated tensors)
            torch.gather(self.scene_object_root_positions, 1, self._gather_indices_pos, out=self._non_target_positions)
            torch.gather(self.scene_object_root_orientations, 1, self._gather_indices_ori, out=self._non_target_orientations)

            # Compute distances from target to all non-target objects (use pre-allocated tensor)
            # target_positions: (num_envs, 3) -> (num_envs, 1, 3)
            # non_target_positions: (num_envs, max_non_targets, 3)
            torch.norm(
                self._non_target_positions - self._target_positions.unsqueeze(1),
                dim=2,
                out=self._distances
            )  # Shape: (num_envs, max_non_targets)

            # Set distance to infinity for padded/invalid objects (use pre-allocated mask)
            torch.logical_and(
                self.non_occupied_object_indices >= 0,
                self.non_occupied_object_indices < self.num_objects_per_env,
                out=self._valid_mask
            )
            torch.where(self._valid_mask, self._distances, self._inf_tensor, out=self._distances)

            # Sort distances and get indices of nearest objects (use pre-allocated tensors)
            torch.sort(self._distances, dim=1, out=(self._sorted_distances, self._sorted_indices))

            # Take only the k nearest (use pre-allocated tensor)
            self._nearest_indices[:] = self._sorted_indices[:, :self.k_nearest]

            # Gather the nearest positions and orientations (use pre-allocated batch_indices)
            self.nearest_non_target_object_positions[:, :self.k_nearest] = self._non_target_positions[self._batch_indices, self._nearest_indices]
            self.nearest_non_target_object_orientations[:, :self.k_nearest] = self._non_target_orientations[self._batch_indices, self._nearest_indices]

            # Handle case where some environments have invalid nearest objects (use pre-allocated tensors)
            torch.lt(self._sorted_distances[:, :self.k_nearest], torch.inf, out=self._valid_nearest)

            # Create expanded invalid masks (use pre-allocated tensors)
            torch.logical_not(self._valid_nearest.unsqueeze(-1).expand(-1, -1, 3), out=self._invalid_mask_pos)
            torch.logical_not(self._valid_nearest.unsqueeze(-1).expand(-1, -1, 4), out=self._invalid_mask_ori)

            # Apply masks to zero out invalid entries
            self.nearest_non_target_object_positions[:, :self.k_nearest, :][self._invalid_mask_pos] = 0.0
            self.nearest_non_target_object_orientations[:, :self.k_nearest, :][self._invalid_mask_ori] = 0.0


        self.object_bboxes_wrt_world[:, :3] = transformation_apply(
            self.object_root_orientations, self.object_root_positions, self.object_bboxes[:, :3]
        )
        self.object_bboxes_wrt_world[:, 3:] = transformation_apply(
            self.object_root_orientations, self.object_root_positions, self.object_bboxes[:, 3:]
        )

        world_to_palm_rotation, world_to_palm_translation = transformation_inverse(
            self.allegro_hand_center_orientations, self.allegro_hand_center_positions
        )

        self.object_bboxes_wrt_palm[:, :3] = transformation_apply(
            world_to_palm_rotation, world_to_palm_translation, self.object_bboxes_wrt_world[:, :3]
        )
        self.object_bboxes_wrt_palm[:, 3:] = transformation_apply(
            world_to_palm_rotation, world_to_palm_translation, self.object_bboxes_wrt_world[:, 3:]
        )

        self.palm_orientations_wrt_object, self.palm_positions_wrt_object = compute_relative_pose(
            self.allegro_hand_center_orientations,
            self.allegro_hand_center_positions,
            self.object_root_orientations,
            self.object_root_positions,
        )

        self.fingertip_orientations_wrt_palm, self.fingertip_positions_wrt_palm = compute_relative_pose(
            self.fingertip_orientations,
            self.fingertip_positions,
            self.allegro_hand_center_orientations[:, None, :],
            self.allegro_hand_center_positions[:, None, :],
        )

        if add_noise:
            obj_pos_estimation_nosie = torch.clamp(
                torch.randn_like(self.object_root_positions.clone()) * np.sqrt(0.0004), -0.02, 0.02
            )
            obj_quat_estimation_noise = np.sqrt(2 / 57.3)

            self.observed_object_positions = self.object_root_positions.clone() + obj_pos_estimation_nosie
            self.observed_object_orientations = random_orientation_within_angle(
                self.object_root_orientations.size(0),
                self.device,
                self.object_root_orientations.clone(),
                obj_quat_estimation_noise,
            )
            self.observed_object_orientations_wrt_palm, self.observed_object_positions_wrt_palm = compute_relative_pose(
                self.observed_object_orientations,
                self.observed_object_positions,
                self.allegro_hand_center_orientations,
                self.allegro_hand_center_positions,
            )

        self.object_orientations_wrt_palm, self.object_positions_wrt_palm = compute_relative_pose(
            self.object_root_orientations,
            self.object_root_positions,
            self.allegro_hand_center_orientations,
            self.allegro_hand_center_positions,
        )

        self.object_positions_wrt_keypoints = self.keypoint_positions - self.object_root_positions[:, None, :]




        self.position_distances = self.object_positions_wrt_palm - self._r_target_object_positions_wrt_palm
        self.orientation_distances = quat_mul(
            self.object_orientations_wrt_palm, quat_conjugate(self._r_target_object_orientations_wrt_palm)
        )
        self.dof_distances = (
            self.allegro_hand_dof_positions[:, self.allegro_digits_actuated_dof_indices]
            - self._r_target_allegro_digits_actuated_dof_positions
        )

        if self.enable_contact_sensors:
            contact_forces = self.allegro_hand_net_contact_forces[:, self.force_sensor_rigid_body_indices, :]
            contact_forces = torch.norm(contact_forces, dim=-1)
            # binary contact sensor
            self.contact_forces = torch.where(contact_forces >= self.contact_sensor_threshold, 1.0, 0.0)
            self.fingertip_contact_forces = self.contact_forces[:, self.fingertip_contact_mask]

            # visualize
            # for (env_id, each_env_contacts) in enumerate(self.contact_forces):
            #     for (contact_idx, each_env_contact) in enumerate(each_env_contacts):
            #         self.gym.set_rigid_body_color(self.envs[env_id], 0, self.force_sensor_parent_rigid_body_indices[contact_idx], gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(each_env_contact, 0.0, 0.0))
            # import time
            # time.sleep(0.1)
        if self.enable_full_pointcloud_observation:
            self.obj_pointclouds_wrt_world = self.compute_object_pointclouds("current")
            self.target_obj_pointclouds_wrt_world = self.compute_object_pointclouds("target")

            self.object_pointclouds = self.obj_pointclouds_wrt_world
            self.pointclouds_wrt_palm = compute_relative_position(
                self.obj_pointclouds_wrt_world,
                self.allegro_hand_center_orientations[:, None, :],
                self.allegro_hand_center_positions[:, None, :],
            )
            self.object_pointclouds_wrt_palm = self.pointclouds_wrt_palm


        if self.enable_imagined_pointcloud_observation:
            self.imagined_pointclouds = self.compute_imagined_pointclouds("current")
            self.imagined_pointclouds_wrt_palm = compute_relative_position(
                self.imagined_pointclouds,
                self.allegro_hand_center_orientations[:, None, :],
                self.allegro_hand_center_positions[:, None, :],
            )



        if self.enable_rendered_pointcloud_observation:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            depth = torch.stack(self.camera_tensors).view(
                self.num_envs, self.num_cameras_per_env, self.camera_properties.height, self.camera_properties.width
            )

            pointclouds, mask = pointcloud_from_depth(
                depth,
                inv_view_matrix=self.camera_inv_view_matrices,
                proj_matrix=self.camera_proj_matrices,
                width=self.camera_properties.width,
                height=self.camera_properties.height,
                u=self.camera_u2,
                v=self.camera_v2,
            )
            corner_min, corner_max = self.render_pointcloud_bbox_corners
            mask = mask & (pointclouds > corner_min).all(dim=-1) & (pointclouds < corner_max).all(dim=-1)

            num_points_per_env = self.num_cameras_per_env * self.camera_properties.height * self.camera_properties.width
            pointclouds = pointclouds.view(self.num_envs, num_points_per_env, 3)
            mask = mask.view(self.num_envs, num_points_per_env)

            # change the layout of current pointclouds
            indices = torch.argsort(
                mask * torch.rand((self.num_envs, num_points_per_env), device=self.device), dim=1, descending=True
            )
            indices = indices.unsqueeze(-1).expand(-1, -1, 3)
            counts = torch.sum(mask, dim=1)
            pointclouds = pointclouds.gather(1, indices)

            if self.rendered_pointcloud_sample_method == "random":
                # random sampling
                location = torch.rand((self.num_envs, self.num_rendered_points), device=self.device)
                indices = torch.floor(location * counts.unsqueeze(-1)).long()
                indices = indices.unsqueeze(-1).expand(-1, -1, 3)
                rendered_pointclouds = pointclouds.gather(1, indices)
            else:
                # farthest point sampling
                maximum_rendered_candidates = self.num_rendered_points * self.rendered_pointcloud_multiplier
                pointclouds = pointclouds[:, : min(maximum_rendered_candidates, counts.max())]
                counts = torch.clamp(counts, max=maximum_rendered_candidates)
                rendered_pointclouds, _ = sample_farthest_points(pointclouds, counts, K=self.num_rendered_points)

            if (counts == 0).any():
                rendered_pointclouds[counts == 0] = 0.0

            if self.rendered_pointcloud_gaussian_noise:
                noise = (
                    torch.randn(rendered_pointclouds.shape, device=self.device)
                    * self.rendered_pointcloud_gaussian_noise_sigma
                )
                mask = (
                    torch.rand((self.num_envs, self.num_rendered_points, 1), device=self.device)
                    < self.rendered_pointcloud_gaussian_noise_ratio
                )
                noise *= mask
                rendered_pointclouds += noise

            self.rendered_pointclouds = rendered_pointclouds

            # import open3d as o3d

            # o3d_pointcloud = o3d.geometry.PointCloud()
            # o3d_pointcloud.points = o3d.utility.Vector3dVector(rendered_pointclouds[0].to("cpu").numpy())

            # origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            # o3d.visualization.draw_geometries([o3d_pointcloud, origin_axis])
            self.gym.end_access_image_tensors(self.sim)

        # compute tip-tip and tip-mid equidistant points - (CASE2023 Baseline)
        if self.method == "case":
            thtip_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], "rh_thtip")
            mfmid_index = self.gym.find_asset_rigid_body_index(
                self.gym_assets["current"]["robot"]["asset"], "rh_mfmiddle"
            )
            mftip_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], "rh_mftip")

            thtip_positions = self.allegro_hand_rigid_body_positions[:, thtip_index]
            mfmid_positions = self.allegro_hand_rigid_body_positions[:, mfmid_index]
            mftip_positions = self.allegro_hand_rigid_body_positions[:, mftip_index]

            alpha = (torch.arange(1, 4, device=self.device) / 4.0).reshape(1, 3, 1)
            tiptip_points = alpha * thtip_positions[:, None, :] + (1 - alpha) * mftip_positions[:, None, :]
            tipmid_points = alpha * thtip_positions[:, None, :] + (1 - alpha) * mfmid_positions[:, None, :]
            kpoint_positions = torch.cat([tiptip_points, tipmid_points], dim=1)
            kpoint_positions_wrt_object = compute_relative_position(
                kpoint_positions,
                self.object_root_orientations[:, None, :],
                self.object_root_positions[:, None, :],
            )
            self.kpoint_distances = point_to_mesh_distance(
                kpoint_positions_wrt_object,
                self.grasping_dataset._sdf_fields,
                self.occupied_mesh_indices,
            )

            fingertip_positions_wrt_object = compute_relative_position(
                self.fingertip_positions,
                self.object_root_orientations[:, None, :],
                self.object_root_positions[:, None, :],
            )
            self.fingertip_distances = point_to_mesh_distance(
                fingertip_positions_wrt_object,
                self.grasping_dataset._sdf_fields,
                self.occupied_mesh_indices,
            )

            norm_object_orientation = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            self.norm_object_orientation_wrt_palm = quat_mul(
                quat_conjugate(self.allegro_hand_center_orientations), norm_object_orientation
            )

    def __configure_specifications(self, specs: Dict, mdp_type: str) -> None:
        assert "__dim__" in specs, "spec must contain `__dim__`"
        assert mdp_type in ["observation", "action"], "mdp_type must be either `observation` or `action`"

        Spec = ObservationSpec if mdp_type == "observation" else ActionSpec

        dims: Dict[str, Union[str, int]] = specs.pop("__dim__")
        for name, value in dims.items():
            assert isinstance(value, int) or isinstance(value, str), "dim must be either int or str"
            dims[name] = value if isinstance(value, int) else getattr(self, value)

        _specs = []
        for name, info in specs.items():
            shape = info["shape"]

            if not isinstance(shape, omegaconf.listconfig.ListConfig):
                shape = [shape]

            shape = [dims[d] if isinstance(d, str) else d for d in shape]
            dim = int(np.prod(shape))

            _specs.append(Spec(name, dim, **info))
        return _specs

    def __configure_observation_specs(self, observation_specs: Dict) -> None:
        """Configure the observation specifications.

        All the observation specifications are stored in `self._observation_specs`

        Args:
            observation_specs (Dict): The observation specifications. (cfg["env"]["observation_specs"])
        """
        self._observation_specs = self.__configure_specifications(observation_specs, "observation")

    def __configure_action_specs(self, action_specs: Dict) -> None:
        """Configure the action specifications.

        All the action specifications are stored in `self._action_specs`

        Args:
            action_specs (Dict): The action specifications. (cfg["env"]["action_specs"])
        """
        self._action_specs = self.__configure_specifications(action_specs, "action")

    def export_observation_metainfo_frame(self) -> pd.DataFrame:
        """Export the observation metainfo as pandas dataframe.

        Returns:
            pd.DataFrame: The observation metainfo frame.
        """
        metainfo = self.export_observation_metainfo()
        for item in metainfo:
            item["tags"] = ",".join(item["tags"])
        return pd.DataFrame(metainfo)

    def export_observation_metainfo(self) -> List[Dict[str, Any]]:
        """Export the observation metainfo.

        Returns:
            List[Dict[str, Any]]: The observation metainfo.
        """
        metainfo = []
        current = 0
        for spec in self._observation_space:
            metainfo.append(
                {
                    "name": spec.name,
                    "dim": spec.dim,
                    "tags": spec.tags,
                    "start": current,
                    "end": current + spec.dim,
                }
            )
            current += spec.dim
        return metainfo

    def export_action_metainfo(self) -> List[Dict[str, Any]]:
        """Export the action metainfo.

        Returns:
            List[Dict[str, Any]]: The action metainfo.
        """
        metainfo = []
        current = 0
        for spec in self._action_space:
            metainfo.append(
                {
                    "name": spec.name,
                    "dim": spec.dim,
                    "start": current,
                    "end": current + spec.dim,
                }
            )
            current += spec.dim
        return metainfo

    def _get_observation_spec(self, name: str) -> ObservationSpec:
        """Get the specification of an observation.

        Args:
            name: The name of the observation.

        Returns:
            The specification of the observation.
        """
        for spec in self._observation_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Observation {name} not found.")

    def _get_observation_dim(self, name: str) -> int:
        """Get the dimension of an observation.

        Args:
            name: The name of the observation.

        Returns:
            The dimension of the observation.
        """
        return self._get_observation_spec(name).dim

    def _get_action_spec(self, name: str) -> ActionSpec:
        """Get the specification of an action.

        Args:
            name: The name of the action.

        Returns:
            The specification of the action.
        """
        for spec in self._action_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Action {name} not found.")

    def _get_action_dim(self, name: str) -> int:
        """Get the dimension of an action.

        Args:
            name: The name of the action.

        Returns:
            The dimension of the action.
        """
        return self._get_action_spec(name).dim

    def __configure_mdp_spaces(self) -> None:
        """Configure the observation, state and action spaces for the task.

        Define the scale and offset for each observation, state and action. Calculate the total number of observations,
        states and actions, and display the information to terminal.
        """
        # configure action space
        self.__configure_action_specs(self.cfg["env"]["actionSpecs"])
        self._action_space = [self._get_action_spec(name) for name in self.cfg["env"]["actionSpace"]]
        self.num_actions = sum([self._get_action_dim(name) for name in self.cfg["env"]["actionSpace"]])
        self.cfg["env"]["numActions"] = self.num_actions

        # configure observation space
        self.__configure_observation_specs(self.cfg["env"]["observationSpecs"])
        observation_space = self.cfg["env"]["observationSpace"]
        observation_space_extra = self.cfg["env"]["observationSpaceExtra"]
        observation_space_extra = [] if observation_space_extra is None else observation_space_extra

        num_observations = (
            sum([self._get_observation_dim(name) for name in observation_space]) * self.stack_frame_number
        )
        self.cfg["env"]["numObservations"] = num_observations
        self.cfg["env"]["numStates"] = self.cfg["env"]["numObservations"] * self.stack_frame_number

        self._observation_space = [self._get_observation_spec(name) for name in observation_space]

        # check if observation space extra already exists in observation space
        for name in observation_space_extra:
            if name in observation_space:
                warnings.warn(f"Observation {name} already exists in the observation space.")
        observation_space_extra = [name for name in observation_space_extra if name not in observation_space]
        observation_space_extra = observation_space + observation_space_extra

        self._observation_space_extra = [self._get_observation_spec(name) for name in observation_space_extra]
        self._required_attributes = [spec.attr for spec in self._observation_space_extra]

        if self.env_info_logging:
            print_observation_space(self._observation_space)
            print_action_space(self._action_space)

        # check imagined pointcloud observation
        if any([("imagined" in spec.tags and "pointcloud" in spec.tags) for spec in self._observation_space_extra]):
            if not self.enable_imagined_pointcloud_observation:
                warnings.warn("imagined pointcloud observation is enabled but not configured")
                warnings.warn("automatically overwrite `enable_imagined_pointcloud_observation` to `True`")
                self.enable_imagined_pointcloud_observation = True

        # check rendered pointcloud observation
        if any([("rendered" in spec.tags and "pointcloud" in spec.tags) for spec in self._observation_space_extra]):
            if not self.enable_rendered_pointcloud_observation:
                warnings.warn("rendered pointcloud observation is enabled but not configured")
                warnings.warn("automatically overwrite `enable_rendered_pointcloud_observation` to `True`")
                self.enable_rendered_pointcloud_observation = True

        # TODO: configure it from observation space
        self.pcl_obs = self.cfg["env"]["pclObs"]
        self.enable_full_pointcloud_observation = (
            "pointcloud_wrt_palm" in observation_space
            or "pclcontact" in self.reward_type
            or "stage" in self.curriculum_mode
            or ("no" in self.curriculum_mode and self.height_scale == 1.0)
            or self.pcl_obs
        )

        if any([("perfect" in spec.tags and "pointcloud" in spec.tags) for spec in self._observation_space_extra]):
            if not self.enable_full_pointcloud_observation:
                warnings.warn("perfect pointcloud observation is enabled but not configured")
                warnings.warn("automatically overwrite `enable_full_pointcloud_observation` to `True`")
                self.enable_full_pointcloud_observation = True

    def _create_ground_plane(self, static_friction: float = 1.0, dynamic_friction: float = 1.0) -> None:
        """Create a ground plane for the simulation.

        The ground plane is created using the `gymapi.PlaneParams` class.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        plane_params.static_friction = static_friction
        plane_params.dynamic_friction = dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def __define_table(self) -> Dict[str, Any]:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        asset = self.gym.create_box(
            self.sim, self._table_x_length, self._table_y_length, self._table_thickness, asset_options
        )


        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._table_pose)

        return {
            "asset": asset,
            "pose": pose,
            "name": "table",
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    def __define_contact_sensors(self, allegro_hand_asset: gymapi.Asset) -> None:
        """Configure the contact sensors.

        All the contact sensors are attached to the allegro Hand. The corresponding link names should start with `sensor_`.

        Args:
            allegro_hand_asset (gymapi.Asset): The allegro Hand asset to configure.
        """
        indices = []
        fingertip_indices = []
        parent_indices = []

        print("Contact sensors:")
        for name, index in self.gym.get_asset_rigid_body_dict(allegro_hand_asset).items():
            if name.startswith("sensor_"):
                indices.append(index)
                if "distal" in name:
                    fingertip_indices.append(index)
                print(f"- {name} ({index})")
        fingertip_contact_mask = [(i in fingertip_indices) for i in indices]

        assert len(indices) > 0, "No contact sensors found in the allegro Hand asset."
        self.force_sensor_rigid_body_indices = torch.tensor(indices).long().sort().values.to(self.device)

        self.force_sensor_names = []
        for i in self.force_sensor_rigid_body_indices:
            name = self.gym.get_asset_rigid_body_name(allegro_hand_asset, i)
            self.force_sensor_names.append(name)

            parent_name = name.replace("sensor", "rh")
            parent_indices.append(self.gym.find_asset_rigid_body_index(allegro_hand_asset, parent_name))

        self.force_sensor_parent_rigid_body_indices = torch.tensor(parent_indices).long().to(self.device)
        self.fingertip_contact_mask = torch.tensor(fingertip_contact_mask).bool().to(self.device)
        self.num_tactile_sensors = self.force_sensor_rigid_body_indices.size(0)
        # find same element in two lists
        self.ft_idx_in_all = [i for (i, index) in enumerate(indices) if index in fingertip_indices]

    @property
    def contact_states(self) -> torch.Tensor:
        """Compute contact states (tactile information) from force sensor data.

        For singulation task, we extract force magnitudes from fingertip force sensors.
        Returns a tensor of shape [num_envs, num_tactile_sensors] where num_tactile_sensors
        could be 4 (force magnitudes only) or 14 (extended tactile info).
        """
        if self.force_sensor_states is None or self.num_force_sensors == 0:
            # If no force sensors, return zeros
            return torch.zeros((self.num_envs, 14), device=self.device, dtype=torch.float)

        raise NotImplementedError("Contact states are not implemented")

    def __configure_robot_dof_indices(self, allegro_hand_asset: gymapi.Asset) -> None:
        """Configure the xArm6 + Allegro Hand DOFs.

        Args:
            allegro_hand_asset (gymapi.Asset): The xArm6 + Allegro Hand asset to configure.
        """
        dof_dict = self.gym.get_asset_dof_dict(allegro_hand_asset)

        actuated_dof_indices = []
        xarm_actuated_dof_indices = []
        allegro_actuated_dof_indices = []
        allegro_digits_actuated_dof_indices = []
        allegro_fingers_actuated_dof_indices = []
        allegro_thumb_actuated_dof_indices = []

        for name, index in dof_dict.items():
            if any([dof in name for dof in self._xarm_dof_names]):
                xarm_actuated_dof_indices.append(index)
            elif any([dof in name for dof in self._allegro_hand_dof_names]):
                allegro_actuated_dof_indices.append(index)
                allegro_digits_actuated_dof_indices.append(index)

                if any([dof in name for dof in self._allegro_fingers_dof_names]):
                    allegro_fingers_actuated_dof_indices.append(index)
                elif any([dof in name for dof in self._allegro_thumb_dof_names]):
                    allegro_thumb_actuated_dof_indices.append(index)

            actuated_dof_indices.append(index)

        def _torchify(indices: List[int]) -> torch.LongTensor:
            return torch.tensor(sorted(indices)).long().to(self.device)

        self.actuated_dof_indices = _torchify(actuated_dof_indices)
        self.xarm_actuated_dof_indices = _torchify(xarm_actuated_dof_indices)
        self.allegro_actuated_dof_indices = _torchify(allegro_actuated_dof_indices)
        self.allegro_digits_actuated_dof_indices = _torchify(allegro_digits_actuated_dof_indices)
        self.allegro_fingers_actuated_dof_indices = _torchify(allegro_fingers_actuated_dof_indices)
        self.allegro_thumb_actuated_dof_indices = _torchify(allegro_thumb_actuated_dof_indices)

    def __define_allegro_hand_with_arm(
        self, asset_name: str = "allegro Hand + xarm"
    ) -> Dict[str, Any]:
        """Define & load the allegro Hand + xarm asset.

        Args:
            asset_name (str, optional): Asset name for logging. Defaults to "allegro Hand + xarm".

        Returns:
            Dict[str, Any]: The configuration of the robot.
        """
        print(">>> Loading allegro Hand + xarm for current scene")
        config = {"name": "allegro_hand"}

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.1

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        if self.env_info_logging:
            print_asset_options(asset_options, asset_name)

        if self.enable_contact_sensors:
            if self.contact_sensor_fingertip_only:
                asset_filename = self._xarm_allegro_hand_right_asset_file.replace(".urdf", "_contact_fingertip.urdf")
            else:
                asset_filename = self._xarm_allegro_hand_right_asset_file.replace(".urdf", "_contact.urdf")
        else:
            asset_filename = self._xarm_allegro_hand_right_asset_file

        asset = self.gym.load_asset(self.sim, self._asset_root, asset_filename, asset_options)
        if self.env_info_logging:
            print_links_and_dofs(self.gym, asset, asset_name)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(asset)

        num_dofs = config["num_dofs"]

        # fmt: off
        mjcf_stiffness = [
            5.0, 5.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]
        mjcf_damping = [
            0.5, 0.5,
            0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1,
        ]
        mjcf_velocity = [
            100.0, 100.0,
            100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0,
        ]
        mjcf_effort = [
            4.7849998e00, 2.1750000e00,
            8.9999998e-01, 8.9999998e-01, 7.2450000e-01, 7.2450000e-01,
            8.9999998e-01, 8.9999998e-01, 7.2450000e-01, 7.2450000e-01,
            8.9999998e-01, 8.9999998e-01, 7.2450000e-01, 7.2450000e-01,
            8.9999998e-01, 8.9999998e-01, 8.9999998e-01, 7.2450000e-01, 7.2450000e-01,
            2.3722000e00, 1.4500000e00, 9.9000001e-01, 9.9000001e-01, 8.1000000e-01,
        ]
        # fmt: on

        dof_props = self.gym.get_asset_dof_properties(asset)
        hand_dof_idx = 0

        # set rigid-shape properties for allegro-hand
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape in rigid_shape_props:
            shape.friction = 3.0
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        for i in range(num_dofs):
            name = self.gym.get_asset_dof_name(asset, i)
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if name.endswith(".0"):
                dof_props["stiffness"][i] = 30
                dof_props["damping"][i] = 1
                dof_props["velocity"][i] = 3.0
                dof_props["effort"][i] = 5
                hand_dof_idx += 1
            else:
                dof_props["stiffness"][i] = 4000
                dof_props["damping"][i] = 80
                # dof_props["stiffness"][i] = 1e6
                # dof_props["damping"][i] = 1e2

        if self.env_info_logging:
            print_dof_properties(self.gym, asset, dof_props, asset_name)

        dof_lower_limits = [dof_props["lower"][i] for i in range(num_dofs)]
        dof_upper_limits = [dof_props["upper"][i] for i in range(num_dofs)]
        dof_init_positions = [0.0 for _ in range(num_dofs)]
        dof_init_velocities = [0.0 for _ in range(num_dofs)]

        # reset xarm initial dof positions
        for name, value in self._xarm_right_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value
        for name, value in self.allegro_hand_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value
            self._allegro_hand_predef_qpos[
                self.gym.find_asset_dof_index(asset, name) - 6
            ] = value  # substract 6 for xarm dofs

        config["limits"] = {}
        config["limits"]["lower"] = torch.tensor(dof_lower_limits).float().to(self.device)
        config["limits"]["upper"] = torch.tensor(dof_upper_limits).float().to(self.device)

        config["init"] = {}
        config["init"]["position"] = torch.tensor(dof_init_positions).float().to(self.device)
        config["init"]["velocity"] = torch.tensor(dof_init_velocities).float().to(self.device)

        if self.enable_contact_sensors:
            self.__define_contact_sensors(asset)
        self.__configure_robot_dof_indices(asset)

        # fmt: off
        close_dof_names = [
            "joint_2.0", "joint_3.0",  # finger 0 (index)
            "joint_6.0", "joint_7.0",  # finger 1 (middle)
            "joint_10.0", "joint_11.0",  # finger 2 (ring)
            "joint_14.0", "joint_15.0",  # thumb
        ]
        # fmt: on

        self.close_dof_indices = torch.tensor(
            [self.gym.find_asset_dof_index(asset, name) for name in close_dof_names],
            dtype=torch.long,
            device=self.device,
        )

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._xarm_right_init_position)
        pose.r = gymapi.Quat(*self._xarm_right_init_orientation)

        self.allegro_center_index = self.gym.find_asset_rigid_body_index(asset, self._allegro_hand_center_prim)
        self.allegro_palm_index = self.gym.find_asset_rigid_body_index(asset, self._allegro_hand_palm_prim)
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._fingertips]
        self.keypoint_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._keypoints]
        self.arm_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._arm_links]
        self.hand_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._hand_links]

        config["asset"] = asset
        config["pose"] = pose
        config["dof_props"] = dof_props

        print(">>> xArm6 + Allegro Hand loaded")
        return config

    def __define_object(self, dataset: str = "boxes") -> Dict[str, Any]:
        """Define & load objects for the current scene.

        For singulation task, we create a grid of boxes instead of loading dataset objects.

        Args:
            dataset (str, optional): Dataset type. Defaults to 'boxes'.

        Returns:
            Dict[str, Any]: The configuration of the objects.
        """
        return self.__create_box_grid()

    def __define_object_deprecated(self, dataset: str = "boxes") -> Dict[str, Any]:
        """Define & load objects for the current scene.

        For singulation task, we create a grid of boxes instead of loading dataset objects.

        Args:
            dataset (str, optional): Dataset type. Defaults to 'boxes'.

        Returns:
            Dict[str, Any]: The configuration of the objects.
        """
        print(">>> Loading objects for current scene")
        config = {}
        config["warehouse"] = []

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        # asset_options.override_com = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64

        # load assets to memory

        if self.resample_object:
            # resample to original distribution
            if self.manipulated_object_codes is None:
                object_codes = self.grasping_dataset.resample(self.num_envs * self.num_objects_per_env)
                self.manipulated_object_codes = object_codes
            else:
                object_codes = self.manipulated_object_codes
        else:
            # select the first-k objects
            object_codes = self.grasping_dataset.manipulated_codes

        loaded = {}
        for i, name in enumerate(object_codes):
            if name in loaded:
                cfg = config["warehouse"][loaded[name]].copy()
            else:
                loaded[name] = i
                asset_filename = os.path.join(dataset, name, "decomposed.urdf")
                asset = self.gym.load_asset(self.sim, self._asset_root, asset_filename, asset_options)

                # set rigid-shape properties
                rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
                for shape in rigid_shape_props:
                    shape.friction = 3.0
                self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

                cfg = {"name": name, "asset": asset}
                cfg["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
                cfg["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
            config["warehouse"].append(cfg)
        config["count"] = len(config["warehouse"])

        num_rigid_bodies = [cfg["num_rigid_bodies"] for cfg in config["warehouse"]]
        num_rigid_shapes = [cfg["num_rigid_shapes"] for cfg in config["warehouse"]]
        config["num_rigid_bodies"] = sum(sorted(num_rigid_bodies, reverse=True)[: self.num_objects_per_env])
        config["num_rigid_shapes"] = sum(sorted(num_rigid_shapes, reverse=True)[: self.num_objects_per_env])

        # define object poses (unused and occupied)
        unused_pose = gymapi.Transform()
        unused_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        occupied_pose = gymapi.Transform()
        if test_sim:
            occupied_pose.p = gymapi.Vec3(0.0, 0.2, 0.7)
        else:
            occupied_pose.p = gymapi.Vec3(0.0, 0.0, 0.7)

        num_objects_per_row = int(np.sqrt(self.num_objects_per_env))

        config["poses"] = []
        for i in range(self.num_objects_per_env):
            row, col = i // num_objects_per_row, i % num_objects_per_row

            x = unused_pose.p.x
            y = unused_pose.p.y
            z = unused_pose.p.z

            x += col * self.object_spacing
            y += row * self.object_spacing

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            config["poses"].append(pose)
        config["occupied_pose"] = occupied_pose

        print(">>> Objects loaded")
        return config

    def __create_box_grid(self) -> Dict[str, Any]:
        """Create a grid of boxes for singulation task.

        Returns:
            Dict[str, Any]: Configuration for the box grid
        """
        print(">>> Creating box grid for singulation task")

        config = {}
        config["warehouse"] = {
            "targ_obj": [],
            "surr_obj": [],
        }

        target_asset_options = gymapi.AssetOptions()
        target_asset_options.density = 500.0
        target_asset_options.convex_decomposition_from_submeshes = True
        target_asset_options.override_com = True
        target_asset_options.override_inertia = True
        
        surrounding_asset_options = gymapi.AssetOptions()
        surrounding_asset_options.density = 500.0
        surrounding_asset_options.convex_decomposition_from_submeshes = True
        surrounding_asset_options.override_com = True
        surrounding_asset_options.override_inertia = True
        surrounding_asset_options.disable_gravity = True
        surrounding_asset_options.fix_base_link = True

        target_box_asset = self.gym.create_box(self.sim, self._obj_width, self._obj_depth, self._obj_height, target_asset_options)
        surrounding_box_asset = self.gym.create_box(self.sim, self._obj_width, self._obj_depth, self._obj_height, surrounding_asset_options)

        _targ_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(target_box_asset)
        _surr_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(surrounding_box_asset)
        for shape in _targ_rigid_shape_props:
            shape.friction = 0.8
            shape.restitution = 0.1
        for shape in _surr_rigid_shape_props:
            shape.friction = 0.8
            shape.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(target_box_asset, _targ_rigid_shape_props)
        self.gym.set_asset_rigid_shape_properties(surrounding_box_asset, _surr_rigid_shape_props)

        config["warehouse"]["targ_obj"].append({
            "name": "target_box",
            "asset": target_box_asset,
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(target_box_asset),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(target_box_asset),
        })
        config["warehouse"]["surr_obj"].append({
            "name": "surrounding_box",
            "asset": surrounding_box_asset,
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(surrounding_box_asset),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(surrounding_box_asset),
        })
        config["count"] = len(config["warehouse"]["targ_obj"])

        num_rigid_bodies = [cfg_targ["num_rigid_bodies"] + cfg_surr["num_rigid_bodies"] for cfg_targ, cfg_surr in zip(config["warehouse"]["targ_obj"], config["warehouse"]["surr_obj"])]
        num_rigid_shapes = [cfg_targ["num_rigid_shapes"] + cfg_surr["num_rigid_shapes"] for cfg_targ, cfg_surr in zip(config["warehouse"]["targ_obj"], config["warehouse"]["surr_obj"])]
        config["num_rigid_bodies"] = sum(sorted(num_rigid_bodies, reverse=True)[: self.num_objects_per_env])
        config["num_rigid_shapes"] = sum(sorted(num_rigid_shapes, reverse=True)[: self.num_objects_per_env])

        config["poses"] = self.__generate_object_poses()

        print(f">>> Box grid created with {len(config['warehouse'])} box assets")
        return config

    def __generate_object_poses(self) -> List[gymapi.Transform]:
        """Generate poses for boxes in a grid pattern on the table.

        Returns:
            List[gymapi.Transform]: List of poses for each box
        """
        poses = []

        # Calculate grid center position on table
        table_center_x = self._table_pose[0]
        table_center_y = self._table_pose[1]
        table_top_z = self._table_pose[2] + (self._table_thickness + self._obj_height) / 2 + 1e-3

        # Calculate grid dimensions
        grid_width = self._grid_cols * self._obj_width + (self._grid_cols - 1) * self._obj_spacing
        grid_height = self._grid_rows * self._obj_depth + (self._grid_rows - 1) * self._obj_spacing

        # Starting position (top-left corner of grid)
        start_x = table_center_x - grid_width / 2 + self._obj_width / 2
        start_y = table_center_y - grid_height / 2 + self._obj_depth / 2

        # Generate poses for each box in the grid
        for i in range(self.num_objects_per_env):
            row = i // self._grid_cols
            col = i % self._grid_cols

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                start_x + col * (self._obj_width + self._obj_spacing),
                start_y + row * (self._obj_depth + self._obj_spacing),
                table_top_z
            )
            pose.r = gymapi.Quat(0, 0, 0, 1)  # No rotation

            poses.append(pose)

        return poses

    def __define_target_allegro_hand(self, asset_name: str = "Target allegro Hand") -> Dict[str, Any]:
        """Define & load the target allegro Hand.

        Args:
            asset_name (str, optional): Asset name for logging. Defaults to "Target allegro Hand".

        Returns:
            Dict[str, Any]: The configuration of the target allegro Hand.
        """
        print(">>> Loading allegro Hand for target scene")
        config = {"name": "target_allegro_hand"}

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        if self.env_info_logging:
            print_asset_options(asset_options, asset_name)

        asset = self.gym.load_asset(self.sim, self._asset_root, self._allegro_hand_right_asset_file, asset_options)
        if self.env_info_logging:
            print_links_and_dofs(self.gym, asset, asset_name)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(asset)

        dof_props = self.gym.get_asset_dof_properties(asset)
        for i in range(config["num_dofs"]):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 3.0
            dof_props["damping"][i] = 0.0
        if self.env_info_logging:
            print_dof_properties(self.gym, asset, dof_props, asset_name)

        self.target_allegro_center_index = self.gym.find_asset_rigid_body_index(asset, self._allegro_hand_center_prim)
        self.target_fingertip_indices = [
            self.gym.find_asset_rigid_body_index(asset, f"rh_{prim}") for prim in self._fingertips
        ]

        pose = gymapi.Transform()

        if self.save_video:
            pose.p = gymapi.Vec3(-0.4 + video_pose[0], 0.3 + video_pose[1], 0.8 + video_pose[2])
        else:
            pose.p = gymapi.Vec3(-0.4, 0.3, 0.8)

        pose.r = gymapi.Quat(0.0, -np.sqrt(0.5), np.sqrt(0.5), 0.0)

        config["asset"] = asset
        config["pose"] = pose
        config["dof_props"] = dof_props

        print(">>> Target allegro Hand loaded")

        return config

    def __define_visual_target_object(self, asset_name: str = "Visual Target Object") -> Dict[str, Any]:
        """Define a visual-only asset to represent the goal position in the environment.

        Args:
            asset_name (str, optional): Name for the asset. Defaults to "Visual Target Object".

        Returns:
            Dict[str, Any]: Configuration dictionary for the visual target object.
        """
        print(f">>> Loading {asset_name}")
        config = {"name": "visual_target_object"}

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True

        # asset = self.gym.create_box(self.sim, self._obj_width, self._obj_depth, self._obj_height, asset_options)
        asset = self.gym.create_sphere(self.sim, 0.02, asset_options)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.goal_position[0], self.goal_position[1], self.goal_position[2])   
        pose.r = gymapi.Quat(self.goal_orientation[0], self.goal_orientation[1], self.goal_orientation[2], self.goal_orientation[3])

        config["asset"] = asset
        config["pose"] = pose
        config["num_rigid_bodies"] = num_rigid_bodies
        config["num_rigid_shapes"] = num_rigid_shapes

        print(f">>> {asset_name} loaded")
        return config

    def __define_camera(self) -> None:
        """Define the cameras for the rendering."""
        if not self.enable_rendered_pointcloud_observation and not self.save_video:
            return

        self._camera_positions = [gymapi.Vec3(0.5, -0.1, 1)]
        self._camera_target_locations = [gymapi.Vec3(0.0, 0.0, 0.3)]

        assert len(self._camera_positions) == len(self._camera_target_locations)
        self.num_cameras_per_env = len(self._camera_positions)

        # allocate tensors for camera data
        self.cameras = [[] for _ in range(self.num_envs)]
        self.camera_tensors = []
        self.camera_positions = torch.zeros((self.num_envs, self.num_cameras_per_env, 3), device=self.device)
        self.camera_orientations = torch.zeros((self.num_envs, self.num_cameras_per_env, 4), device=self.device)
        self.camera_inv_view_matrices = torch.zeros((self.num_envs, self.num_cameras_per_env, 4, 4), device=self.device)
        self.camera_proj_matrices = torch.zeros((self.num_envs, self.num_cameras_per_env, 4, 4), device=self.device)

        # define camera properties
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.width = 1024
        self.camera_properties.height = 768
        self.camera_properties.enable_tensors = True

        # define related indices for pointcloud computation
        self.camera_u = torch.arange(0, self.camera_properties.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_properties.height, device=self.device)
        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing="ij")

        # define bounding box corners for pointcloud computation
        self.render_pointcloud_bbox_corners = (
            torch.tensor([-self._table_x_length / 2, -self._table_y_length / 2, 0.34], device=self.device),
            torch.tensor([self._table_x_length / 2, self._table_y_length / 2, 1.20], device=self.device),
        )

    def __create_box_grid_dataset(self, device=None) -> None:
        # Create simple box grid dataset for singulation task
        from .dataset import BoxGridDataset

        self.grasping_dataset = BoxGridDataset(
            grid_rows=self._grid_rows,
            grid_cols=self._grid_cols,
            grid_layers=self._grid_layers,
            box_width=self._obj_width,
            box_depth=self._obj_depth,
            box_height=self._obj_height,
            device=device,
        )

        self.num_categories = self.grasping_dataset._category_matrix.shape[1]



    def __reset_grasping_joint_indices(self) -> None:
        # if "target" in self.gym_assets and "robot" in self.gym_assets["target"]:
        #     asset = self.gym_assets["target"]["robot"]["asset"]
        # else:
        #     asset = self.__define_target_allegro_hand()["asset"]

        asset = self.gym_assets["target"]["robot"]["asset"]

        indices = [self.gym.find_asset_dof_index(asset, name) for name in self.grasping_dataset.dof_names]
        print("grasping dataset joints:", self.grasping_dataset.dof_names)
        self.grasping_joint_indices = torch.tensor(indices).long().to(self.device)

    def __reset_action_indices(self) -> None:
        (
            self.arm_trans_action_indices,
            self.arm_rot_action_indices,
            self.arm_roll_action_indices,
            self.hand_action_indices,
        ) = get_action_indices(self._action_space, device=self.device)

    def __create_sim_actor(
        self,
        env: gymapi.Env,
        config: Dict[str, Any],
        group: int,
        name: Optional[str] = None,
        pose: Optional[gymapi.Transform] = None,
        color: Optional[gymapi.Vec3] = None,
        actor_handle: Optional[bool] = False,
        filter:int = 0
    ) -> int:
        """Create an `Actor` in the simulator.

        Args:
            env (gymapi.Env): The environment to create the actor in.
            config (Dict[str, Any]): The configuration of the actor.
            group (int): The collision group of the actor.
            name (Optional[str], optional): The name of the actor. Defaults to None.
            pose (Optional[gymapi.Transform], optional): The pose of the actor. Defaults to None.
            color (Optional[gymapi.Vec3], optional): The color of the actor. Defaults to None.

        Returns:
            int: The index of the actor. (Domain: gymapi.DOMAIN_SIM)
        """
        asset = config.get("asset", None)
        name = name if name is not None else config["name"]
        pose = pose if pose is not None else config["pose"]
        assert asset is not None and name is not None and pose is not None

        self.aggregate_tracker.update(config["num_rigid_bodies"], config["num_rigid_shapes"])

        # create the actor
        actor = self.gym.create_actor(env, asset, pose, name, group, filter, 0)

        # set the dof properties if `dof_props` exists in the config
        dof_props = config.get("dof_props", None)
        if dof_props is not None:
            self.gym.set_actor_dof_properties(env, actor, dof_props)

        # set the color
        if color is not None:
            self.gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        else:
            # set the color of the contact sensors (blue by default)
            for name, index in self.gym.get_actor_rigid_body_dict(env, actor).items():
                if not name.startswith("sensor_"):
                    continue
                self.gym.set_rigid_body_color(
                    env, actor, index, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0.8)
                )

        if actor_handle:
            return self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM), actor
        else:
            return self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

    def compute_maximum_aggregate_bodies_and_shapes(self, gym_assets: Optional[Dict] = None) -> Tuple[int, int]:
        """Compute the maximum number of rigid bodies and shapes in the environment.

        fetch `num_rigid_bodies` and `num_rigid_shapes` from the `gym_assets` dict.
        Args:
            gym_assets (Optional[Dict], optional): The gym assets to compute. Defaults to None.
                if None, use `self.gym_assets`.

        Returns:
            Tuple[int, int]: The maximum number of rigid bodies and shapes.
        """
        max_aggregate_bodies, max_aggregate_shapes = 0, 0
        for i in range(self.num_envs):
            num_bodies, num_shapes = self.compute_aggregate_bodies_and_shapes(i, gym_assets)
            max_aggregate_bodies = max(max_aggregate_bodies, num_bodies)
            max_aggregate_shapes = max(max_aggregate_shapes, num_shapes)
        return max_aggregate_bodies, max_aggregate_shapes

    def compute_aggregate_bodies_and_shapes(self, env: int, gym_assets: Optional[Dict] = None) -> Tuple[int, int]:
        """Compute the number of rigid bodies and shapes in the environment.

        Args:
            env (int): The index of the environment.
            gym_assets (Optional[Dict], optional): The gym assets to compute. Defaults to None.
                if None, use `self.gym_assets`.

        Returns:
            Tuple[int, int]: The number of rigid bodies and shapes in the environment.
        """
        if gym_assets is None:
            gym_assets = self.gym_assets

        num_bodies, num_shapes = 0, 0

        num_bodies += gym_assets["current"]["robot"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["robot"]["num_rigid_shapes"]
        num_current_objects = gym_assets["current"]["objects"]["count"]
        
        num_bodies += gym_assets["current"]["objects"]["warehouse"]["targ_obj"][(env * self.num_objects_per_env) % num_current_objects]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["objects"]["warehouse"]["targ_obj"][(env * self.num_objects_per_env) % num_current_objects]["num_rigid_shapes"]

        for i in range(1, self.num_objects_per_env):
            cur = (env * self.num_objects_per_env + i) % num_current_objects

            num_bodies += gym_assets["current"]["objects"]["warehouse"]["surr_obj"][cur]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["objects"]["warehouse"]["surr_obj"][cur]["num_rigid_shapes"]

        num_bodies += gym_assets["current"]["table"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["table"]["num_rigid_shapes"]
        
        num_bodies += gym_assets["current"]["visual_target_object"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["visual_target_object"]["num_rigid_shapes"]

        # num_bodies += gym_assets["current"]["objects"]["warehouse"][0]["num_rigid_bodies"]
        # num_shapes += gym_assets["current"]["objects"]["warehouse"][0]["num_rigid_shapes"]

        return num_bodies, num_shapes

    def _create_envs(self, num_envs: int, spacing: float, num_objects_per_env: int = 1):
        print(">>> Setting up %d environments" % num_envs)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(num_envs))

        print(">>> Defining gym assets")

        self.gym_assets["current"]["robot"] = self.__define_allegro_hand_with_arm()
        self.gym_assets["current"]["objects"] = self.__define_object()
        self.gym_assets["current"]["table"] = self.__define_table()
        self.gym_assets["current"]["visual_target_object"] = self.__define_visual_target_object()

        # self.gym_assets["target"]["robot"] = self.__define_target_allegro_hand()

        self.__define_camera()

        print(">>> Done defining gym assets")

        max_aggregate_bodies, max_aggregate_shapes = self.compute_maximum_aggregate_bodies_and_shapes()

        self.envs = []
        self.cameras_handle = []

        allegro_hand_indices = []
        table_indices = []
        visual_target_object_indices = []
        object_indices = [[] for _ in range(num_envs)]
        object_encodings = [[] for _ in range(num_envs)]
        object_names = [[] for _ in range(num_envs)]
        occupied_object_indices = []
        surr_object_indices = []
        non_occupied_object_indices = [[] for _ in range(num_envs)]
        scene_object_indices = [[] for _ in range(num_envs)]
        occupied_object_indices_per_env = [random.randint(0, self.num_objects_per_env - 1) for _ in range(num_envs)]

        print(">>> Creating environments")
        print("    - max_aggregate_bodies: ", max_aggregate_bodies)
        print("    - max_aggregate_shapes: ", max_aggregate_shapes)

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.aggregate_tracker.reset()

            if self.aggregate_mode != 0:
                num_bodies, num_shapes = self.compute_aggregate_bodies_and_shapes(i)
                agg_success = self.gym.begin_aggregate(env, max_aggregate_bodies, max_aggregate_shapes, True)
                if not agg_success:
                    raise RuntimeError("begin_aggregate failed")

            # add allegro hand to the environment
            actor_index, actor_handle = self.__create_sim_actor(
                env, self.gym_assets["current"]["robot"], i, actor_handle=True
            )
            allegro_hand_indices.append(actor_index)

            poses = self.gym_assets["current"]["objects"]["poses"]
            surr_obj_cur_idx = 0
            for k in range(self.num_objects_per_env):
                is_target = (k == occupied_object_indices_per_env[i])
                cfg = self.gym_assets["current"]["objects"]["warehouse"]["targ_obj"][k % len(self.gym_assets["current"]["objects"]["warehouse"]["targ_obj"])] if is_target else self.gym_assets["current"]["objects"]["warehouse"]["surr_obj"][k % len(self.gym_assets["current"]["objects"]["warehouse"]["surr_obj"])]
                pose = poses[k]

                surr_obj_color = gymapi.Vec3(0.9, 0.0, 0.0)
                targ_obj_color = gymapi.Vec3(0.9, 0.9, 0.9)

                if is_target:
                    actor_index = self.__create_sim_actor(env, cfg, i, "targ_obj", pose, color=targ_obj_color)
                else:
                    surr_obj_name = f"sur_obj_{surr_obj_cur_idx}"
                    surr_obj_cur_idx += 1
                    actor_index = self.__create_sim_actor(env, cfg, i, surr_obj_name, pose, color=surr_obj_color)


                object_indices[i].append(actor_index)
                object_names[i].append(cfg["name"])
                object_encodings[i].append(k)

                if is_target:
                    occupied_object_indices.append(actor_index)  # global actor index for root_states access
                else:
                    non_occupied_object_indices[i].append(k)  #relative index within environment
                    surr_object_indices.append(actor_index)

                scene_object_indices[i].append(k)  # relative index within environment

            # add table to the environment
            actor_index, actor_handle = self.__create_sim_actor(
                env, self.gym_assets["current"]["table"], -1, actor_handle=True, color=gymapi.Vec3(0.0, 0.0, 0.0)
            )
            table_indices.append(actor_handle)

            # add visual target object to the environment
            actor_index, actor_handle = self.__create_sim_actor(
                env, self.gym_assets["current"]["visual_target_object"], i + self.num_envs, actor_handle=True, color=gymapi.Vec3(0.6, 0.72, 0.98)
            )
            visual_target_object_indices.append(actor_handle)


            if self.enable_rendered_pointcloud_observation or self.save_video:
                for k in range(self.num_cameras_per_env):
                    camera = self.gym.create_camera_sensor(env, self.camera_properties)
                    self.cameras_handle.append(camera)

                    self.gym.set_camera_location(
                        camera, env, self._camera_positions[k], self._camera_target_locations[k]
                    )
                    image = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera, gymapi.IMAGE_DEPTH)
                    image = gymtorch.wrap_tensor(image)

                    view_matrix = self.gym.get_camera_view_matrix(self.sim, env, camera)
                    proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env, camera)

                    view_matrix = torch.tensor(view_matrix).to(self.device)
                    proj_matrix = torch.tensor(proj_matrix).to(self.device)
                    inv_view_matrix = torch.inverse(view_matrix)

                    origin: gymapi.Vec3 = self.gym.get_env_origin(env)
                    inv_view_matrix[3][0] -= origin.x
                    inv_view_matrix[3][1] -= origin.y
                    inv_view_matrix[3][2] -= origin.z

                    # the `inv_view_matrix` is a transposed version of transformation matrix
                    # the quaternions are in the order of (w, x, y, z) in pytorch3d, need to be converted to (x, y, z, w)
                    camera_position = inv_view_matrix[3, :3]
                    camera_orientation = matrix_to_quaternion(inv_view_matrix[:3, :3].T)
                    camera_orientation = torch.cat([camera_orientation[1:], camera_orientation[:1]])

                    self.cameras[i].append(camera)
                    self.camera_tensors.append(image)
                    self.camera_inv_view_matrices[i, k] = inv_view_matrix
                    self.camera_proj_matrices[i, k] = proj_matrix
                    self.camera_positions[i, k] = camera_position
                    self.camera_orientations[i, k] = camera_orientation
                    if self.env_info_logging:
                        print("view_matrix: ", view_matrix)
                        print("proj_matrix: ", proj_matrix)
                        print("image.shape: ", image.shape)

            # if i==0:
            #     self.test_pcl(env, 0)

            if self.aggregate_mode != 0:
                agg_success = self.gym.end_aggregate(env)
                if not agg_success:
                    raise RuntimeError("end_aggregate failed")

                assert self.aggregate_tracker.aggregate_bodies == num_bodies
                assert self.aggregate_tracker.aggregate_shapes == num_shapes

            self.envs.append(env)

        print(f">>> Done creating {num_envs} environments")

        allegro_hand = self.gym.find_actor_handle(env, "allegro_hand")
        self.allegro_hand_index = self.gym.get_actor_index(env, allegro_hand, gymapi.DOMAIN_ENV)
        
        # Object Rigid Body Index Tracking
        self.target_object_rigid_body_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.surr_object_rigid_body_indices = torch.zeros((num_envs, self.max_non_targets), dtype=torch.long, device=self.device)
        
        for i in range(num_envs):
            env = self.envs[i]
            
            # Get target object rigid body index
            target_obj_handle = self.gym.find_actor_handle(env, "targ_obj")
            target_object_actor = self.gym.get_actor_index(env, target_obj_handle, gymapi.DOMAIN_ENV)
            target_rb_index = self.gym.get_actor_rigid_body_index(
                env, target_object_actor, 0, gymapi.DOMAIN_SIM
            )
            self.target_object_rigid_body_indices[i] = target_rb_index
            
            # Get surrounding object rigid body indices surr_object_indices to [env_id, max_non_targets]
            for j in range(self.max_non_targets):
                surr_obj_handle = self.gym.find_actor_handle(env, f"sur_obj_{j}")
                surr_object_actor = self.gym.get_actor_index(env, surr_obj_handle, gymapi.DOMAIN_ENV)
                surr_rb_index = self.gym.get_actor_rigid_body_index(
                    env, surr_object_actor, 0, gymapi.DOMAIN_SIM
                )
                self.surr_object_rigid_body_indices[i, j] = surr_rb_index

        # define start and end indices for allegro hand DOFs to create contiguous slices
        self.allegro_hand_dof_start = self.gym.get_actor_dof_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_dof_end = self.allegro_hand_dof_start + self.gym_assets["current"]["robot"]["num_dofs"]
        self.allegro_hand_indices = torch.tensor(allegro_hand_indices).long().to(self.device)
        self.allegro_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_rigid_body_end = (
            self.allegro_hand_rigid_body_start + self.gym_assets["current"]["robot"]["num_rigid_bodies"]
        )


        self.table_indices = torch.tensor(table_indices).long().to(self.device)
        self.visual_target_object_indices = torch.tensor(visual_target_object_indices).long().to(self.device)

        self.object_indices = torch.tensor(object_indices).long().to(self.device)
        self.object_names = object_names
        self.object_encodings = torch.tensor(object_encodings).long().to(self.device)

        self.occupied_object_indices = (torch.tensor(occupied_object_indices).long().to(self.device))  # (env_id) - global actor indices
        self.occupied_object_relative_indices = (torch.tensor(occupied_object_indices_per_env).long().to(self.device))  # (env_id) - relative indices 0 to num_objects_per_env-1
        self.non_occupied_object_indices = (torch.tensor(non_occupied_object_indices).long().to(self.device))  # (env_id, max_non_targets)
        self.surr_object_indices = (torch.tensor(surr_object_indices).long().to(self.device))  # (env_id, max_non_targets)
        self.scene_object_indices = (torch.tensor(scene_object_indices).long().to(self.device))  # (env_id, object_id)
        # fmt off

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == "z" else 1

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"])

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def compute_observations(self, reset_env_ids: Optional[torch.LongTensor] = None) -> None:
        """Compute the observations.

        The observations required for the task training are stored in `self.obs_buf`.

        Args:
            reset_env_ids (Optional[torch.LongTensor], optional): The indices of the environments to reset. Defaults to None.
                corresponding envs will be reset to the initial state if self.stack_frame_number > 1.
        """
        observation_dict: OrderedDict = self.retrieve_observation_dict()

        # only fetch the observations required for the task training
        observations: torch.Tensor = torch.cat(
            [observation_dict[spec.name].reshape(self.num_envs, -1) for spec in self._observation_space], dim=-1
        )

        if self.stack_frame_number > 1:
            if len(self.frames) == 0:
                self.frames.extend([observations.clone() for _ in range(self.stack_frame_number)])
            else:
                self.frames.append(observations.clone())
                if reset_env_ids is not None:
                    for frame in self.frames:
                        frame[reset_env_ids] = observations[reset_env_ids]

            self.obs_buf[:] = torch.cat(list(self.frames), 1)
        else:
            self.obs_buf[:] = observations

    def retrieve_observation_dict(self) -> OrderedDict:
        """Retrieve the observation dict.

        Returns:
            OrderedDict[str, torch.Tensor]: The observation dict.
        """
        self._refresh_sim_tensors()

        observations = OrderedDict()
        for spec in self._observation_space_extra:
            observation: torch.Tensor = getattr(self, spec.attr)

            if "dof" in spec.tags and "position" in spec.tags:
                observation = normalize(
                    observation,
                    self.gym_assets["current"]["robot"]["limits"]["lower"],
                    self.gym_assets["current"]["robot"]["limits"]["upper"],
                )
            elif "velocity" in spec.tags:
                observation = observation * self.velocity_observation_scale
            elif "orientation" in spec.tags:
                observation = quat_to_6d(observation)

            observations[spec.name] = observation

            if add_noise:
                if "object_position_wrt_palm" == spec.name:
                    observations[spec.name] = self.observed_object_positions_wrt_palm.clone()
                if "object_orientation_wrt_palm" == spec.name:
                    observations[spec.name] = self.observed_object_orientations_wrt_palm.clone()
        return observations

    def compute_fingertip_to_obj_center_reward(self):
        """Compute the reward based on the distance between the fingertip and the object center.

        occupied_object_init_root_positions: (num_envs, 3)
        fingertip_positions: (num_envs, num_fingertips, 3)
        """
        fingertip_to_obj_center_dist = (
            (self.object_root_positions.unsqueeze(1) - self.fingertip_positions).norm(dim=-1, p=2).mean(-1)
        )
        if "negft2oc" in self.reward_type:
            self.ft2oc_rew_scaled = -fingertip_to_obj_center_dist * 0.1
            # self.ft2oc_rew_scaled = - torch.exp(2.0 * torch.clamp(fingertip_to_obj_center_dist-self.obj_max_length * 0.5, 0, None)) * 0.1
        else:
            self.ft2oc_rew_scaled = (
                self.part_reward_scale
                * 10
                * self.obj_max_length
                * 0.5
                / (fingertip_to_obj_center_dist + self.obj_max_length * 0.5)
            )
        self.extras["ft2oc"] = fingertip_to_obj_center_dist.clone()
        self.extras["ft2oc_rew"] = self.ft2oc_rew_scaled.clone()

    def compute_ori_reward(self, mutual=False):
        """Compute the reward based on the distance between the object orientation and the target orientation."""
        # if self.env_mode == "orn":
        #     quat_diff = quat_mul(self.object_root_orientations, quat_conjugate(self.object_targets))
        # elif self.env_mode == "relpose" or self.env_mode == "relposecontact" or self.env_mode == "pgm":
        #     quat_diff = quat_mul(self.object_orientations_wrt_palm, quat_conjugate(self.object_targets[:, 3:7]))
        # self.rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        # print(self.rot_dist)
        # print(quat_diff_rad(self.object_orientations_wrt_palm.clone(), self._r_target_object_orientations_wrt_palm.clone()))

        # self.rot_dist = quat_diff_rad(
        #     self.object_orientations_wrt_palm.clone(), self._r_target_object_orientations_wrt_palm.clone()
        # )
        self.rot_dist = quat_diff_rad(self.object_orientations_wrt_palm, self._r_target_object_orientations_wrt_palm)

        if self.relative_part_reward:
            no_prev_dist_ids = torch.where(self.prev_rot_dist == -1)[0]
            self.prev_rot_dist[no_prev_dist_ids] = self.rot_dist[no_prev_dist_ids].clone()
            rot_rew = (self.prev_rot_dist - self.rot_dist) / (self._max_xarm_endeffector_rot_vel * self.dt)
            self.prev_rot_dist = self.rot_dist.clone()
        else:
            rot_rew = 1.0 / (torch.abs(self.rot_dist) + self.rot_eps)

        self.extras["rot_dist"] = self.rot_dist.clone()

        if negative_part_reward:
            rot_rew = -1.0 / rot_rew
        if mutual:
            return rot_rew
        self.rot_rew_scaled = rot_rew * self.rot_reward_scale * self.part_reward_scale

    def compute_pos_reward(self, mutual=False):
        """Compute the reward based on the distance between the object position and the target position."""
        # self.pos_dist = F.pairwise_distance(self.object_positions_wrt_palm, self.object_targets[:, :3])

        # print(self.pos_dist)
        # print(F.pairwise_distance(self.object_positions_wrt_palm, self._r_target_object_positions_wrt_palm))

        self.pos_dist = F.pairwise_distance(self.object_positions_wrt_palm, self._r_target_object_positions_wrt_palm)

        if self.relative_part_reward:
            no_prev_dist_ids = torch.where(self.prev_pos_dist == -1)[0]
            self.prev_pos_dist[no_prev_dist_ids] = self.pos_dist[no_prev_dist_ids].clone()
            pos_rew = (self.prev_pos_dist - self.pos_dist) / (self._max_xarm_endeffector_pos_vel * self.dt)
            self.prev_pos_dist = self.pos_dist.clone()
        else:
            pos_rew = (1.0 / trans_scale) / (torch.abs(self.pos_dist) + self.rot_eps / trans_scale)

        self.extras["pos_dist"] = self.pos_dist.clone()

        if negative_part_reward:
            pos_rew = -1.0 / pos_rew
        if mutual:
            return pos_rew
        self.pos_rew_scaled = pos_rew * self.tran_reward_scale * self.part_reward_scale

    def compute_contact_dist(self, obj_pcl: torch.Tensor, hand_pcl: torch.Tensor) -> torch.Tensor:
        """Compute the minimum distance between each point in the object pointcloud and the hand pointcloud.

        Args:
            obj_pcl (Tensor): The object pointcloud. (num_envs, num_object_points, 3)
            hand_pcl (Tensor): The hand pointcloud. (num_envs, num_hand_points, 3)

        Returns:
            Tensor: The minimum distance. (num_envs, num_object_points)
        """

        if batch_size >= self.num_envs:
            contract_dist = torch.cdist(obj_pcl, hand_pcl).min(dim=-1)[0]
        else:
            contract_dist = torch.zeros(self.num_envs, self.num_object_points, dtype=torch.float, device=self.device)
            for i in range(int(np.ceil(self.num_envs / batch_size))):
                start, end = i * batch_size, min((i + 1) * batch_size, self.num_envs)
                contract_dist[start:end] = torch.cdist(obj_pcl[start:end], hand_pcl[start:end]).min(dim=-1)[0]
        return contract_dist

    def compute_contact_match(self, obj_pcl: torch.Tensor, hand_pcl: torch.Tensor) -> torch.LongTensor:
        """Compute the index of the hand pointcloud that is closest to each point in the object pointcloud.

        Args:
            obj_pcl (Tensor): The object pointcloud. (num_envs, num_object_points, 3)
            hand_pcl (Tensor): The hand pointcloud. (num_envs, num_hand_points, 3)

        Returns:
            Tensor: The indices. (num_envs, num_object_points)
        """
        if batch_size >= self.num_envs:
            contact_indices = torch.cdist(obj_pcl, hand_pcl).min(dim=-1)[1]
        else:
            contact_indices = torch.zeros(self.num_envs, self.num_object_points, dtype=torch.long, device=self.device)
            for i in range(int(np.ceil(self.num_envs / batch_size))):
                start, end = i * batch_size, min((i + 1) * batch_size, self.num_envs)
                contact_indices[start:end, :] = torch.cdist(obj_pcl[start:end], hand_pcl[start:end]).min(dim=-1)[1]
        return contact_indices

    def compute_contact_map(self, pcl_type):
        if pcl_type == "current":
            hand_pcl = self.compute_imagined_pointclouds("current")
            if "pclcontactmatch" in self.reward_type:
                contact_idx = self.compute_contact_match(self.obj_pointclouds_wrt_world, hand_pcl)
            else:
                contact_dist = self.compute_contact_dist(self.obj_pointclouds_wrt_world, hand_pcl)
        elif pcl_type == "target":
            target_hand_pcl = self.compute_imagined_pointclouds("target")
            if "pclcontactmatch" in self.reward_type:
                contact_idx = self.compute_contact_match(self.target_obj_pointclouds_wrt_world, target_hand_pcl)
            else:
                contact_dist = self.compute_contact_dist(self.target_obj_pointclouds_wrt_world, target_hand_pcl)

        if "pclcontactmatch" in self.reward_type:
            # env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
            # ii = env_ids.reshape(-1, 1).repeat(1, 128).reshape(-1)
            # jj = contact_idx.reshape(-1)
            # contact_map = torch.zeros(self.num_envs, self.num_object_points, dtype=torch.float, device=self.device)
            # contact_map[ii, jj] = 1
            contact_map = contact_idx
        else:
            contact_map = 1 - 2 * (torch.sigmoid(contact_dist) - 0.5)

        # if pcl_type == "target":
        #     self.vis_contact_map(contact_map, target_hand_pcl)
        return contact_map

    def vis_contact_map(self, contact_map, target_hand_pcl):
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.target_obj_pointclouds_wrt_world.cpu().numpy()[0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_hand_pcl.cpu().numpy()[0])

        if "pclcontactmatch" in self.reward_type:
            # color = np.zeros([contact_map[0].size(0),3])
            # for idx in contact_map[0]:
            #     color[idx,0] = 1.0
            color = []
            for cmap_value in contact_map[0]:
                color.append([cmap_value.cpu().numpy(), 0, 0])
            color = np.concatenate(color).reshape(-1, 3)
            obj_pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            color = []
            cmap = normalize(contact_map[0], torch.min(contact_map[0]), torch.max(contact_map[0]))
            for cmap_value in cmap:
                color.append([cmap_value.cpu().numpy(), 0, 0])
            color = np.concatenate(color).reshape(-1, 3)
            obj_pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([obj_pcd, pcd])

    def compute_contact_reward(self, mutual=False):
        # fingerjoint error
        self.fj_dist = F.pairwise_distance(
            self.allegro_hand_dof_positions[:, self.allegro_digits_actuated_dof_indices],
            self._r_target_allegro_digits_actuated_dof_positions,
        )
        # print(self.target_allegro_hand_dof_positions[:, self.actuated_dof_indices])
        # self.fj_dist = (
        #     self.allegro_hand_dof_positions[:, self.allegro_digits_actuated_dof_indices] - self.object_targets[:, 7:]
        # ).norm(p=1, dim=1) / len(self.allegro_actuated_dof_indices)
        self.extras["fj_dist"] = self.fj_dist.clone()

        if "fjcontact" in self.reward_type:
            # fingerjoint error
            if high_thumb_reward:
                finger_dof_dis = F.pairwise_distance(
                    self.allegro_hand_dof_positions[:, self.allegro_fingers_actuated_dof_indices],
                    self._r_target_allegro_fingers_actuated_dof_positions,
                )
                thumb_dof_dis = F.pairwise_distance(
                    self.allegro_hand_dof_positions[:, self.allegro_thumb_actuated_dof_indices],
                    self._r_target_allegro_thumb_actuated_dof_positions,
                )
                self.contact_dist = torch.sqrt(
                    finger_dof_dis * finger_dof_dis * 9 / 13 + thumb_dof_dis * thumb_dof_dis * 9 / 5
                )
            else:
                self.contact_dist = F.pairwise_distance(
                    self.allegro_hand_dof_positions[:, self.allegro_digits_actuated_dof_indices],
                    self._r_target_allegro_digits_actuated_dof_positions,
                )

            if self.relative_part_reward:
                no_prev_dist_ids = torch.where(self.prev_contact_dist == -1)[0]
                self.prev_contact_dist[no_prev_dist_ids] = self.contact_dist[no_prev_dist_ids].clone()
                contact_rew = (self.prev_contact_dist - self.contact_dist) / (self.dof_speed_scale * self.dt)
                self.prev_contact_dist = self.contact_dist.clone()
            else:
                contact_rew = 10.0 * self.contact_eps / (torch.abs(self.contact_dist) + self.contact_eps)
            # self.contact_dist = (
            #     self.allegro_hand_dof_positions[:, self.allegro_digits_actuated_dof_indices] - self.object_targets[:, 7:]
            # ).norm(p=1, dim=1) / len(self.allegro_actuated_dof_indices)
            self.extras["contact_dist"] = self.contact_dist.clone()

            if negative_part_reward:
                contact_rew = -1.0 / contact_rew
            if mutual:
                return contact_rew

            if self.relative_part_reward:
                prox_pos_dist = torch.tensor(0.265, device=self.device)
                prox_rot_dist = torch.tensor(1.0, device=self.device)
                prox_pos_coff = 1 - torch.min(self.pos_dist, prox_pos_dist) / prox_pos_dist
                prox_rot_coff = 1 - torch.min(self.rot_dist, prox_rot_dist) / prox_rot_dist
            else:
                prox_pos_coff = 1.0
                prox_rot_coff = 1.0

            self.contact_rew_scaled = (
                prox_pos_coff * prox_rot_coff * contact_rew * self.rot_reward_scale * self.part_reward_scale
            )
        elif "pclcontact" in self.reward_type:
            contact_map = self.compute_contact_map(pcl_type="current")
            target_contact_map = self.compute_contact_map(pcl_type="target")
            if "pclcontactmatch" in self.reward_type:
                self.contact_dist = torch.sum(target_contact_map != contact_map, 1) / hand_pcl_num
            else:
                self.contact_dist = F.pairwise_distance(contact_map, target_contact_map)
            self.extras["contact_dist"] = self.contact_dist.clone()
            contact_rew = 10.0 * self.contact_eps / (torch.abs(self.contact_dist) + self.contact_eps)
            if negative_part_reward:
                contact_rew = -1.0 / contact_rew
            if mutual:
                return contact_rew
            self.contact_rew_scaled = contact_rew * self.contact_reward_scale * self.part_reward_scale

    def compute_mutual_reward(self):
        if self.env_mode == "relpose":
            rot_rew = self.compute_ori_reward(mutual=True)
            pos_rew = self.compute_pos_reward(mutual=True)
            rot_idx = self.rot_eps / torch.max(self.rot_dist, torch.tensor(self.rot_eps, device=self.device))
            pos_idx = (self.rot_eps / trans_scale) / torch.max(
                self.pos_dist, torch.tensor(self.rot_eps / trans_scale, device=self.device)
            )

            self.rot_rew_scaled = rot_rew * pos_idx * self.part_reward_scale
            self.pos_rew_scaled = pos_rew * rot_idx * self.part_reward_scale
        elif self.env_mode == "relposecontact" or self.env_mode == "pgm":
            rot_rew = self.compute_ori_reward(mutual=True)
            pos_rew = self.compute_pos_reward(mutual=True)
            contact_rew = self.compute_contact_reward(mutual=True)

            if negative_part_reward:
                rot_dist_scale = (self.rot_dist + self.rot_eps) / self.rot_eps
                pos_dist_scale = self.pos_dist + self.rot_eps / trans_scale
                if "fjcontact" in self.reward_type:
                    contact_dist_scale = (self.contact_dist + self.contact_eps) / self.contact_eps
                elif "pclcontact" in self.reward_type:
                    contact_dist_scale = (self.contact_dist + self.contact_eps) / self.contact_eps
                rot_idx = torch.max(torch.max(pos_dist_scale, rot_dist_scale), contact_dist_scale)
                pos_idx = torch.max(torch.max(pos_dist_scale, rot_dist_scale), contact_dist_scale)
                contact_idx = torch.max(torch.max(pos_dist_scale, rot_dist_scale), contact_dist_scale)
            else:
                rot_dist_scale = self.rot_eps / (self.rot_dist + self.rot_eps)
                pos_dist_scale = (self.rot_eps / trans_scale) / (self.pos_dist + self.rot_eps / trans_scale)
                if "fjcontact" in self.reward_type:
                    contact_dist_scale = self.contact_eps / (self.contact_dist + self.contact_eps)
                elif "pclcontact" in self.reward_type:
                    contact_dist_scale = self.contact_eps / (self.contact_dist + self.contact_eps)
                rot_idx = torch.min(torch.min(pos_dist_scale, rot_dist_scale), contact_dist_scale)
                pos_idx = torch.min(torch.min(pos_dist_scale, rot_dist_scale), contact_dist_scale)
                contact_idx = torch.min(torch.min(pos_dist_scale, rot_dist_scale), contact_dist_scale)

            # if "curr" in self.reward_type:
            #     thres = 1.2
            #     close_env_id = torch.where(
            #         (torch.abs(self.rot_dist) <= success_tolerance * thres)
            #         & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale) * thres),
            #         torch.ones_like(self.reset_buf),
            #         torch.zeros_like(self.reset_buf),
            #     )
            #     contact_rew *= close_env_id

            self.rot_rew_scaled = rot_rew * rot_idx * self.part_reward_scale
            self.pos_rew_scaled = pos_rew * pos_idx * self.part_reward_scale * self.tran_reward_scale
            self.contact_rew_scaled = contact_rew * contact_idx * self.part_reward_scale

    def compute_succ_reward(self):
        if self.env_mode == "orn" or test_rel:
            self.succ_rew = torch.where(
                (torch.abs(self.rot_dist) <= success_tolerance),
                torch.ones_like(self.reset_buf),
                torch.zeros_like(self.reset_buf),
            )
        elif self.env_mode == "relpose":
            self.succ_rew = torch.where(
                (torch.abs(self.rot_dist) <= success_tolerance)
                & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale)),
                torch.ones_like(self.reset_buf),
                torch.zeros_like(self.reset_buf),
            )
        elif self.env_mode == "relposecontact" or self.env_mode == "pgm":
            if "pclcontactonly" in self.reward_type or "pclcontactmatch" in self.reward_type:
                self.succ_rew = torch.where(
                    (torch.abs(self.contact_dist) <= self.contact_eps),
                    torch.ones_like(self.reset_buf),
                    torch.zeros_like(self.reset_buf),
                )
            elif "pclcontact" in self.reward_type:
                self.succ_rew = torch.where(
                    (torch.abs(self.rot_dist) <= success_tolerance)
                    & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale))
                    & (torch.abs(self.contact_dist) <= self.contact_eps),
                    torch.ones_like(self.reset_buf),
                    torch.zeros_like(self.reset_buf),
                )
            else:
                if self.env_mode == "pgm":
                    if self.height_scale == 0:
                        self.succ_rew = torch.where(
                            (torch.abs(self.rot_dist) <= success_tolerance)
                            & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale))
                            & (torch.abs(self.fj_dist) <= self.contact_eps),
                            torch.ones_like(self.reset_buf),
                            torch.zeros_like(self.reset_buf),
                        )
                        if self.mode == "eval" and local_test:
                            self.pose_succ = torch.where(
                                (torch.abs(self.rot_dist) <= success_tolerance)
                                & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale)),
                                torch.ones_like(self.reset_buf),
                                torch.zeros_like(self.reset_buf),
                            )
                    else:
                        if self.enable_full_pointcloud_observation:
                            # compare lowest point of object pointcloud with table height
                            lifted = torch.min(self.obj_pointclouds_wrt_world[:, :, 2], dim=1)[0] >= (
                                self._table_thickness / 2 + self._table_pose[2] + 0.005
                            )
                        else:
                            lifted = (
                                self.object_root_positions[:, 2]
                                >= self.occupied_object_init_root_positions[:, 2] + height_success_tolerance
                            )
                        self.succ_rew = torch.where(
                            (torch.abs(self.rot_dist) <= success_tolerance)
                            & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale))
                            & (torch.abs(self.fj_dist) <= self.contact_eps)
                            & lifted,
                            torch.ones_like(self.reset_buf),
                            torch.zeros_like(self.reset_buf),
                        )
                else:
                    self.succ_rew = torch.where(
                        (torch.abs(self.rot_dist) <= success_tolerance)
                        & (torch.abs(self.pos_dist) <= (success_tolerance / trans_scale))
                        & (torch.abs(self.fj_dist) <= self.contact_eps),
                        torch.ones_like(self.reset_buf),
                        torch.zeros_like(self.reset_buf),
                    )

        self.succ_rew_scaled = self.succ_rew * self.reach_goal_bonus

        if self.mode == "eval" and local_test:
            # pose_succ_envs = (self.pose_succ == 1).nonzero(as_tuple=False).squeeze(-1)
            # self.set_table_color(pose_succ_envs, color=[0.0,1.0,0.0])
            succ_envs = (self.succ_rew == 1).nonzero(as_tuple=False).squeeze(-1)
            self.set_table_color(succ_envs, color=[1.0, 0.0, 0.0])
            self.render()

    def set_table_color(self, env_ids, color=[0, 0, 0]):
        for succ_env_id in env_ids:
            self.gym.set_rigid_body_color(
                self.envs[succ_env_id], self.table_indices[succ_env_id], 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color)
            )

    def compute_action_reward(self, actions):
        if self.action_penalty_scale < 0:
            action_penalty = torch.sum(actions**2, dim=-1)
            self.action_penalty_scaled = action_penalty * self.action_penalty_scale
        elif self.wrist_action_penalty_scale < 0:
            action_penalty = torch.sum(self.allegro_dof_speeds[:2] ** 2, dim=-1)
            self.action_penalty_scaled = action_penalty * self.wrist_action_penalty_scale
        elif self.arm_action_penalty_scale < 0:
            ur_action = torch.cat([self.eef_translation, self.eef_rotation], dim=1)
            action_penalty = torch.sum(ur_action**2, dim=-1)
            self.action_penalty_scaled = action_penalty * self.arm_action_penalty_scale
        else:
            action_penalty = torch.sum(actions**2, dim=-1)
            self.action_penalty_scaled = action_penalty * 0

    def compute_reach_reward_(self):
        """Compute reach reward - reward for reaching the target object."""
        # max(d_closest - d, 0) d is between mean position for fingertips and target object
        self.keypoints_to_obj_dist = torch.mean(
            self.keypoint_positions - self.object_root_positions.unsqueeze(1), dim=1
        ).norm(dim=1)
        if not hasattr(self, "keypoints_to_obj_dist_min"):
            self.keypoints_to_obj_dist_min = torch.ones_like(self.keypoints_to_obj_dist) * 0.3
        self.reach_rew = torch.clip(self.keypoints_to_obj_dist_min - self.keypoints_to_obj_dist, min=0) * (~self.picked).float()
        self.reach_rew_scaled = self.reach_rew * 50
        self.extras["reach_rew"] = self.reach_rew_scaled.clone()
        self.keypoints_to_obj_dist_min = torch.min(
            self.keypoints_to_obj_dist_min, self.keypoints_to_obj_dist
        )
        self.extras["keypoints_to_obj_dist_min"] = self.keypoints_to_obj_dist_min.clone()

    def compute_reach_reward(self):
        """Compute reach reward - reward for reaching the target object."""
        # max(d_closest - d, 0) d is between mean position for fingertips and target object
        surface_offset = torch.tensor([0.0, 0.0, 0.125], device=self.device)
        object_root_positions_with_offset = compute_offset_point_world(
            self.object_root_positions,
            self.object_root_orientations,
            surface_offset
        )
        # self.fingertips_to_obj_dist = torch.mean(
        #     self.fingertip_positions - object_root_positions_with_offset.unsqueeze(1), dim=1
        # ).norm(dim=1)
        self.fingertips_to_obj_dist = (self.index_fingertip_positions - object_root_positions_with_offset).norm(dim=1)
        if not hasattr(self, "fingertips_to_obj_dist_min"):
            self.fingertips_to_obj_dist_min = torch.ones_like(self.fingertips_to_obj_dist) * 0.3
        self.reach_rew = torch.clip(self.fingertips_to_obj_dist_min - self.fingertips_to_obj_dist, min=0) * (~self.picked).float()
        self.reach_rew_scaled = self.reach_rew * 50
        self.extras["reach_rew"] = self.reach_rew_scaled.clone()
        self.fingertips_to_obj_dist_min = torch.min(
            self.fingertips_to_obj_dist_min, self.fingertips_to_obj_dist
        )
        self.extras["fingertips_to_obj_dist_min"] = self.fingertips_to_obj_dist_min.clone()

    def compute_pick_reward(self):
        """Compute pick reward - reward for picking the target object."""
        # pick = (1 - 1_picked) * h_t + r_picked

        self.delta_obj_height = self.object_root_positions[:, 2] - self.occupied_object_init_root_positions[:, 2]
        self.picked_curr = self.delta_obj_height > 0.12
        picked = self.picked | self.picked_curr
        newly_picked = ~self.picked & picked
        newly_picked_bonus = newly_picked * 350

        self.pick_rew = torch.clip((1 - picked.float()) * self.delta_obj_height * 20, min=0) + newly_picked_bonus
        
        self.pick_rew = self.pick_rew * (self.y_displacement > 0.0).float()
        
        self.pick_rew_scaled = self.pick_rew * (1)
        
        self.picked = picked

        self.extras["pick_rew"] = self.pick_rew_scaled.clone()
        self.extras["picked"] = picked.clone()
        self.extras["delta_obj_height"] = self.delta_obj_height.clone()

        # satisfy_contact = torch.where(
        #     (torch.abs(self.rot_dist) <= height_scale * success_tolerance)
        #     & (torch.abs(self.pos_dist) <= height_scale * (success_tolerance / trans_scale))
        #     & (torch.abs(self.fj_dist) <= height_scale * self.contact_eps),
        #     torch.ones_like(self.reset_buf),
        #     torch.zeros_like(self.reset_buf),
        # )
        # if "deltaheight" in self.reward_type:
        #     self.hand_delta_height = self.endeffector_states[:, 2] - self.prev_endeffector_states[:, 2]
        #     self.height_rew_scaled = (
        #         satisfy_contact * self.hand_delta_height * self.reach_goal_bonus * self.height_scale
        #     )
        # elif "zheight" in self.reward_type and self.ur_control_type == "osc":
        #     self.height_rew_scaled = (
        #         satisfy_contact
        #         * (1 + self.actions[:, self.arm_trans_action_indices[2]])
        #         * self.height_reward_scale
        #         * self.height_scale
        #     )

    def compute_targ_reward(self):
        """Compute target reward - reward for reaching the target state."""
        # 1_picked * max(d_closest - d_target, 0) + r_succ d is between mean position for target object's target pos and target object
        if not hasattr(self, "goal_position"):
            # self.goal_position = torch.tensor([0.0, 0.5, 0.75], device=self.device, dtype=torch.float)
            self.goal_position = torch.tensor([0.0, 0.2, 1.0], device=self.device, dtype=torch.float)
        self.goal_position_dist = torch.norm(
            self.goal_position.unsqueeze(0) - self.object_root_positions, dim=1
        )
        # print(self.goal_position_dist[0])
        if not hasattr(self, "goal_position_dist_min"):
            self.goal_position_dist_min = torch.ones_like(self.goal_position_dist) * 0.50
        delta = self.goal_position_dist_min - self.goal_position_dist
        clipped_delta = torch.maximum(delta, torch.zeros_like(delta))

        # self.targ_rew = torch.where(
        #     self.goal_position_dist < 0.075,
        #     self.picked.float() * clipped_delta * 250 + self.reach_goal_bonus,
        #     self.picked.float() * clipped_delta * 250,
        # ) * (self.y_displacement > 0.0).float() * (self.picked).float()
        
        self.targ_rew = clipped_delta * 250 * (self.picked).float() * (self.y_displacement > 0.0).float()
        
        
        self.targ_rew_scaled = self.targ_rew * (1)
        self.near_goal = (self.goal_position_dist_min <= 0.075).float()
        self.goal_position_dist_min = torch.min(self.goal_position_dist_min, self.goal_position_dist)
        self.extras["near_goal"] = self.near_goal.clone()
        self.extras["targ_rew"] = self.targ_rew_scaled.clone()
        self.extras["goal_position_dist_min"] = self.goal_position_dist_min.clone()

    def compute_reorient_obj_reward(self):
        if self.env_mode == "pgm":
            self.nominal_dist = quat_diff_rad(self.object_root_orientations, self._object_nominal_orientation)

            if self.relative_part_reward:
                no_prev_dist_ids = torch.where(self.prev_nominal_dist == -1)[0]
                self.prev_nominal_dist[no_prev_dist_ids] = self.nominal_dist[no_prev_dist_ids].clone()
                nominal_rew = 500 * (self.prev_nominal_dist - self.nominal_dist) / torch.pi
                self.prev_nominal_dist = self.nominal_dist.clone()
            else:
                nominal_rew = 1.0 / (torch.abs(self.nominal_dist) + self.rot_eps)

            self.extras["nominal_dist"] = self.nominal_dist.clone()
            self.nominal_rew_scaled = nominal_rew
            self.extras["nominal_rew"] = self.nominal_rew_scaled.clone()

    def compute_similarity_reward(self):
        curr_endeffector_positions = self.endeffector_positions.clone()
        curr_endeffector_orientations = self.endeffector_orientations.clone()
        curr_endeffector_orientations_euler = torch.stack(get_euler_xyz(curr_endeffector_orientations), dim=1)
        curr_allegro_actuated_dof_positions = self.allegro_hand_dof_positions[:, self.allegro_actuated_dof_indices].clone()

        curr_endeffector_orientations_norm = torch.arccos(torch.cos(curr_endeffector_orientations_euler))
        prev_endeffector_orientations_norm = torch.arccos(torch.cos(self.prev_endeffector_orientations_euler))

        diff_endeffector_positions = curr_endeffector_positions - self.prev_endeffector_positions
        diff_endeffector_orientations = curr_endeffector_orientations_norm - prev_endeffector_orientations_norm
        diff_allegro_actuated_dof_positions = (
            curr_allegro_actuated_dof_positions - self.prev_allegro_actuated_dof_positions
        )

        delta_arm_pos_state = diff_endeffector_positions
        delta_arm_rot_state = diff_endeffector_orientations
        delta_hand_state = diff_allegro_actuated_dof_positions

        arm_pos_similarity = (
            delta_arm_pos_state
            * self.action_gf[:, self.arm_trans_action_indices]
            / (abs(self.action_gf[:, self.arm_trans_action_indices]) + 1e-5)
        ) / (self._max_xarm_endeffector_pos_vel * self.dt)
        arm_rot_similarity = (
            delta_arm_rot_state
            * self.action_gf[:, self.arm_rot_action_indices]
            / (abs(self.action_gf[:, self.arm_rot_action_indices]) + 1e-5)
        ) / (self._max_xarm_endeffector_rot_vel * self.dt)
        hand_similarity = (
            delta_hand_state
            * self.action_gf[:, self.hand_action_indices]
            / (abs(self.action_gf[:, self.hand_action_indices]) + 1e-5)
        ) / (self.dof_speed_scale * self.dt)

        similarity_reward = (
            torch.sum(torch.cat([arm_pos_similarity, arm_rot_similarity, hand_similarity], -1), -1) / self.num_actions
        )
        if "decay_similarity" in self.reward_type:
            decrease_ratio = 1 - self.progress_buf / self.max_episode_length
        elif "dist_similarity" in self.reward_type:
            rot_dist_scale = self.rot_eps / (self.rot_dist + self.rot_eps)
            pos_dist_scale = (self.rot_eps / trans_scale) / (self.pos_dist + self.rot_eps / trans_scale)
            decrease_ratio = 1 - torch.min(rot_dist_scale, pos_dist_scale)
        elif "dedi_similarity" in self.reward_type:
            rot_dist_scale = self.rot_eps / (self.rot_dist + self.rot_eps)
            pos_dist_scale = (self.rot_eps / trans_scale) / (self.pos_dist + self.rot_eps / trans_scale)
            dist_decrease_ratio = 1 - torch.min(rot_dist_scale, pos_dist_scale)
            decay_decrease_ratio = 1 - self.progress_buf / self.max_episode_length
            decrease_ratio = decay_decrease_ratio * dist_decrease_ratio
        else:
            decrease_ratio = 1
        self.similarity_reward_scaled = similarity_reward * self.similarity_reward_scale * decrease_ratio
        self.extras["similarity_reward"] = self.similarity_reward_scaled.clone()
        # TODO smaller change no need to consider as diff
        self.extras["diff_direction"] = torch.sum(
            torch.sign(torch.cat([arm_pos_similarity, arm_rot_similarity, hand_similarity], -1)) < 0 
            * (abs(torch.cat([arm_pos_similarity, arm_rot_similarity, hand_similarity], -1)) > 0.01), -1,
        )
        # arm pos diff direction
        self.extras["arm_pos_diff_direction"] = torch.sum(
            torch.sign(arm_pos_similarity) < 0 * (abs(arm_rot_similarity) > 0.01), -1
        ).to(torch.float32)
        # arm rot diff direction
        self.extras["arm_rot_diff_direction"] = torch.sum(
            (torch.sign(arm_rot_similarity) < 0) * (abs(arm_rot_similarity) > 0.01), -1
        ).to(torch.float32)
        # hand diff direction
        self.extras["hand_diff_direction"] = torch.sum(
            (torch.sign(hand_similarity) < 0) * (abs(hand_similarity) > 0.01), -1
        ).to(torch.float32)

    def compute_manipulability_penalty(self):
        J = torch.linalg.det(self.j_eef)
        self.max_J = torch.max(0.15 * J, self.max_J)
        manipulability_penalty = 1 - 2 / (1 + ((torch.where(J < self.max_J, J, self.max_J)) / self.max_J) ** 3)
        self.manipulability_penalty_scaled = manipulability_penalty * self.manipulability_penalty_scale
        self.extras["manipulability_penalty"] = self.manipulability_penalty_scaled.clone()

    def compute_tilt_reward(self):
        """Compute tilt reward - reward for tilting the object toward the target rotation (-45 degrees around x-axis)."""
        from .torch_utils import get_euler_xyz

        current_euler = torch.stack(get_euler_xyz(self.object_root_orientations), dim=1)

        target_x_rotation = self.goal_rotation_x
        x_rotation_error = torch.abs(current_euler[:, 0] - target_x_rotation)

        self.tilt_reward = torch.exp(-x_rotation_error / 0.1)
        self.tilt_reward_scaled = self.tilt_reward * self.tilt_reward_scale

        self.extras["tilt_reward"] = self.tilt_reward_scaled.clone()
        self.extras["x_rotation_error"] = x_rotation_error.clone()

    def compute_slide_reward(self):
        """Compute slide reward - reward for moving the object forward along z-axis."""

        self.y_displacement = self.object_root_positions[:, 1] - self.occupied_object_init_root_positions[:, 1]
        if not hasattr(self, "y_displacement_max"): 
            self.y_displacement_max = torch.zeros_like(self.y_displacement)
        delta = self.y_displacement - self.y_displacement_max
        clipped_delta = torch.maximum(delta, torch.zeros_like(delta))

        target_y_displacement = self.goal_translation_y
        y_error = torch.abs(self.y_displacement - target_y_displacement)

        # Reward inversely proportional to distance from target
        # ln(40 * (-y_error + 1.25))
        # self.slide_reward = torch.log(16 * (-y_error + 1.25))
        # self.slide_reward = torch.clamp(-200 * (y_error - 0.05) * (y_error + 0.05), min=0)
        self.slide_reward = clipped_delta * 100
        
        self.slided = torch.maximum(self.y_displacement_max, self.y_displacement) > self.goal_translation_y
        
        
        self.slide_reward_scaled = self.slide_reward * self.slide_reward_scale
        
        self.y_displacement_max = torch.max(self.y_displacement_max, self.y_displacement)

        self.extras["slide_reward"] = self.slide_reward_scaled.clone()
        self.extras["y_displacement"] = self.y_displacement.clone()
        # self.extras["y_error"] = y_error.clone()

    def compute_neighbor_pos_penalty(self):
        # pos_diff = self.surr_object_root_positions - self.prev_surr_object_root_positions

        pos_diff = self.surr_object_root_positions - self.prev_surr_object_root_positions
        pos_displacement_per_object = torch.norm(pos_diff, dim=2, p=2)
        target_pos = self.object_root_positions  # [num_envs, 3] 
        surr_pos = self.surr_object_root_positions  # [num_envs, num_surr_objects, 3]
        
        distances_to_target = torch.norm(
            surr_pos - target_pos.unsqueeze(1), dim=2, p=2
        )  # [num_envs, num_surr_objects]
        
        sigma = 0.15  
        distance_weights = torch.exp(-distances_to_target ** 2 / (2 * sigma ** 2))
        
        pos_threshold = 0.002  
        significant_displacement = torch.clamp(
            pos_displacement_per_object - pos_threshold, min=0.0
        )
        
        weighted_pos_penalty_per_object = distance_weights * significant_displacement
        total_pos_penalty = torch.sum(weighted_pos_penalty_per_object, dim=1)  # [num_envs]
        
        self.neighbor_pos_penalty_scale = getattr(self, 'neighbor_pos_penalty_scale', 50.0)  # 默认缩放
        self.neighbor_pos_penalty = total_pos_penalty
        self.neighbor_pos_penalty_scaled = self.neighbor_pos_penalty * self.neighbor_pos_penalty_scale
        
        self.extras["neighbor_pos_penalty"] = self.neighbor_pos_penalty_scaled.clone()
        self.extras["neighbor_pos_displacement"] = torch.sum(pos_displacement_per_object, dim=1).clone()
        self.extras["avg_distance_to_target"] = torch.mean(distances_to_target, dim=1).clone()
    
    def compute_neighbor_rot_penalty(self):
        rot_diff_per_object_rad = quat_diff_rad_normalized(self.surr_object_root_orientations, self.prev_surr_object_root_orientations)
        
        target_pos = self.object_root_positions
        surr_pos = self.surr_object_root_positions
        distances_to_target = torch.norm(
            surr_pos - target_pos.unsqueeze(1), dim=2, p=2
        )
        
        sigma = 0.15  
        distance_weights = torch.exp(-distances_to_target ** 2 / (2 * sigma ** 2))
        
        rot_threshold = 0.05  # ~3
        significant_rotation = torch.clamp(
            rot_diff_per_object_rad - rot_threshold, min=0.0
        )
        
        weighted_rot_penalty_per_object = distance_weights * significant_rotation
        total_rot_penalty = torch.sum(weighted_rot_penalty_per_object, dim=1)
        
        self.neighbor_rot_penalty_scale = getattr(self, 'neighbor_rot_penalty_scale', 20.0)
        self.neighbor_rot_penalty = total_rot_penalty
        self.neighbor_rot_penalty_scaled = self.neighbor_rot_penalty * self.neighbor_rot_penalty_scale
        
        self.extras["neighbor_rot_penalty"] = self.neighbor_rot_penalty_scaled.clone()
        self.extras["neighbor_rot_diff_rad"] = torch.sum(rot_diff_per_object_rad, dim=1).clone()
        
    def compute_contact_based_safety_penalty(self):
        """Compute safety penalty based on contact forces - more immediate than displacement"""
        
        surr_contact_magnitudes = torch.clamp(torch.norm(self.surr_object_contact_forces, dim=2, p=2), max=300.0) 

        target_pos = self.object_root_positions
        surr_pos = self.surr_object_root_positions
        distances_to_target = torch.norm(surr_pos - target_pos.unsqueeze(1), dim=2, p=2)
        
        sigma = 0.15
        distance_weights = torch.exp(-distances_to_target ** 2 / (2 * sigma ** 2))
        
        # Contact force threshold (ignore small contact forces)
        contact_threshold = 1
        significant_contacts_mask = (surr_contact_magnitudes - contact_threshold) > 0.0
        weighted_contact_penalty = distance_weights * significant_contacts_mask.float() * surr_contact_magnitudes
        total_contact_penalty = -(1 - torch.exp(torch.max(weighted_contact_penalty, dim=1).values / 50))

        self.contact_penalty_scale = getattr(self, 'contact_penalty_scale', 10.0)
        self.contact_safety_penalty = total_contact_penalty * self.contact_penalty_scale
        # print(f"total_contact_penalty: {torch.max(self.contact_safety_penalty)}")
        
        # Debug
        self.extras["contact_safety_penalty"] = self.contact_safety_penalty.clone()
        self.extras["total_surr_contact_force"] = torch.sum(surr_contact_magnitudes, dim=1).clone()
        self.extras["target_contact_force"] = torch.norm(self.target_object_contact_forces, dim=1, p=2).clone()

    def compute_safety_penalty(self):

        self.compute_neighbor_pos_penalty()
        self.compute_neighbor_rot_penalty()
        self.compute_contact_based_safety_penalty()
        total_safety_penalty = (
            self.neighbor_pos_penalty_scaled + 
            self.neighbor_rot_penalty_scaled + 
            self.contact_safety_penalty
        )
        
        extreme_threshold = 0.1
        extreme_penalty = torch.where(
            self.neighbor_pos_penalty > extreme_threshold,
            100.0,
            0.0
        )
        
        self.safety_penalty = total_safety_penalty
        self.extras["safety_penalty"] = self.safety_penalty.clone()
        self.extras["extreme_disturbance"] = (self.neighbor_pos_penalty > extreme_threshold).float().clone()

    def compute_singulation_success_reward(self):
        """Compute success reward based on reaching the goal pose (position + rotation)."""
        # Check position success: z-displacement within tolerance
        y_displacement = self.object_root_positions[:, 1] - self.occupied_object_init_root_positions[:, 1]
        position_success = torch.abs(y_displacement - self.goal_translation_y) <= self.goal_tolerance_position

        # Check rotation success: x-rotation within tolerance
        from .torch_utils import get_euler_xyz
        current_euler = torch.stack(get_euler_xyz(self.object_root_orientations), dim=1)
        rotation_error = torch.abs(current_euler[:, 0] - self.goal_rotation_x)
        rotation_success = rotation_error <= self.goal_tolerance_rotation

        # Overall success: both position and rotation criteria met
        self.singulation_success = position_success & rotation_success

        # Convert to reward
        self.succ_rew = self.singulation_success.float()
        self.succ_rew_scaled = self.succ_rew * self.reach_goal_bonus

        self.extras["singulation_success"] = self.succ_rew.clone()
        self.extras["position_success"] = position_success.float()
        self.extras["rotation_success"] = rotation_success.float()

    def compute_done(self, is_success):
        if not test_sim:
            if self.env_mode == "pgm":
                fall_env_mask = (
                    (self.object_root_positions[:, 2] < self._table_pose[2] - 0.1)
                )
                arm_contact_mask = (self.arm_contact_forces.norm(p=2, dim=2) > 1).any(dim=1)

                failed_env_ids = (fall_env_mask | arm_contact_mask).nonzero(as_tuple=False).squeeze(-1)
            else:
                fall_env_mask = (
                    (self.object_root_positions[:, 2] < self.allegro_hand_center_positions[:, 2] - 0.2)
                )
                arm_contact_mask = (self.arm_contact_forces.norm(p=2, dim=2) > 1).any(dim=1)

                failed_env_ids = (fall_env_mask | arm_contact_mask).nonzero(as_tuple=False).squeeze(-1)
                
            if self.max_consecutive_successes > 0:
                self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
                self.reset_buf = torch.where(self.successes >= self.max_consecutive_successes, 1, self.reset_buf)

            self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, 1, self.reset_buf)
            
            self.reset_buf[failed_env_ids] = 1

        # success
        succ_env_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        self.reset_buf[succ_env_ids] = 1
        self.successes[succ_env_ids] = 1

        self.done_successes[failed_env_ids] = 0
        self.done_successes[succ_env_ids] = 1

        if "height" in self.reward_type:
            self.extras["final_object_height"] = self.delta_obj_height[
                self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            ].clone()
        self.extras["success_num"] = torch.sum(self.successes).unsqueeze(-1)

    def compute_reward(self, actions: Tensor) -> None:
        # self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, 1, self.reset_buf)
        # self.done_successes[self.reset_buf.nonzero(as_tuple=False).squeeze(-1)] = 0

        # if "tilt" in self.reward_type:
        #     self.compute_tilt_reward()
        if "slide" in self.reward_type:
            self.compute_slide_reward()
        # if "neighbor" in self.reward_type:
        #     self.compute_neighbor_stability_penalty()
        # if "stability" in self.reward_type:
        #     self.compute_stability_penalty()

        self.compute_reach_reward()
        self.compute_pick_reward()
        self.compute_targ_reward()
        
        # self.compute_neighbor_pos_penalty()
        # self.compute_neighbor_rot_penalty()
        
        self.near_goal_steps += self.near_goal.to(torch.int32)
        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success
        self.successes += is_success
        self.reset_goal_buf[:] = goal_resets
        
        self.extras["near_goal_steps"] = self.near_goal_steps.clone()
        self.extras["successes"] = self.successes.clone()

        # self.compute_action_reward(actions)
        # self.extras["action_penalty"] = self.action_penalty_scaled.clone()
        
        bonus_rew = self.near_goal * (self.reach_goal_bonus / self.success_steps)


        self.rew_buf[:] = (
            self.reach_rew_scaled + self.pick_rew_scaled + self.targ_rew_scaled + bonus_rew
        )
        self.task_reward = self.rew_buf.clone()
        
        self.compute_safety_penalty()
        
        self.rew_buf[:] = self.task_reward - self.safety_penalty
        
        self.extras["task_reward"] = self.task_reward.clone()
        self.extras["safety_ratio"] = (self.safety_penalty / (torch.abs(self.task_reward) + 1e-6)).clone()

        # Add singulation-specific rewards
        if "tilt" in self.reward_type:
            self.rew_buf[:] += self.tilt_reward_scaled
        if "slide" in self.reward_type:
            self.rew_buf[:] += self.slide_reward_scaled
        if "neighbor" in self.reward_type:
            self.rew_buf[:] += self.neighbor_stability_penalty_scaled
        if "stability" in self.reward_type:
            self.rew_buf[:] += self.stability_penalty_scaled

        # Legacy reward terms (keep for backward compatibility)
        # if hasattr(self, 'rot_rew_scaled'):
        #     self.rew_buf[:] += self.rot_rew_scaled

        # if not test_rel and self.env_mode == "relpose":
        #     self.rew_buf[:] += self.pos_rew_scaled
        # if self.env_mode == "relposecontact" or self.env_mode == "pgm":
        #     if "pclcontactonly" in self.reward_type or "pclcontactmatch" in self.reward_type:
        #         self.rew_buf[:] = (
        #             self.contact_rew_scaled + self.succ_rew_scaled + self.action_penalty_scaled + self.time_step_penatly
        #         )
        #     else:
        #         self.rew_buf[:] += self.pos_rew_scaled + self.contact_rew_scaled

        #     if self.env_mode == "pgm" and "height" in self.reward_type:
        #         self.rew_buf[:] += self.height_rew_scaled
        #     if "nominal" in self.reward_type:
        #         self.rew_buf[:] += self.nominal_rew_scaled
        #     if "ft2oc" in self.reward_type:
        #         self.rew_buf[:] += self.ft2oc_rew_scaled
        #     if "similarity" in self.reward_type:
        #         self.rew_buf[:] += self.similarity_reward_scaled * (
        #             self.progress_buf % self.similarity_reward_freq == 0
        #         )
        #     if "manipen" in self.reward_type:
        #         self.rew_buf[:] += self.manipulability_penalty_scaled

        self.compute_done(is_success)

    def reset(self, dones=None, first_time=False):
        """Is called only once when environment starts to provide the first observations.

        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        if dones is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = dones.nonzero(as_tuple=False).flatten()

        # reset idx
        if env_ids.shape[0] > 0:
            self.reset_idx(env_ids, first_time=first_time)

        self.compute_observations(env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_idx(self, env_ids: LongTensor, first_time=False) -> None:
        num_reset_envs: int = env_ids.shape[0]

        if self.randomize:
            if "sim_params" in self.randomization_params:
                if "gravity" in self.randomization_params:
                    if first_time:
                        # weired thing is that the first time randomization will save the randmized prop as og prop, make origin gravity change...
                        self.randomization_params["sim_params"]["gravity"]["range"] = [1.0, 1.0]
                    else:
                        self.randomization_params["sim_params"]["gravity"]["range"] = [0.1, 0.1]
            self.apply_randomizations(self.randomization_params)
        if self.mode == "eval" and local_test:
            self.set_table_color(env_ids, color=[1.0, 1.0, 1.0])

        noise = torch.rand(env_ids.shape[0], 3, device=self.device) * 2.0 - 1.0

        if self.relative_part_reward:
            self.prev_pos_dist[env_ids] = -1
            self.prev_rot_dist[env_ids] = -1
            self.prev_contact_dist[env_ids] = -1
            self.prev_nominal_dist[env_ids] = -1
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset action noise times tracker
        if self.action_noise and self.action_noise_level == "step" and self.action_noise_max_times > 0:
            self.action_noise_times[env_ids] = 0

        occupied_object_relative_indices = self.occupied_object_relative_indices[env_ids]

        if self.env_info_logging:
            for i, env_id in enumerate(env_ids):
                # print(env_id, self.object_names[env_id][occupied_object_relative_indices[i]])
                pass

        object_indices = self.object_encodings[env_ids, occupied_object_relative_indices]
        examples = self.grasping_dataset.sample(object_indices)

        pointclouds = examples["pointcloud"]
        bbox = examples["bbox"]
        onehot = examples["category_onehot"]
        clutser_ids = examples["cluster"]

        if self.enable_full_pointcloud_observation:
            self.pointclouds[env_ids] = pointclouds

        self.occupied_object_cluster_ids[env_ids] = torch.from_numpy(clutser_ids).to(self.device).to(torch.long)

        if torch.sum(self.done_successes) / self.num_envs > self.curriculum_thres:
            self.done_successes[:] = 0
            if self.env_mode == "pgm" and "stage" in self.curriculum_mode:
                # switch stage2 reward
                self.height_scale = 1.0
                if self.relative_part_reward:
                    self.part_reward_scale = 1.0
                else:
                    self.part_reward_scale = 0.15
                self.nominal_env_ratio = 0.2
                self.curriculum_mode = "no"
            if self.env_mode == "pgm" and "pose" in self.curriculum_mode:
                self.height_scale = 0.0
                if self.relative_part_reward:
                    self.part_reward_scale = 1.0
                else:
                    self.part_reward_scale = 0.3
                self.nominal_env_ratio = 0.2
                self.curriculum_mode = "no"

        # Comment out target hand pose setup since we don't need it for singulation
        # reset target object root state
        # target_occupied_object_init_root_position = position(
        #     self.gym_assets["target"]["objects"]["occupied_pose"], self.device
        # )

        # reset target allegro-hand dof positions & velocities

        # do not set wrist angle for target!!!
        # do not use current hand pose for compute target!!!
        # allegro_hand_root_position = self._target_hand_palm_pose[:3].reshape(-1, 3).repeat(num_reset_envs, 1)
        # allegro_hand_root_orientation = self._target_hand_palm_pose[3:7].reshape(-1, 4).repeat(num_reset_envs, 1)

        # object_positions_wrt_palm = poses[:, 0:3]
        # object_orientations_wrt_palm = poses[:, 3:7]

        # palm_orientations_wrt_object, palm_positions_wrt_object = transformation_inverse(
        #     object_orientations_wrt_palm, object_positions_wrt_palm
        # )

        # # TODO change to forearm pose
        # object_orientation, object_position = transformation_multiply(
        #     allegro_hand_root_orientation,
        #     allegro_hand_root_position,
        #     object_orientations_wrt_palm,
        #     object_positions_wrt_palm,
        # )

        # ii, jj = torch.meshgrid(env_ids, self.grasping_joint_indices[2:], indexing="ij")
        # self._r_target_object_positions_wrt_palm[env_ids] = poses[:, 0:3]
        # self._r_target_object_orientations_wrt_palm[env_ids] = poses[:, 3:7]
        # self._r_target_allegro_dof_positions[ii, jj] = joints[:, 2:]
        # self._r_target_object_root_orientations[env_ids] = object_orientation
        # self._r_target_object_root_positions[env_ids] = object_position
        # self._r_target_palm_positions_wrt_object[env_ids] = palm_positions_wrt_object
        # self._r_target_palm_orientations_wrt_object[env_ids] = palm_orientations_wrt_object

        # ii, jj = torch.meshgrid(env_ids, self.allegro_digits_actuated_dof_indices - 6, indexing="ij")
        # self._r_target_allegro_digits_actuated_dof_positions[env_ids] = self._r_target_allegro_dof_positions[
        #     ii, jj
        # ].clone()

        # ii, jj = torch.meshgrid(env_ids, self.allegro_fingers_actuated_dof_indices - 6, indexing="ij")
        # self._r_target_allegro_fingers_actuated_dof_positions[env_ids] = self._r_target_allegro_dof_positions[
        #     ii, jj
        # ].clone()

        # ii, jj = torch.meshgrid(env_ids, self.allegro_thumb_actuated_dof_indices - 6, indexing="ij")
        # self._r_target_allegro_thumb_actuated_dof_positions[env_ids] = self._r_target_allegro_dof_positions[
        #     ii, jj
        # ].clone()

        # print(self._r_target_allegro_dof_positions.shape, self.target_allegro_hand_dof_positions.shape)

        # Comment out target setting since we don't need it for singulation
        # if self.env_mode == "orn":
        #     self.object_targets[env_ids] = object_orientation.clone()
        # elif self.env_mode == "relpose":
        #     self.object_targets[env_ids] = poses.clone()
        # elif self.env_mode == "relposecontact" or self.env_mode == "pgm":
        #     self.object_targets[env_ids, :7] = poses.clone()
        #     self.object_targets[env_ids, 7:25] = self._r_target_allegro_digits_actuated_dof_positions[env_ids]


        # TODO: add noise to the initial DOF positions
        dof_init_positions = self.gym_assets["current"]["robot"]["init"]["position"]
        dof_init_velocities = self.gym_assets["current"]["robot"]["init"]["velocity"]
        self.allegro_hand_dof_positions[env_ids, :] = dof_init_positions
        self.allegro_hand_dof_velocities[env_ids, :] = dof_init_velocities

        self.prev_targets[env_ids] = dof_init_positions
        self.curr_targets[env_ids] = dof_init_positions

        # random object orientation
        # if self.reset_obj_ori_noise > 0:
        #     occupied_object_init_root_orientation = random_orientation_within_angle(
        #         num_reset_envs, self.device, object_orientation, self.reset_obj_ori_noise / (180 / torch.pi)
        #     )
        # else:
        #     occupied_object_init_root_orientation = random_orientation(num_reset_envs, self.device)

        # For singulation task, use nominal orientation (no rotation) for all boxes
        occupied_object_init_root_orientation = self._object_nominal_orientation.clone().detach().repeat(num_reset_envs, 1)

        # Compute statastics of object pointclouds
        pointclouds_wrt_world = quat_rotate(occupied_object_init_root_orientation[:, None, :], pointclouds)

        obj_min_z = torch.min(pointclouds_wrt_world[:, :, 2], dim=1)[0]
        obj_x_length = torch.max(pointclouds[:, :, 0], dim=1)[0] - torch.min(pointclouds[:, :, 0], dim=1)[0]
        obj_y_length = torch.max(pointclouds[:, :, 1], dim=1)[0] - torch.min(pointclouds[:, :, 1], dim=1)[0]
        obj_z_length = torch.max(pointclouds[:, :, 2], dim=1)[0] - torch.min(pointclouds[:, :, 2], dim=1)[0]
        obj_max_length = torch.max(torch.stack([obj_x_length, obj_y_length, obj_z_length]), dim=0)[0]
        self.obj_max_length[env_ids] = obj_max_length.clone()
        self.object_bboxes[env_ids] = bbox.clone()
        self.object_categories[env_ids] = onehot.clone()

        if hasattr(self, "fingertips_to_obj_dist_min"):
            self.fingertips_to_obj_dist_min[env_ids] = 0.3
        if hasattr(self, "keypoints_to_obj_dist_min"):
            self.keypoints_to_obj_dist_min[env_ids] = 0.3
        if hasattr(self, "goal_position_dist_min"):
            self.goal_position_dist_min[env_ids] = 1.0
        # Set occupied object root positions & orientations
        self.root_positions[self.object_indices.view(self.num_envs, -1)[env_ids], :] = self.init_scene_object_root_positions[env_ids, :]
        self.root_orientations[self.object_indices.view(self.num_envs, -1)[env_ids], :] = self.init_scene_object_root_orientations[env_ids, :]
        self.root_linear_velocities[self.object_indices.view(self.num_envs, -1)[env_ids], :] = 0.0
        self.root_angular_velocities[self.object_indices.view(self.num_envs, -1)[env_ids], :] = 0.0

        self.robot_init_dof[env_ids, :] = dof_init_positions.clone()

        # Set dof-position-targets & dof-states
        indices = self.allegro_hand_indices[env_ids]
        indices = indices.flatten().to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        # Set actor-root-states
        indices = self.object_indices[env_ids]
        indices = indices.flatten().to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        # Reset progress-buffer, reset-buffer, success-buffer
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        
        self.near_goal_steps[env_ids] = 0
        self.reset_goal_buf[env_ids] = 0
        self.picked[env_ids] = 0
        self.picked_curr[env_ids] = 0
        self.near_goal[env_ids] = 0
        # self.goal_position_dist_min[env_ids] = 1.0
        

    def get_env_metainfo(self, field: Optional[str] = None) -> Union[pd.DataFrame, Sequence]:
        """Get environment meta information. (info not changed during the episode)

        Returns:
            pd.DataFrame: Environment meta information
        """

        indices = np.arange(self.num_envs)
        codes = self.occupied_object_codes
        grasps = self.occupied_object_grasps
        cluster_ids = self.occupied_object_cluster_ids.cpu().numpy()

        init_positions = self.occupied_object_init_root_positions.cpu().numpy()
        init_x = init_positions[:, 0]
        init_y = init_positions[:, 1]
        init_z = init_positions[:, 2]

        metainfo = {
            "index": indices,
            "code": codes,
            "grasp": grasps,
            "cluster_id": cluster_ids,
            "init_x": init_x,
            "init_y": init_y,
            "init_z": init_z,
        }

        assert field is None or field in metainfo, f"field {field} not found in metainfo"
        if field is not None:
            return metainfo[field]
        return pd.DataFrame(metainfo)

    def set_states(
        self,
        robot_dof,
        object_targets=None,
        obj_pos=None,
        obj_orn=None,
        env_ids=None,
        step_time=-1,
        denomalize_robot_dof=False,
        set_dof_state=True,
        arm_ik=False,
    ):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device)

        if object_targets is not None and obj_pos is not None and obj_orn is not None:
            self.object_targets[env_ids] = object_targets
            # these are used for reward computation
            self._r_target_object_positions_wrt_palm[env_ids] = object_targets[:, :3]
            self._r_target_object_orientations_wrt_palm[env_ids] = object_targets[:, 3:7]
            ii, jj = torch.meshgrid(env_ids, self.allegro_digits_actuated_dof_indices - 6, indexing="ij")
            self._r_target_allegro_dof_positions[ii, jj] = object_targets[:, 7:25]

            self.occupied_object_init_root_positions[env_ids, :] = obj_pos
            self.occupied_object_init_root_orientations[env_ids, :] = obj_orn

            self.root_positions[self.occupied_object_indices[env_ids], :] = obj_pos
            self.root_orientations[self.occupied_object_indices[env_ids], :] = obj_orn

            indices = torch.unique((self.object_indices[env_ids]).flatten().to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(indices),
                indices.shape[0],
            )

        if arm_ik:
            targets = self.prev_targets.clone()

            cur_pos = self.endeffector_positions.clone()
            cur_quat = self.endeffector_orientations.clone()
            arm_pos = robot_dof[:, self.arm_trans_action_indices].clone()
            arm_rot_euler = robot_dof[:, self.arm_rot_action_indices].clone()
            arm_rot_quat = quat_from_euler_xyz(arm_rot_euler[:, 0], arm_rot_euler[:, 1], arm_rot_euler[:, 2])
            hand_target_dof = robot_dof[:, self.hand_action_indices]
            delta_joint_move = ik(
                self.j_eef,
                cur_pos,
                cur_quat,
                arm_pos,
                arm_rot_quat,
            )
            arm_target_dof = targets[:, self.ur_actuated_dof_indices] + delta_joint_move

            current_dof = targets.clone()
            current_dof[:, self.allegro_actuated_dof_indices] = hand_target_dof
            current_dof[:, self.allegro_tendon_dof_indices] = saturate(
                current_dof[:, self.allegro_coupled_dof_indices]
                - self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_coupled_dof_indices],
                self.gym_assets["current"]["robot"]["limits"]["lower"][self.allegro_tendon_dof_indices],
                self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_tendon_dof_indices],
            )
            robot_dof = current_dof.clone()
            robot_dof[:, self.ur_actuated_dof_indices] = arm_target_dof

            robot_dof = saturate(
                robot_dof,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )

        if denomalize_robot_dof:
            robot_dof = denormalize(
                robot_dof,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )

        if set_dof_state:
            self.allegro_hand_dof_positions[env_ids, :] = robot_dof
        self.prev_targets[env_ids] = robot_dof
        self.curr_targets[env_ids] = robot_dof

        indices = torch.unique((self.allegro_hand_indices[env_ids]).flatten().to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        if set_dof_state:
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(indices), indices.shape[0]
            )

        if step_time > 0:
            self.step_simulation(step_time)

        self.compute_observations()

    def move_arm_to_pose(self, position, orientation):
        for _ in range(100):
            targets = self.prev_targets.clone()

            cur_pos = self.endeffector_positions.clone()
            cur_quat = self.endeffector_orientations.clone()

            delta_joint_move = ik(self.j_eef, cur_pos, cur_quat, position, orientation) * self.dof_speed_scale * self.dt

            self.curr_targets[:, self.ur_actuated_dof_indices] = (
                targets[:, self.ur_actuated_dof_indices] + delta_joint_move
            )

            self.curr_targets[:] = saturate(
                self.curr_targets,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )
            self.prev_targets[:] = self.curr_targets[:]

            indices = torch.unique(
                torch.cat([self.allegro_hand_indices, self.target_allegro_hand_indices]).flatten().to(torch.int32)
            )

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.curr_targets_buffer),
                gymtorch.unwrap_tensor(indices),
                indices.shape[0],
            )

            # step physics and render each frame
            for i in range(self.control_freq_inv):
                if self.force_render:
                    self.render()
                self.gym.simulate(self.sim)

            # to fix!
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)

            self._refresh_sim_tensors()

    def _refresh_action_tensors(self, actions: torch.Tensor) -> None:
        """Given a batch of actions, refresh the action tensors.

        Args:
            actions (torch.Tensor): A batch of actions. [batch_size, action_dim]
        """
        current = 0
        for spec in self._action_space:
            setattr(self, spec.attr, actions[:, current : current + spec.dim])
            current += spec.dim

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.training:
            self.reset_done()

        if self.action_noise:
            noise = torch.randn_like(actions) * self.action_noise_sigma
            if self.action_noise_level == "value":
                mask = torch.rand((self.num_envs, self.num_actions), device=self.device) < self.action_noise_ratio
            elif self.action_noise_level == "step":
                mask = torch.rand((self.num_envs), device=self.device) < self.action_noise_ratio
                if self.action_noise_max_times > 0:
                    mask = mask & (self.action_noise_times < self.action_noise_max_times)
                    self.action_noise_times[mask] += 1
                mask = mask.unsqueeze(-1).repeat(1, self.num_actions)
            # ignore the actions that are already zero
            zero = (actions.abs() < 1e-8).all(dim=1).unsqueeze(-1)
            mask = mask & ~zero
            # add noise
            actions[mask] += noise[mask]

        self.actions = actions.clone().to(self.device)
        self.clamped_actions = actions.clone().to(self.device)
        self.prev_endeffector_states = self.endeffector_states.clone()
        self.prev_endeffector_positions = self.endeffector_positions.clone()
        self.prev_endeffector_orientations = self.endeffector_orientations.clone()
        self.prev_endeffector_orientations_euler = torch.stack(get_euler_xyz(self.prev_endeffector_orientations), dim=1)
        self.prev_allegro_actuated_dof_positions = self.allegro_hand_dof_positions[
            :, self.allegro_actuated_dof_indices
        ].clone()
        self._refresh_action_tensors(self.actions)

        if self.use_relative_control:
            print(111)
            breakpoint()
            targets = self.prev_targets.clone()
            if self.ur_control_type == "osc":
                xarm_dof_movements, self.target_eef_pos, self.target_eef_euler = compute_relative_xarm_dof_positions(
                    self.endeffector_positions,
                    self.endeffector_orientations,
                    self.j_eef,
                    self.eef_translation,
                    self.eef_rotation,
                    self._max_xarm_endeffector_pos_vel,
                    self._max_xarm_endeffector_rot_vel,
                    self.dt,
                )
            else:
                xarm_dof_speeds = torch.cat([self.eef_translation, self.eef_rotation], dim=1)
                xarm_dof_movements = xarm_dof_speeds * self.dof_speed_scale * self.dt

            if getattr(self, "eef_translation", None) is None and getattr(self, "eef_rotation", None) is None:
                xarm_dof_movements[:] = 0

            self.curr_targets[:, self.xarm_actuated_dof_indices] = (
                targets[:, self.xarm_actuated_dof_indices] + xarm_dof_movements
            )

            if getattr(self, "allegro_dof_speeds", None) is not None:
                # hand moving
                # targets[:, self.allegro_coupled_dof_indices] = (
                #     targets[:, self.allegro_coupled_dof_indices] + targets[:, self.allegro_tendon_dof_indices]
                # )
                if self.use_predef_hand_pose:
                    self.curr_targets[
                        :, self.allegro_actuated_dof_indices
                    ] = torch.tensor(
                        self._allegro_hand_predef_qpos,
                        device=self.device,
                        dtype=torch.float,
                    )
                else:
                    self.curr_targets[:, self.allegro_actuated_dof_indices] = (
                        targets[:, self.allegro_actuated_dof_indices]
                        + self.allegro_dof_speeds * self.dof_speed_scale * self.dt
                    )
                # self.curr_targets[:, self.allegro_tendon_dof_indices] = saturate(
                #     self.curr_targets[:, self.allegro_coupled_dof_indices]
                #     - self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_coupled_dof_indices],
                #     self.gym_assets["current"]["robot"]["limits"]["lower"][self.allegro_tendon_dof_indices],
                #     self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_tendon_dof_indices],
                # )
        else:
            print(222)
            # simulate the tendon coupling
            self.curr_targets[:, self.actuated_dof_indices] = self.actions
            self.curr_targets[:, self.allegro_tendon_dof_indices] = (
                torch.clamp_min(self.curr_targets[:, self.allegro_coupled_dof_indices], 0.0) * 2.0 - 1.0
            )
            self.curr_targets[:, self.allegro_coupled_dof_indices] = (
                torch.clamp_max(self.curr_targets[:, self.allegro_coupled_dof_indices], 0.0) * 2.0 + 1.0
            )
            # denormalize & saturate the targets
            self.curr_targets[:] = denormalize(
                self.curr_targets,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )
            self.curr_targets[:] = (
                self.act_moving_average * self.curr_targets + (1.0 - self.act_moving_average) * self.prev_targets
            )

        self.curr_targets[:] = saturate(
            self.curr_targets,
            self.gym_assets["current"]["robot"]["limits"]["lower"],
            self.gym_assets["current"]["robot"]["limits"]["upper"],
        )

        # return
        if test_sim:
            self.curr_targets[:, 6:] = self.gym_assets["current"]["robot"]["limits"]["lower"][6:]
        self.prev_targets[:] = self.curr_targets[:]

        indices = self.allegro_hand_indices
        indices = indices.flatten().to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        # self.target_allegro_hand_dof_positions[:] = self.curr_target_targets[:]
        # self.target_allegro_hand_dof_velocities[:] = 0.0
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.dof_states),
        #     gymtorch.unwrap_tensor(self.target_allegro_hand_indices.to(torch.int32)),
        #     self.num_envs,
        # )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()

        if self.method == "case":
            self.compute_case2023_reward()
        else:
            self.compute_reward(self.actions)

        # track gpu memory usage
        if self.device.startswith("cuda"):
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info(device=self.device)
            gpu_mem_occupied = torch.tensor([gpu_mem_total - gpu_mem_free], device=self.device)
            self.extras["gpu_mem_occupied_MB"] = gpu_mem_occupied / 1024 / 1024
            self.extras["gpu_mem_occupied_GB"] = gpu_mem_occupied / 1024 / 1024 / 1024
            self.extras["gpu_mem_occupied_ratio"] = gpu_mem_occupied / gpu_mem_total

        self.extras["max_jacobian_det"] = torch.max(torch.det(self.j_eef).abs()).reshape(1)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            origin_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            origin_orientations = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
            origin_orientations[:, 3] = 1
            draw_axes(self.gym, self.viewer, self.envs, origin_positions, origin_orientations, 0.5)
            draw_axes(self.gym, self.viewer, self.envs, self.object_root_positions, self.object_root_orientations, 0.1)

            if self.enable_rendered_pointcloud_observation:
                self.draw_camera_axes()

            if self.enable_contact_sensors:
                self.draw_force_sensor_axes()

    def reset_obj_vel(self, env_ids):
        # important reset object velocity and angular velocity to zero
        occupied_object_indices = torch.unique(torch.cat([self.occupied_object_indices[env_ids]]).to(torch.int32))
        self.root_states[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_states[self.occupied_object_indices[env_ids], 7:13]
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(occupied_object_indices),
            len(occupied_object_indices),
        )

    def lift_test(self, env_ids, close_dis=0.1, close_dof_indices=None, only_evaluate_height=False):
        # TODO: finger will lose during the lifting?
        # generate stable grasp
        # if close_dof_indices is None:
        #     close_dof_indices = self.close_dof_indices.clone()

        # self.curr_targets[env_ids, :] = self.allegro_hand_dof_positions.clone()
        # self.close(env_ids, close_dis, close_dof_indices)
        # self.curr_targets[env_ids, :] = self.allegro_hand_dof_positions.clone()
        # self.reset_obj_vel(env_ids)

        current_pos = self.endeffector_positions.clone()
        target_pos = current_pos.clone()
        target_pos[:, 2] += 0.3

        for i in range(100):
            delta_joint_move = ik(
                self.j_eef,
                self.endeffector_positions,
                self.endeffector_orientations,
                target_pos,
                self.endeffector_orientations,
            )
            delta_joint_move = delta_joint_move * self.dof_speed_scale * self.dt

            targets = self.allegro_hand_dof_positions.clone()
            ii, jj = torch.meshgrid(env_ids, self.ur_actuated_dof_indices, indexing="ij")
            self.curr_targets[ii, jj] = targets[ii, jj] + delta_joint_move
            # apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            # apply_forces[env_ids, self.allegro_center_index, 2] = 10
            # self.gym.apply_rigid_body_force_tensors(
            #     self.sim, gymtorch.unwrap_tensor(apply_forces), None, gymapi.ENV_SPACE
            # )

            # ii, jj = torch.meshgrid(env_ids, close_dof_indices)
            # self.curr_targets[ii, jj] += 0.02

            indices = torch.unique(
                torch.cat([self.allegro_hand_indices, self.target_allegro_hand_indices]).flatten().to(torch.int32)
            )
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.curr_targets_buffer),
                gymtorch.unwrap_tensor(indices),
                indices.shape[0],
            )
            # step physics and render each frame
            for i in range(self.control_freq_inv):
                if self.force_render:
                    self.render()
                self.gym.simulate(self.sim)

            self._refresh_sim_tensors()

            print(
                F.pairwise_distance(
                    self.allegro_hand_dof_positions[0, 6:],
                    self.curr_targets_buffer[0, self.allegro_hand_dof_start : self.allegro_hand_dof_end][6:],
                )
            )
        print("lifted")

    # Visualization Utilities

    def close(self, env_ids, close_dis=0.3, close_dof_indices=None, check_contact=False):
        for i in range(50):
            if i < 30:
                targets = self.allegro_hand_dof_positions.clone()
                ii, jj = torch.meshgrid(env_ids, close_dof_indices, indexing="ij")
                self.curr_targets[ii, jj] = targets[ii, jj] + close_dis / 30
                indices = torch.unique(
                    torch.cat([self.allegro_hand_indices, self.target_allegro_hand_indices]).flatten().to(torch.int32)
                )
                self.gym.set_dof_position_target_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.curr_targets_buffer),
                    gymtorch.unwrap_tensor(indices),
                    indices.shape[0],
                )
            if self.force_render and i % 1 == 0:
                self.render()
            self.gym.simulate(self.sim)
            self._refresh_sim_tensors()

    def draw_force_sensor_axes(self) -> None:
        positions: torch.Tensor = self.allegro_hand_rigid_body_positions[:, self.force_sensor_rigid_body_indices]
        orientations: torch.Tensor = self.allegro_hand_rigid_body_orientations[:, self.force_sensor_rigid_body_indices]
        draw_boxes(self.gym, self.viewer, self.envs, positions, orientations, 0.001)

    def draw_camera_axes(self) -> None:
        for i in range(self.num_cameras_per_env):
            draw_axes(
                self.gym, self.viewer, self.envs, self.camera_positions[:, i], self.camera_orientations[:, i], 0.1
            )

    def print_force_sensor_info(self, env_id: int = 0) -> None:
        force_sensor_states = self.force_sensor_states.view(self.num_envs, self.num_force_sensors, 6)
        force_sensor_state = force_sensor_states[env_id, ...]

        forces = force_sensor_state[:, 0:3]
        magnitudes = torch.norm(forces, dim=-1)
        print("force_magnitudes: ", magnitudes)
        # print("force_sensor_state: ", force_sensor_state)

    def get_images(self, img_width=1024, img_height=768, env_ids=None, simulate=True):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # vis part env
        env_ids = env_ids[: self.vis_env_num]
        # step the physics simulation
        if simulate:
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)

        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        if self.force_render:
            self.render()

        images = []
        # get rgb image
        for env_id in env_ids:
            image = self.gym.get_camera_image(
                self.sim, self.envs[env_id], self.cameras_handle[env_id], gymapi.IMAGE_COLOR
            )
            image = np.reshape(image, (np.shape(image)[0], -1, 4))[..., :3]
            image = image[:, :, (2, 1, 0)]
            image = cv2.resize(image, (img_width, img_height))
            images.append(image)

        images = np.stack(images, axis=0)
        images = to_torch(images, device=self.device)
        return images

def compute_offset_point_world(
    obj_position: torch.Tensor,
    obj_orientation: torch.Tensor,
    local_offset: torch.Tensor,
) -> torch.Tensor:
    """Compute the global position of a point defined in an object's local frame.
    
    Args:
        obj_position (torch.Tensor): Object's global position, shape (..., 3)
        obj_orientation (torch.Tensor): Object's global orientation (quaternion), shape (..., 4)
        local_offset (torch.Tensor): Offset in object's local frame, shape (3) or (..., 3)
        
    Returns:
        torch.Tensor: Global position of the offset point, shape (..., 3)
    """
    # 确保所有输入具有兼容的形状
    # 处理 local_offset 的广播
    if local_offset.dim() == 1 and obj_position.dim() > 1:
        # 如果 local_offset 是单个向量 (3,)，但 obj_position 是批次形式 (N, 3)
        # 则扩展 local_offset 以匹配批次大小
        local_offset = local_offset.expand_as(obj_position)
    
    # 处理 obj_orientation 的广播
    if obj_orientation.shape != obj_position.shape[:-1] + (4,):
        # 如果方向不是正确的形状，尝试广播
        obj_orientation = obj_orientation.expand(*obj_position.shape[:-1], 4)
    
    # 应用物体的旋转到局部偏移量
    rotated_offset = quat_apply(obj_orientation, local_offset)
    
    # 将旋转后的偏移量加到物体位置
    return obj_position + rotated_offset

def compute_relative_pose(
    a_orientation: torch.Tensor,
    a_position: torch.Tensor,
    b_orientation: torch.Tensor,
    b_position: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a pose in b's frame.

    Args:
        a_orientation (torch.Tensor): Orientations of a, shape (..., 4).
        a_position (torch.Tensor): Positions of a, shape (..., 3).
        b_orientation (torch.Tensor): Orientations of b, shape (..., 4).
        b_position (torch.Tensor): Positions of b, shape (..., 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Orientation & Position of a in b's frame.
    """
    assert a_position.dim() == b_position.dim()
    assert a_orientation.dim() == b_orientation.dim()

    w2b_rotation, w2b_translation = transformation_inverse(b_orientation, b_position)

    a_position, w2b_translation = torch.broadcast_tensors(a_position, w2b_translation)
    a_orientation, w2b_rotation = torch.broadcast_tensors(a_orientation, w2b_rotation)

    orientation, position = transformation_multiply(w2b_rotation, w2b_translation, a_orientation, a_position)
    return orientation, position


def compute_relative_position(
    a_position: torch.Tensor,
    b_orientation: torch.Tensor,
    b_position: torch.Tensor,
) -> torch.Tensor:
    """Compute a position in b's frame.

    Args:
        a_position (torch.Tensor): Positions of a, shape (..., 3).
        b_orientation (torch.Tensor): Orientations of b, shape (..., 4).
        b_position (torch.Tensor): Positions of b, shape (..., 3).

    Returns:
        torch.Tensor: Position of a in b's frame.
    """
    assert a_position.dim() == b_position.dim() == b_orientation.dim()

    w2b_rotation, w2b_translation = transformation_inverse(b_orientation, b_position)

    a_position, w2b_translation = torch.broadcast_tensors(a_position, w2b_translation)
    quaternion_shape = a_position.shape[:-1] + (4,)
    w2b_rotation = torch.broadcast_to(w2b_rotation, quaternion_shape)

    position = quat_apply(w2b_rotation, a_position) + w2b_translation
    return position


@torch.jit.script
def pointcloud_from_depth(
    depth: torch.Tensor,
    inv_view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    width: Optional[int] = None,
    height: Optional[int] = None,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    threshold: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct point cloud from depth image.

    Args:
        depth (torch.Tensor): depth image, shape (..., height, width)
        inv_view_matrix (torch.Tensor): inverse view matrix, shape (..., 4, 4)
        proj_matrix (torch.Tensor): projection matrix, shape (..., 4, 4)
        width (Optional[int]): width of depth image. Defaults to depth.shape[1].
        height (Optional[int]): height of depth image. Defaults to depth.shape[0].
        u (Optional[torch.Tensor], optional): 2d grid of u coordinates. Defaults to None.
        v (Optional[torch.Tensor], optional): 2d grid of v coordinates. Defaults to None.
        threshold (float, optional): depth threshold. Defaults to 10.0.

    Returns:
        - torch.Tensor: point cloud, shape (..., height * width, 3)
        - torch.Tensor: mask, shape (..., height * width)
    """
    assert depth.ndim >= 2
    assert depth.device == inv_view_matrix.device == proj_matrix.device
    assert u is None or u.device == depth.device
    assert v is None or v.device == depth.device
    device = depth.device

    if width is None:
        width = depth.size(-1)

    if height is None:
        height = depth.size(-2)

    if u is None or v is None:
        v, u = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

    fu = 2 / proj_matrix[..., 0, 0]
    fv = 2 / proj_matrix[..., 1, 1]

    fu = fu.unsqueeze(-1).unsqueeze(-1)
    fv = fv.unsqueeze(-1).unsqueeze(-1)

    center_u = width / 2
    center_v = height / 2

    z = depth
    x = -(u - center_u) / width * z * fu
    y = (v - center_v) / height * z * fv

    x, y, z = x.flatten(-2), y.flatten(-2), z.flatten(-2)

    mask = z > -threshold
    points = torch.stack((x, y, z), dim=-1)

    rotation = inv_view_matrix[..., 0:3, 0:3].unsqueeze(-3)
    translation = inv_view_matrix[..., 3, 0:3].unsqueeze(-2).unsqueeze(-2)

    points.unsqueeze_(-2)
    points = (points @ rotation) + translation
    points.squeeze_(-2)

    return points, mask


def compute_relative_xarm_dof_positions(
    current_eef_positions: torch.Tensor,
    current_eef_orientations: torch.Tensor,
    eef_jacobian: torch.Tensor,
    eef_translations: torch.Tensor,
    eef_rotations: torch.Tensor,
    max_eef_translation_speed: float,
    max_eef_rotation_speed: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute relative xarm dof positions.

    Args:
        current_eef_positions (torch.Tensor): Current end effector positions, shape (N, 3).
        current_eef_orientations (torch.Tensor): Current end effector orientations, shape (N, 4).
        eef_jacobian (torch.Tensor): End effector jacobian, shape (N, 6, 6).
        eef_translations (torch.Tensor): End-effector translations, shape (N, 3). - action
        eef_rotations (torch.Tensor): End-effector rotations, shape (N, 3). - action
        max_eef_translation_speed (float): The upper bound of end-effector translation speed.
        max_eef_rotation_speed (float): The upper bound of end-effector rotation speed.
        dt (float): Time step. (be used to compute max_eef_translation and max_eef_rotation)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - relative_xarm_dof_positions (torch.Tensor): Relative xarm dof positions, shape (N, 6).
            - target_eef_positions (torch.Tensor): Target end effector positions, shape (N, 3).
            - target_eef_euler (torch.Tensor): Target end effector euler angles, shape (N, 3).
    """

    # compute the max translation and rotation in single time step
    max_eef_translation: float = max_eef_translation_speed * dt
    max_eef_rotation: float = max_eef_rotation_speed * dt

    # compute the current time step action
    diff_translations = eef_translations * max_eef_translation
    diff_rotations = eef_rotations * max_eef_rotation

    # linear interpolation - translation
    dist = torch.norm(diff_translations, dim=-1, keepdim=True)
    t = torch.where(dist > max_eef_translation, max_eef_translation / dist, torch.ones_like(dist))
    diff_translations = t * diff_translations
    target_eef_positions = current_eef_positions + diff_translations

    current_eef_euler = torch.stack(get_euler_xyz(current_eef_orientations), dim=1)
    target_eef_euler = current_eef_euler + diff_rotations
    target_eef_orientations = quat_from_euler_xyz(*target_eef_euler.unbind(1))

    # slerp - rotation
    theta = quat_diff_rad(current_eef_orientations, target_eef_orientations).unsqueeze(1)
    mask = (theta > max_eef_rotation).squeeze(1)
    target_eef_orientations[mask] = (
        torch.sin(theta[mask] - max_eef_rotation) * current_eef_orientations[mask]
        + math.sin(max_eef_rotation) * target_eef_orientations[mask]
    ) / (torch.sin(theta[mask]))

    # for data collection
    target_eef_euler = torch.stack(get_euler_xyz(target_eef_orientations), dim=1)

    return (
        ik(
            eef_jacobian,
            current_eef_positions,
            current_eef_orientations,
            target_eef_positions,
            target_eef_orientations,
        ),
        target_eef_positions,
        target_eef_euler,
    )
