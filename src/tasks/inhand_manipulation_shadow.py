from isaacgymenvs.tasks.shadow_hand import ShadowHand
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
import torch
from .curiosity import NeuralHashCuriosity
import math
from .torch_utils import *
from isaacgym import gymapi
from isaacgym import gymtorch
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import json
from .isaacgym_utils import (
    ObservationSpec,
    ActionSpec,
    print_observation_space,
    print_action_space,
    print_asset_options,
    print_links_and_dofs,
    print_dof_properties,
)
import omegaconf
import warnings
from collections import OrderedDict, deque
import random
import os


class InhandManipulationShadow(ShadowHand):
    _asset_root: str = "/home/nus/IsaacGymEnvs/assets"
    
    _observation_specs: Sequence[ObservationSpec] = []
    _action_specs: Sequence[ActionSpec] = []
    _shadow_hand_center_prim: str = "palm_link"
    _fingertips: List[str] = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:thdistal"] # index, middle, ring, thumb
    _keypoints: List[str] = [
        # 'robot0:palm', 
        # 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
        # 'robot0:mfproximal', 'robot0:mfmiddle', 'robot0:mfdistal', 
        # 'robot0:rfproximal', 'robot0:rfmiddle', 'robot0:rfdistal', 
        # 'robot0:lfmetacarpal', 'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal', 
        # 'robot0:thproximal', 'robot0:thmiddle', 'robot0:thdistal'
        'robot0:ffproximal', 'robot0:ffmiddle',
        'robot0:mfproximal', 'robot0:mfmiddle',
        'robot0:rfproximal', 'robot0:rfmiddle',
        'robot0:thproximal', 'robot0:thmiddle'
        
    ]
    _keypoints_info_path: str = "assets/urdf/robot_shadow_hand_keypoints.json" # XXX: one keypointper link for now.
    _keypoints_info: Dict[str, List[List[float]]] = json.load(open(_keypoints_info_path, "r"))
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        seed = cfg["env"]["seed"]
        torch.manual_seed(seed)  # cpu
        random.seed(seed)
        np.random.seed(seed)

        self.cfg = cfg
        
        self.env_info_logging = self.cfg["logging"]["envInfo"]
        self.stack_frame_number = self.cfg["env"]["stackFrameNumber"]
        self.enable_contact_sensors = self.cfg["env"]["enableContactSensors"]
        self.reward_type = self.cfg["env"]["rewardType"]
        self.mode = self.cfg["env"]["mode"]
        self.velocity_observation_scale = self.cfg["env"]["velocityObservationScale"]
        
        self.gym_assets = {}
        self.gym_assets["current"] = {}
        
        self.__configure_mdp_spaces()
        
        # =========================================================== ShadowHand Task __init__() ===================================================
        # directly calling parent class's __init__ will make wrong _observation_space
        
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 211
        }

        self.up_axis = 'z'

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        # self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        # self.cfg["env"]["numStates"] = num_states
        # self.cfg["env"]["numActions"] = 20

        super(ShadowHand, self).__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        # =================================================================== ShadowHand Task __init__() END ===================================================
        
        dof_state = self.dof_state.view(self.num_envs, -1, 2)
        self.shadow_hand_dof_positions = dof_state[:, self.shadow_hand_dof_start : self.shadow_hand_dof_end, 0]
        self.shadow_hand_dof_velocities = dof_state[:, self.shadow_hand_dof_start : self.shadow_hand_dof_end, 1]
        self.shadow_hand_dof_forces = self.dof_force_tensor[:, self.shadow_hand_dof_start : self.shadow_hand_dof_end]
        
        self.shadow_hand_rigid_body_states = self.rigid_body_states[
            :, self.shadow_hand_rigid_body_start : self.shadow_hand_rigid_body_end, :
        ]    
        self.shadow_hand_rigid_body_positions = self.shadow_hand_rigid_body_states[..., 0:3]
        self.shadow_hand_rigid_body_orientations = self.shadow_hand_rigid_body_states[..., 3:7]
        self.shadow_hand_rigid_body_linear_velocities = self.shadow_hand_rigid_body_states[..., 7:10]
        self.shadow_hand_rigid_body_angular_velocities = self.shadow_hand_rigid_body_states[..., 10:13]
        
        self.shadow_hand_center_states = self.shadow_hand_rigid_body_states[:, self.shadow_center_index, :]
        self.shadow_hand_center_positions = self.shadow_hand_center_states[:, 0:3]
        self.shadow_hand_center_orientations = self.shadow_hand_center_states[:, 3:7]
        
        self.keypoint_offset = torch.tensor(self.keypoint_offset, device=self.device).reshape(1, -1, 3)
        
        self.curiosity_repr = self.cfg["env"].get("curiosityRepresentation", "nearest_surface")
        curiosity_cfg = self.cfg["env"]["curiosity"]
        self.curiosity_handler = NeuralHashCuriosity(
            curiosity_cfg, self.device, self.num_envs
        )
        self.curiosity_reward_scale = curiosity_cfg["reward_scale"]
        
        self.__create_object_dataset(device=sim_device)
                
        _net_contact_forces: torch.Tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.num_rigid_bodies: int = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs
        self.net_contact_forces: torch.Tensor = gymtorch.wrap_tensor(_net_contact_forces)
        
        self.observation_info = {}
        observation_space = self.cfg["env"]["observationSpace"]
        for name in observation_space:
            self.observation_info[name] = self._get_observation_dim(name)
            
        # meaning less attributes, only for log in ppo.py
        self.object_codes = ["all"]
        self.label_paths = ["in_hand_manipulation_shadow"]
        self.num_objects = 1
        self.object_cat = self.object_type
        self.max_per_cat = -1
        self.object_geo_level = "all"
        self.object_scale = "all"

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.reset_arm(first_time=True)

    def reset_arm(self, first_time=False):
        self.reset(first_time=first_time)
        for _ in range(10):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self.compute_observations()

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
        for name in observation_space:
            print(f"Observation: {name}, dim: {self._get_observation_dim(name)}")
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
    
    def reset(self, dones=None, first_time=False):
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

    def reset_idx(self, env_ids: torch.LongTensor, first_time=False) -> None:
        super().reset_idx(env_ids, env_ids)
        
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
                # observation = quat_to_6d(observation)
                observation = observation

            observations[spec.name] = observation
            # print("retrieved observation:", spec.name, observation.shape, spec.tags, spec.attr)

        return observations
    
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
        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        self.goal_ori_dist = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        
        self.object_root_positions = self.object_pos
        self.object_root_orientations = self.object_rot
        self.object_root_linear_velocities = self.object_linvel
        self.object_root_angular_velocities = self.object_angvel

        self.keypoint_positions = self.shadow_hand_rigid_body_positions[:, self.keypoint_indices, :]
        self.keypoint_orientations = self.shadow_hand_rigid_body_orientations[:, self.keypoint_indices, :]
        self.keypoint_positions_with_offset = self.keypoint_positions + quat_apply(self.keypoint_orientations, self.keypoint_offset.repeat(self.num_envs, 1, 1))
        self.keypoint_orientations_with_offset = self.keypoint_orientations # currently no orientation offset
        
        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)
        self.keypoint_contact_forces = net_contact_forces[:, self.keypoint_indices, :]
        
        self.fingertip_states = self.shadow_hand_rigid_body_states[:, self.fingertip_indices, :]
        self.fingertip_positions = self.fingertip_states[..., 0:3]
        self.fingertip_orientations = self.fingertip_states[..., 3:7]
        self.fingertip_linear_velocities = self.fingertip_states[..., 7:10]
        self.fingertip_angular_velocities = self.fingertip_states[..., 10:13]
        
        self.fingertip_orientations_wrt_palm, self.fingertip_positions_wrt_palm = compute_relative_pose(
            self.fingertip_orientations,
            self.fingertip_positions,
            self.shadow_hand_center_orientations[:, None, :],
            self.shadow_hand_center_positions[:, None, :],
        )
        
        self.object_orientations_wrt_palm, self.object_positions_wrt_palm = compute_relative_pose(
            self.object_root_orientations,
            self.object_root_positions,
            self.shadow_hand_center_orientations,
            self.shadow_hand_center_positions,
        )

        self.object_positions_wrt_keypoints = self.keypoint_positions - self.object_root_positions[:, None, :]

    def __create_object_dataset(self, device=None) -> None:
        # Create simple box grid dataset for singulation task
        from .dataset import ObjectDataset

        self.grasping_dataset = ObjectDataset(
            object=self.object_type,
            device=device
        )
        
    def _define_shadow_hand(
        self, asset_name: str = "hand"
    ) -> Dict[str, Any]:
        """Define & load the shadow Hand  asset.

        Args:
            asset_name (str, optional): Asset name for logging. Defaults to "hand".

        Returns:
            Dict[str, Any]: The configuration of the robot.
        """
        print(">>> Loading shadow Hand for current scene")
        config = {"name": "hand"}

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.1

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        if self.env_info_logging:
            print_asset_options(asset_options, asset_name)
            
        shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName")

        shadow_hand_asset = self.gym.load_asset(self.sim, self._asset_root, shadow_hand_asset_file, asset_options)
        if self.env_info_logging:
            print_links_and_dofs(self.gym, shadow_hand_asset, asset_name)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(shadow_hand_asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(shadow_hand_asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(shadow_hand_asset)

        num_dofs = config["num_dofs"]

        shadow_hand_asset = self.gym.load_asset(self.sim, self._asset_root, shadow_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]
        
        # get shadow_hand dof properties, loaded by Isaac Gym from the MJCF file
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        # hand_dof_idx = 0

        # # set rigid-shape properties for shadow-hand
        # rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        # for shape in rigid_shape_props:
        #     shape.friction = 3.0
        # self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        # for i in range(num_dofs):
        #     name = self.gym.get_asset_dof_name(asset, i)
        #     dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        #     if name.endswith(".0"):
        #         dof_props["stiffness"][i] = 30
        #         dof_props["damping"][i] = 1
        #         dof_props["velocity"][i] = 3.0
        #         dof_props["effort"][i] = 5
        #         hand_dof_idx += 1
        #     else:
        #         dof_props["stiffness"][i] = 4000
        #         dof_props["damping"][i] = 80
        #         # dof_props["stiffness"][i] = 1e6
        #         # dof_props["damping"][i] = 1e2

        if self.env_info_logging:
            print_dof_properties(self.gym, shadow_hand_asset, shadow_hand_dof_props, asset_name)

        dof_lower_limits = [shadow_hand_dof_props["lower"][i] for i in range(num_dofs)]
        dof_upper_limits = [shadow_hand_dof_props["upper"][i] for i in range(num_dofs)]
        dof_init_positions = [0.0 for _ in range(num_dofs)]
        dof_init_velocities = [0.0 for _ in range(num_dofs)]

        config["limits"] = {}
        config["limits"]["lower"] = torch.tensor(dof_lower_limits).float().to(self.device)
        config["limits"]["upper"] = torch.tensor(dof_upper_limits).float().to(self.device)

        config["init"] = {}
        config["init"]["position"] = torch.tensor(dof_init_positions).float().to(self.device)
        config["init"]["velocity"] = torch.tensor(dof_init_velocities).float().to(self.device)
        
        # fmt: off
        close_dof_names = [
            "joint_2.0", "joint_3.0",  # finger 0 (index)
            "joint_6.0", "joint_7.0",  # finger 1 (middle)
            "joint_10.0", "joint_11.0",  # finger 2 (ring)
            "joint_14.0", "joint_15.0",  # thumb
        ]
        # fmt: on

        self.close_dof_indices = torch.tensor(
            [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in close_dof_names],
            dtype=torch.long,
            device=self.device,
        )

        self.shadow_center_index = self.gym.find_asset_rigid_body_index(shadow_hand_asset, self._shadow_hand_center_prim)
        self.keypoint_indices = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, prim) for prim in self._keypoints]
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, prim) for prim in self._fingertips]
        self.keypoint_offset = [self._keypoints_info[link_name] for link_name in self._keypoints] # (#key_link, 1, 3)

        config["asset"] = shadow_hand_asset
        config["dof_props"] = shadow_hand_dof_props

        print(">>> Shadow Hand loaded")
        return config
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        print(">>> Setting up %d environments" % num_envs)
        num_per_row = int(np.sqrt(num_envs))
        
        # =========================================================== ShadowHand Task _create_envs() ===================================================
        # directly calling parent class's _create_envs() will make wrong number of rigid bodies
        
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        print(">>> Defining gym assets")

        asset_root = self._asset_root
        
        # actor has already been defined in parent class, here we just accomplish the gym_assets dict
        self.gym_assets["current"]["robot"] = self._define_shadow_hand()
        shadow_hand_asset = self.gym_assets["current"]["robot"]["asset"]
        shadow_hand_dof_props = self.gym_assets["current"]["robot"]["dof_props"]

        object_asset_file = self.asset_files_dict[self.object_type]

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = shadow_hand_start_pose.p.x
        pose_dy, pose_dz = -0.39, 0.10  

        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        
        # ===================================================== Shadow Hand _create_envs END =====================================================
        env = self.envs[-1] # it seems we can use the last env to get indices
        
        shadow_hand = self.gym.find_actor_handle(env, "hand")
        self.shadow_hand_index = self.gym.get_actor_index(env, shadow_hand, gymapi.DOMAIN_ENV)
        
        # define start and end indices for shadow hand DOFs to create contiguous slices
        self.shadow_hand_dof_start = self.gym.get_actor_dof_index(env, shadow_hand, 0, gymapi.DOMAIN_ENV)
        self.shadow_hand_dof_end = self.shadow_hand_dof_start + self.gym_assets["current"]["robot"]["num_dofs"]
        self.shadow_hand_indices = torch.tensor(self.hand_indices).long().to(self.device)
        self.shadow_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(env, shadow_hand, 0, gymapi.DOMAIN_ENV)
        self.shadow_hand_rigid_body_end = (
            self.shadow_hand_rigid_body_start + self.gym_assets["current"]["robot"]["num_rigid_bodies"]
        )
        
    def _get_target_surface_points_world(self) -> torch.Tensor:
        # (num_envs, P, 3)
        canonical = self.grasping_dataset._pointclouds  # (1,P,3)
        pc_world = quat_rotate(self.object_root_orientations[:, None, :], canonical) + self.object_root_positions[:, None, :]
        # self.visualize_pointcloud(pc_world)
        return pc_world

    # def visualize_pointcloud(self, pc, color=(0,1,0)):
    #     # 转成 isaac gym API 需要的 [pos, color] 格式
    #     colors = torch.tensor(color, dtype=torch.float32).reshape(1, 1, -1).repeat(pc.shape[0], pc.shape[1], 1).cpu().numpy()
    #     verts = torch.cat([pc, pc+0.01], dim=-1).cpu().numpy()  # (N, 6)

    #     self.gym.clear_lines(self.viewer)
    #     for i, env in enumerate(self.envs):
    #         # 传给 gym viewer
    #         self.gym.add_lines(
    #             self.viewer,
    #             env,  # 或者对应的 env
    #             pc.shape[1],
    #             verts[i],
    #             colors[i],
    #         )
    
    def compute_curiosity_observations_surface_all_keypoints(self) -> torch.Tensor:
        # Features per fingertip: [u_hat(3), r_log_norm(1)] → total 4 fingertips × 4 = 16
        pcl_world = self._get_target_surface_points_world()   # (N, P, 3)
        offseted_keypoints = self.keypoint_positions_with_offset

        # pairwise distances (batched): (N, 4, P)
        dists = torch.cdist(offseted_keypoints, pcl_world)
        min_dists_per_finger, idx_p = torch.min(dists, dim=2)  # (N,4), (N,4)

        # gather nearest surface point for each fingertip
        idx_p_exp = idx_p.unsqueeze(-1).expand(-1, -1, 3)      # (N,4,3)
        obj_pts = torch.gather(pcl_world, 1, idx_p_exp)        # (N,4,3)
        u = obj_pts - offseted_keypoints                                     # (N,4,3)

        r = torch.norm(u, dim=2, keepdim=True).clamp_min(1e-6) # (N,4,1)
        u_hat = u / r                                          # (N,4,3)

        r0 = self.cfg["env"].get("curiosity", {}).get("r0", 0.02)
        r_max = self.cfg["env"].get("curiosity", {}).get("r_max", 0.20)
        r_log = torch.log1p(r / r0)                            # (N,4,1)
        r_log_norm = (r_log / math.log1p(r_max / r0)).clamp(0.0, 1.0)

        feat = torch.cat([u_hat, r_log_norm], dim=-1)          # (N,4,4)
        return feat.reshape(self.num_envs, -1), r                 # (N,16)
        
    def compute_contact_filtered_keypoints_relative_pos(self):
        """
        Compute keypoint positions relative to the target object's center, filtered by contact.
        Returns:
            filtered_rel (Tensor): (N, 4, 3)
            has_contact (BoolTensor): (N,) any keypoint satisfied both conditions
        """
        # Relative keypoint positions to object center: (N,4,3)
        rel_pos = self.keypoint_positions - self.object_root_positions.unsqueeze(1)

        # Distance to nearest surface point per keypoint: (N,4,1)
        _, r = self.compute_curiosity_observations_surface_all_keypoints()  # r: (N,4,1)


        contact_mag = self.keypoint_contact_forces.norm(dim=-1, p=2)
        # print("contact_mag", contact_mag[0])

        # Contact filters
        near_surface = (r.squeeze(-1) < 0.01)           # (N,4)
        # print("near_surface", near_surface[0])
        has_force = (contact_mag > 0.5)                 # (N,4)
        contact_mask = near_surface & has_force         # (N,4)

        # Apply mask
        filtered_rel = rel_pos * contact_mask.unsqueeze(-1)  # (N,4,3)

        has_contact = contact_mask.any(dim=1)  # (N,)
        return filtered_rel, has_contact
        
    def compute_curiosity_observations(self):
        """Compute the curiosity observations."""
        # Contact-filtered fingertip relative positions  (N, 4 * 3)
        # filtered_rel, has_contact = self.compute_contact_filtered_fingertips_relative_pos()
        filtered_rel, has_contact = self.compute_contact_filtered_keypoints_relative_pos()
        self.contact_filtered_fingertips_relative_pos = filtered_rel
        self.ContactFilterFingertipsRelativePulse = filtered_rel.reshape(self.num_envs, -1)
        
        # print(self.ContactFilterFingertipsRelativePulse[0].view(4, 3))

        curiosity_obs = self.ContactFilterFingertipsRelativePulse  # (N,12)
        return curiosity_obs, has_contact
        
    def compute_curiosity_reward(self):
        """Compute the curiosity reward."""
        
        curiosity_obs, nearest_contact = self.compute_curiosity_observations()  
        # print("curiosity_obs", curiosity_obs[0])
        exploration_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        exploration_bonus = torch.zeros(self.num_envs, device=self.device)
        if exploration_mask.any():
            masked_obs = curiosity_obs[exploration_mask]
            masked_bonus = self.curiosity_handler.update_curiosity(
                masked_obs, self.curiosity_reward_scale
            )
            exploration_bonus[exploration_mask] = masked_bonus * 40
            # print("curiosity_reward", exploration_bonus[0])
            
        self.extras["curiosity_reward"] = exploration_bonus.clone()
        self.extras["exploration_rate"] = exploration_mask.float().clone()
        
        return exploration_bonus
    
    def compute_reach_reward_keypoints(self):
        """Reaching reward using keypoint-to-object-surface distances with historical minima."""
        pcl_world = self._get_target_surface_points_world()
        keypoints_w = self.keypoint_positions_with_offset

        # Current nearest distances from each keypoint to the object surface: (N, K)
        # torch.cdist: (N, K, P) → min over P
        dists = torch.cdist(keypoints_w, pcl_world)
        cur_min_dist, _ = torch.min(dists, dim=2)  # (N, K)

        if not hasattr(self, "keypoints_to_surface_dist_min"):
            self.keypoints_to_surface_dist_min = torch.full_like(cur_min_dist, 0.30)  # meters

        delta = (self.keypoints_to_surface_dist_min - cur_min_dist) #.clamp_min(0.0)  # (N, K)
        self.keypoints_to_surface_dist_min = torch.min(self.keypoints_to_surface_dist_min, cur_min_dist)

        reach_rew_keypoints = delta.mean(dim=1)  # (N,)
        self.reach_rew_keypoints = reach_rew_keypoints
        self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 1.0

        # Logging
        self.extras["keypoint_surface_distances"] = cur_min_dist.clone()
        self.extras["keypoints_to_surface_dist_min"] = self.keypoints_to_surface_dist_min.clone()
        self.extras["reach_rew_keypoints"] = self.reach_rew_scaled_keypoints.clone()
        
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        
        self.task_reward = self.rew_buf.clone()
        self.extras["task_reward"] = self.task_reward.clone()
        self.compute_reach_reward_keypoints(); 
        self.reach_rew_scaled = self.reach_rew_scaled_keypoints.clone()
        self.rew_buf[:] += self.reach_rew_scaled
        
        self.curiosity_reward = self.compute_curiosity_reward()
        self.rew_buf[:] += self.curiosity_reward

        # self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
        
        self.extras["success_num"] = torch.sum(self.successes>0).unsqueeze(-1).clone()
        
    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False        
        
        
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

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
