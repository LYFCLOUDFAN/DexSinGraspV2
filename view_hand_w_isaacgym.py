# minimal_urdf_viewer.py
# Isaac Gym (Preview 4) minimal example:
# Load a URDF and visualize it in the built-in viewer.

from isaacgym import gymapi, gymutil

def main():
    gym = gymapi.acquire_gym()

    # ---------------------------------------
    # Create sim (PhysX, with simple settings)
    # ---------------------------------------
    args = gymutil.parse_arguments(
        description="Minimal URDF viewer",
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": "/absolute/path/to/assets"},
            {"name": "--urdf_file",  "type": str, "default": "urdf/your_robot.urdf"},
            # {"name": "--graphics_device_id", "type": int, "default": 0},
            # {"name": "--compute_device_id", "type": int, "default": 0},
        ],
    )

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # PhysX defaults that work well for viewing
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ---------------------------------------
    # Viewer (GUI)
    # ---------------------------------------
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    # ---------------------------------------
    # Load URDF asset
    # ---------------------------------------
    asset_root = args.asset_root
    urdf_file  = args.urdf_file

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False           # 需要固定底座就改成 True
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = False
    asset_options.use_mesh_materials = True       # 如果 URDF 里有材质/颜色
    asset_options.armature = 0.01                 # 稍微加一点转动惯量稳定性

    asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_root}/{urdf_file}")

    # ---------------------------------------
    # Create env & actor
    # ---------------------------------------
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3( spacing,  spacing, 0.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.1)  # 把机器人抬离地面一点以免穿地
    # pose.r = gymapi.Quat.from_euler_zyx(yaw, pitch, roll)  # 需要的话设置初始姿态

    actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

    # 相机视角
    cam_pos = gymapi.Vec3(2.5, 2.5, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # ---------------------------------------
    # Main loop
    # ---------------------------------------
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
