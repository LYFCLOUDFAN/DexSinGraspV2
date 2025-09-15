"""Record3D

Parse and stream Record3D captures. To get demo data, see `../assets/download_assets.sh`.

**Features:**

* :mod:`viser.extras` Record3D parser for RGBD data
* Point cloud visualization from depth maps
* Camera pose trajectory display
* Temporal playback controls with scrubbing

.. note::
    This example requires external assets. To download them, run:

    .. code-block:: bash

        git clone https://github.com/nerfstudio-project/viser.git
        cd viser/examples
        ./assets/download_assets.sh
        python 04_demos/00_record3d_visualizer.py  # With viser installed.
"""

import time
from pathlib import Path

import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf

import trimesh
import matplotlib.pyplot as plt
from viser.extras import ViserUrdf


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def generate_random_point_cloud(batch_size: int, num_points: int, size: float) -> np.ndarray:
    """Generate random point clouds on the surface of unit cubes.

    Args:
        batch_size: Number of point clouds to generate.
        num_points: Number of points to generate.
        size: Size of the cube.

    Returns:
        points: A (batch_size, N, 3) array of points.
    """
    points = []
    for i in range(6):
        face_points = np.random.uniform(-size/2, size/2, size=(batch_size, num_points // 6 + 1, 3))
        if i == 0:
            face_points[..., 0] = size/2
        elif i == 1:
            face_points[..., 0] = -size/2
        elif i == 2:
            face_points[..., 1] = size/2
        elif i == 3:
            face_points[..., 1] = -size/2
        elif i == 4:
            face_points[..., 2] = size/2
        elif i == 5:
            face_points[..., 2] = -size/2
        points.append(face_points)
    points = np.concatenate(points, axis=1)
    points = points[:, :num_points, :]
    return points


def generate_random_heatmap(batch_size: int, num_points: int) -> np.ndarray:
    """Generate random heatmaps on the surface of unit cubes.

    Args:
        batch_size: Number of point clouds to generate.
        num_points: Number of points to generate.

    Returns:
        A (batch_size, N, 1) array of heatmap values.
    """
    heatmap = np.random.uniform(0.0, 1.0, size=(batch_size, num_points, 1))
    return heatmap


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    """Convert a heatmap to RGB colors. using matplotlib.
    Args:
        heatmap: A (N, 1) array of heatmap values.
    Returns:
        A (N, 3) array of RGB colors.
    """
    cmap = plt.get_cmap("jet")
    colors = cmap(heatmap)[..., :3]
    return colors


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def quat_from_two_vectors(a, b):
    """返回把向量 a 旋到向量 b 的四元数（wxyz）。"""
    a = normalize(a)
    b = normalize(b)
    c = np.cross(a, b)
    d = np.dot(a, b)
    if d < -0.999999:  # 近似反向
        # 选择一个与 a 不平行的轴
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        rot_axis = normalize(np.cross(a, axis))
        # 180 度旋转的四元数：w=0, xyz=轴向
        return np.array([0.0, rot_axis[0], rot_axis[1], rot_axis[2]])
    w = np.sqrt((1.0 + d) * 2.0) * 0.5
    xyz = c / (2.0 * w)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def make_arrow_trimesh(length=0.2, shaft_radius=0.005, head_length=0.06, head_radius=0.015, sections=32):
    """沿 +Z 轴的箭头：圆柱(杆) + 圆锥(箭头)"""
    shaft_len = max(1e-6, length - head_length)
    # 杆：中心在 z=shaft_len/2
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_len, sections=sections)
    shaft.apply_translation([0, 0, shaft_len / 2.0])

    # 头：锥体底部在 z=shaft_len，顶端在 z=length
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=sections)
    head.apply_translation([0, 0, shaft_len])

    arrow = trimesh.util.concatenate([shaft, head])
    arrow.merge_vertices()  # 可选，轻度清理
    return arrow


def add_arrow(
    server,
    name: str,
    parent: str = "/frames",     # 也可以是 f"/frames/t{i}"
    origin=(0, 0, 0),
    direction=(1, 0, 0),
    length=0.2,
    shaft_radius=0.005,
    head_length=0.06,
    head_radius=0.015,
):
    # 1) 造一个沿 +Z 的箭头
    mesh = make_arrow_trimesh(length, shaft_radius, head_length, head_radius)

    # 2) 计算把 +Z 旋到 direction 的四元数（wxyz）
    q = quat_from_two_vectors(np.array([0, 0, 1.0]), np.array(direction))

    # 3) 在 parent 下添加
    server.scene.add_mesh_trimesh(
        name=f"{parent}/{name}",
        mesh=mesh,
        wxyz=q,
        position=np.array(origin, dtype=float),
        visible=True,
        # cast_shadow=True,
        # receive_shadow=True,
        scale=1.0,
    )


def main(share: bool = False) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Generating point clouds!")
    batch_size = 20
    num_points = 1024
    points = generate_random_point_cloud(batch_size, num_points, size=0.2)
    heatmap = generate_random_heatmap(batch_size, num_points)
    colors = heatmap_to_color(heatmap[..., 0])
    
    print("Loading robot urdf!")
    # robot urdf parent frame
    server.scene.add_frame(
        "/robot",
        position=(-0.5, 0, 0),
        show_axes=False,
    )
    
    urdf_path = Path(__file__).absolute().parent.parent.parent / "assets" / "urdf" / "xarm6_allegro_right.urdf"
    print("URDF path:", urdf_path)
    load_meshes = True
    load_collision_meshes = False
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robot",
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))
    delta_configs = - np.linspace(0, np.pi, batch_size)
    seq_configs = np.zeros((batch_size, len(initial_config)))
    move_idx = 1
    for i in range(batch_size):
        seq_configs[i, move_idx] = seq_configs[i, move_idx] + delta_configs[i]
    
    num_frames = batch_size
    initial_fps = 1.0

    # Initial camera pose.
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (-1.554, -1.013, 1.142)
        client.camera.look_at = (-0.005, 2.283, -0.156)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.02,
            step=1e-3,
            initial_value=0.01,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=initial_fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames
        viser_urdf.update_cfg(seq_configs[gui_timestep.value])

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames
        viser_urdf.update_cfg(seq_configs[gui_timestep.value])

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    for i in tqdm(range(num_frames)):
        position, color = points[i], colors[i]
        # breakpoint()

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=position,
                colors=color,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
        )
        
        # add arrow
        random_arrow = np.random.uniform(-1, 1, size=(3,))
        random_origin = position[np.random.randint(0, num_points)]
        add_arrow(
            server,
            name="arrow",
            parent=f"/frames/t{i}",
            origin=random_origin,
            direction=random_arrow,
            length=0.2,
            shaft_radius=0.01,
            head_length=0.08,
            head_radius=0.02,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[
            (gui_timestep.value + 1) % num_frames
        ].point_size = gui_point_size.value
        viser_urdf.update_cfg(seq_configs[gui_timestep.value])

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
