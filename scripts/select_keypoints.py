#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
import xml.etree.ElementTree as ET


@dataclass
class CollisionGeometry:
    kind: str  # 'mesh' or 'box'
    mesh_path: Optional[str] = None
    box_size: Optional[Tuple[float, float, float]] = None
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class HandLink:
    name: str
    collisions: List[CollisionGeometry]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx # check urdf convention


def urdf_find_text_float_list(elem: ET.Element, path: str, default: Optional[List[float]] = None) -> Optional[List[float]]:
    found = elem.find(path)
    if found is None:
        return default
    text = found.get("xyz") if path.endswith("origin") else found.get("size")
    if text is None and path.endswith("origin"):
        text = found.get("rpy")
    if text is None:
        return default
    try:
        return [float(x) for x in text.strip().split()]
    except Exception:
        return default


def parse_urdf_for_hand_links(urdf_path: str, include_palm: bool = False) -> List[HandLink]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    def is_hand_link(name: str) -> bool:
        if name.startswith("link_base"): # xarm base link
            return False
        if name.startswith("link_"):
            return True
        if include_palm and name in {"palm", "base_link"}:
            return True
        # exclude xArm links: link1..link6
        if name in {"wrist"} and include_palm:
            return True
        return False

    links: List[HandLink] = []
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "")
        if not is_hand_link(link_name):
            continue
        collisions: List[CollisionGeometry] = []
        for coll in link_elem.findall("collision"):
            origin = coll.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if origin is not None:
                if origin.get("xyz"):
                    xyz_vals = [float(v) for v in origin.get("xyz").split()]
                    xyz = (xyz_vals[0], xyz_vals[1], xyz_vals[2])
                if origin.get("rpy"):
                    rpy_vals = [float(v) for v in origin.get("rpy").split()]
                    rpy = (rpy_vals[0], rpy_vals[1], rpy_vals[2])
            geom = coll.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None and mesh.get("filename"):
                collisions.append(
                    CollisionGeometry(
                        kind="mesh",
                        mesh_path=mesh.get("filename"),
                        origin_xyz=xyz,
                        origin_rpy=rpy,
                    )
                )
                continue
            box = geom.find("box")
            if box is not None and box.get("size"):
                size_vals = [float(v) for v in box.get("size").split()]
                collisions.append(
                    CollisionGeometry(
                        kind="box",
                        box_size=(size_vals[0], size_vals[1], size_vals[2]),
                        origin_xyz=xyz,
                        origin_rpy=rpy,
                    )
                )
        if len(collisions) == 0:
            # no coll link here
            continue
        links.append(HandLink(name=link_name, collisions=collisions))

    # from pdb import set_trace
    # set_trace()
    # Ensure finger links are covered at least
    if len([l for l in links if l.name.startswith("link_")]) == 0:
        print("[WARN] No Allegro finger links found in URDF. Check naming conventions.")
        exit()
    return links


def parse_all_links_collisions(urdf_path: str) -> Dict[str, List[CollisionGeometry]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    all_links: Dict[str, List[CollisionGeometry]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "")
        collisions: List[CollisionGeometry] = []
        for coll in link_elem.findall("collision"):
            origin = coll.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if origin is not None:
                if origin.get("xyz"):
                    xyz_vals = [float(v) for v in origin.get("xyz").split()]
                    xyz = (xyz_vals[0], xyz_vals[1], xyz_vals[2])
                if origin.get("rpy"):
                    rpy_vals = [float(v) for v in origin.get("rpy").split()]
                    rpy = (rpy_vals[0], rpy_vals[1], rpy_vals[2])
            geom = coll.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None and mesh.get("filename"):
                collisions.append(
                    CollisionGeometry(
                        kind="mesh",
                        mesh_path=mesh.get("filename"),
                        origin_xyz=xyz,
                        origin_rpy=rpy,
                    )
                )
                continue
            box = geom.find("box")
            if box is not None and box.get("size"):
                size_vals = [float(v) for v in box.get("size").split()]
                collisions.append(
                    CollisionGeometry(
                        kind="box",
                        box_size=(size_vals[0], size_vals[1], size_vals[2]),
                        origin_xyz=xyz,
                        origin_rpy=rpy,
                    )
                )
        if collisions:
            all_links[link_name] = collisions
    return all_links


def load_collision_trimesh(base_dir: str, col: CollisionGeometry) -> trimesh.Trimesh:
    if col.kind == "mesh":
        mesh_path = col.mesh_path
        assert mesh_path is not None
        resolved = mesh_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(base_dir, mesh_path)
        if not os.path.isfile(resolved):
            alt = os.path.join(base_dir, os.path.basename(mesh_path))
            if os.path.isfile(alt):
                resolved = alt
            else:
                raise FileNotFoundError(f"Mesh file not found: {mesh_path} (resolved: {resolved})")
        mesh = trimesh.load(resolved, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    else:
        assert col.box_size is not None
        size = np.array(col.box_size, dtype=float)
        mesh = trimesh.creation.box(extents=size)
    # apply collision origin (into link-local frame)
    R = rpy_to_matrix(*col.origin_rpy)
    t = np.array(col.origin_xyz, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh


def sample_points_from_link(link_meshes: List[trimesh.Trimesh], points_per_link: int = 2000) -> np.ndarray:
    if len(link_meshes) == 0:
        return np.zeros((0, 3))
    areas = np.array([m.area for m in link_meshes], dtype=float)
    areas = np.where(areas <= 1e-9, 1e-9, areas)
    probs = areas / np.sum(areas)
    pts_list: List[np.ndarray] = []
    for i, m in enumerate(link_meshes):
        k = max(1, int(round(points_per_link * probs[i])))
        pts, _ = trimesh.sample.sample_surface(m, k)
        pts_list.append(pts)
    pts_all = np.concatenate(pts_list, axis=0)
    return pts_all.astype(np.float32)


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3 = o3d.geometry.TriangleMesh()
    o3.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3.compute_vertex_normals()
    return o3


def hash_color(name: str) -> Tuple[float, float, float]:
    """Deterministic pseudo-random color from link name (in 0..1)."""
    # randomize hue -> rgb
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    sat = 0.65
    val = 0.95
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    i = i % 6
    if i == 0:
        r, g, b = val, t, p
    elif i == 1:
        r, g, b = q, val, p
    elif i == 2:
        r, g, b = p, val, t
    elif i == 3:
        r, g, b = p, q, val
    elif i == 4:
        r, g, b = t, p, val
    else:
        r, g, b = val, p, q
    return (float(r), float(g), float(b))


def build_global_points_and_meshes(link_names: List[str],
                                   link_to_world_meshes_all: Dict[str, List[o3d.geometry.TriangleMesh]],
                                   link_to_local_points: Dict[str, np.ndarray],
                                   T_world: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[o3d.geometry.TriangleMesh], Dict[str, Tuple[int, int]]]:
    """Compose a single global point cloud (Nx3) and color array (Nx3),
    plus colored world meshes for all links, and a mapping from link->(start,end)."""
    global_points = []
    global_colors = []
    colored_meshes: List[o3d.geometry.TriangleMesh] = []
    link_ranges: Dict[str, Tuple[int, int]] = {}

    cursor = 0
    for link in link_names:
        pts_local = link_to_local_points.get(link, np.zeros((0, 3), dtype=np.float32))
        T = T_world.get(link, np.eye(4))
        R = T[:3, :3]
        t = T[:3, 3]
        pts_world = (pts_local @ R.T) + t
        color = np.array(hash_color(link), dtype=float)
        if pts_world.shape[0] > 0:
            start = cursor
            end = start + pts_world.shape[0]
            link_ranges[link] = (start, end)
            cursor = end
            global_points.append(pts_world)
            global_colors.append(np.tile(color.reshape(1, 3), (pts_world.shape[0], 1)))
        # color meshes for this link
        for m in link_to_world_meshes_all.get(link, []):
            m_col = o3d.geometry.TriangleMesh(m)
            m_col.paint_uniform_color(color)
            colored_meshes.append(m_col)

    if len(global_points) > 0:
        global_points_arr = np.concatenate(global_points, axis=0)
        global_colors_arr = np.concatenate(global_colors, axis=0)
    else:
        global_points_arr = np.zeros((0, 3), dtype=np.float32)
        global_colors_arr = np.zeros((0, 3), dtype=np.float32)

    return global_points_arr, global_colors_arr, colored_meshes, link_ranges


def o3d_pick_link_global(global_points: np.ndarray,
                         global_colors: np.ndarray,
                         colored_meshes: List[o3d.geometry.TriangleMesh],
                         window_name: str = "Select a link (click a colored point)",
                         point_size: int = 4) -> List[int]:
    """Return picked point indices from the global combined point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points)
    if global_colors.shape[0] == global_points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(global_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.asarray([1, 1, 1])
    vis.add_geometry(pcd)  # first for picking
    for m in colored_meshes:
        vis.add_geometry(m)
    print("Global view: Shift+LeftClick a point to choose a link; Q to finish picking.")
    vis.run()
    idxs = vis.get_picked_points()
    vis.destroy_window()
    return idxs


def o3d_pick_points_with_context(points_xyz_world: np.ndarray,
                                 robot_meshes_world: List[o3d.geometry.TriangleMesh],
                                 highlight_meshes_world: List[o3d.geometry.TriangleMesh],
                                 window_name: str = "Pick points",
                                 point_size: int = 3) -> List[int]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz_world)
    # color is applied by caller via painting robot meshes
    pcd.paint_uniform_color([1.0, 0.2, 0.2])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)
    render_opt.background_color = np.asarray([1, 1, 1])

    vis.add_geometry(pcd)

    for m in robot_meshes_world:
        m_bg = o3d.geometry.TriangleMesh(m)
        vis.add_geometry(m_bg)

    for m in highlight_meshes_world:
        m_h = o3d.geometry.TriangleMesh(m)
        vis.add_geometry(m_h)

    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    return picked


def show_confirmation_with_context(points_xyz_world: np.ndarray,
                                   picked_idx: List[int],
                                   robot_meshes_world: List[o3d.geometry.TriangleMesh],
                                   highlight_meshes_world: List[o3d.geometry.TriangleMesh],
                                   window_name: str = "Confirm selection") -> None:
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points_xyz_world)
    colors = np.tile(np.array([[0.6, 0.0, 0.0]], dtype=float), (points_xyz_world.shape[0], 1))
    if len(picked_idx) > 0:
        colors[picked_idx] = np.array([0.0, 1.0, 0.0])
    pcd_all.colors = o3d.utility.Vector3dVector(colors)

    geometries: List[o3d.geometry.Geometry] = []
    geometries.extend(robot_meshes_world)
    geometries.extend(highlight_meshes_world)
    geometries.append(pcd_all)

    o3d.visualization.draw_geometries(geometries, window_name=window_name, point_show_normal=False)


def load_existing_keypoints(json_path: str) -> Dict[str, List[List[float]]]:
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    valid: Dict[str, List[List[float]]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            pts = []
            for p in v:
                if isinstance(p, list) and len(p) == 3:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
            valid[k] = pts
    return valid


def save_keypoints(json_path: str, keypoints: Dict[str, List[List[float]]]) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(keypoints, f, indent=2)
    print(f"[INFO] Saved keypoints to {json_path}")


def compute_fk_zero(urdf_path: str) -> Dict[str, np.ndarray]:
    """Compute link world transforms (4x4) with all joint positions set to zero.
    Returns dict: link_name -> T_world_link
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build graph: parent_link -> [(child_link, joint_origin_T)]
    children: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    all_links = set()
    parent_of: Dict[str, str] = {}

    for link in root.findall("link"):
        all_links.add(link.get("name", ""))

    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link") if joint.find("parent") is not None else None
        child = joint.find("child").get("link") if joint.find("child") is not None else None
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        R = rpy_to_matrix(*rpy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array(xyz)
        if parent not in children:
            children[parent] = []
        children[parent].append((child, T))
        parent_of[child] = parent

    # identify root link
    roots = [l for l in all_links if l not in parent_of]
    if not roots:
        roots = ["world"]

    T_world: Dict[str, np.ndarray] = {}

    def dfs(link: str, T_accum: np.ndarray):
        T_world[link] = T_accum
        for child, T_joint in children.get(link, []):
            dfs(child, T_accum @ T_joint)

    for r in roots:
        if r not in all_links:
            # Sometimes the root is an implicit world; start from its children
            for child, T_joint in children.get(r, []):
                dfs(child, np.eye(4) @ T_joint)
        else:
            dfs(r, np.eye(4))

    return T_world


def build_robot_world_meshes(urdf_path: str, base_dir: str) -> Tuple[Dict[str, List[trimesh.Trimesh]], Dict[str, List[o3d.geometry.TriangleMesh]]]:
    """Load all link collision meshes (link-local) and produce world-frame o3d meshes using FK at zero q."""
    all_collisions = parse_all_links_collisions(urdf_path)
    T_world = compute_fk_zero(urdf_path)

    link_to_local_meshes: Dict[str, List[trimesh.Trimesh]] = {}
    link_to_world_o3d: Dict[str, List[o3d.geometry.TriangleMesh]] = {}

    for link, cols in all_collisions.items():
        local_meshes: List[trimesh.Trimesh] = []
        world_meshes_o3d: List[o3d.geometry.TriangleMesh] = []
        for col in cols:
            try:
                m_local = load_collision_trimesh(base_dir, col)  # in link frame
            except Exception as e:
                print(f"[WARN] Failed to load collision for {link}: {e}")
                continue
            local_meshes.append(m_local)
            # transform to world
            T = T_world.get(link, np.eye(4))
            m_world = m_local.copy()
            m_world.apply_transform(T)
            world_meshes_o3d.append(trimesh_to_o3d(m_world))
        if local_meshes:
            link_to_local_meshes[link] = local_meshes
        if world_meshes_o3d:
            link_to_world_o3d[link] = world_meshes_o3d

    return link_to_local_meshes, link_to_world_o3d


def interactive_edit_points(link_name: str,
                             local_points: np.ndarray,
                             link_world_T: np.ndarray,
                             robot_world_meshes: List[o3d.geometry.TriangleMesh],
                             link_world_meshes: List[o3d.geometry.TriangleMesh],
                             link_color: Tuple[float, float, float]) -> List[List[float]]:
    """Interactive picking with full robot context and consistent link color."""
    R = link_world_T[:3, :3]
    t = link_world_T[:3, 3]
    points_world = (local_points @ R.T) + t

    robot_colored = []
    for m in robot_world_meshes:
        m_bg = o3d.geometry.TriangleMesh(m)
        m_bg.paint_uniform_color([0.85, 0.85, 0.85])
        robot_colored.append(m_bg)
    link_highlight = []
    for m in link_world_meshes:
        m_h = o3d.geometry.TriangleMesh(m)
        m_h.paint_uniform_color(link_color)
        link_highlight.append(m_h)

    print(f"\n=== Link: {link_name} ===")
    print("Shift + Left Click to pick points; press Q to finish.")

    picked_idx = o3d_pick_points_with_context(points_world, robot_colored, link_highlight, window_name=f"Pick: {link_name}")
    show_confirmation_with_context(points_world, picked_idx, robot_colored, link_highlight, window_name=f"Confirm: {link_name}")

    picked_local = local_points[picked_idx] if len(picked_idx) > 0 else np.zeros((0, 3))
    selected = [[float(x), float(y), float(z)] for x, y, z in picked_local]
    return selected


def main():
    parser = argparse.ArgumentParser(description="Interactive Allegro Hand keypoint selector")
    parser.add_argument("urdf", type=str, help="Path to xArm+Allegro URDF")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--include_palm", action="store_true", help="Include base_link/palm/wrist in selection")
    parser.add_argument("--points_per_link", type=int, default=500, help="Number of sampled points per link")
    parser.add_argument("--existing", type=str, default=None, help="Existing JSON to preload/edit")
    parser.add_argument("--require_fingers", action="store_true", help="Enforce at least one keypoint per finger link")
    parser.add_argument("--csv_output", type=str, default=None, help="Optional CSV output path for world coords with link name")
    args = parser.parse_args()

    urdf_path = args.urdf
    if not os.path.isfile(urdf_path):
        print(f"[ERROR] URDF not found: {urdf_path}")
        sys.exit(1)
    base_dir = os.path.dirname(os.path.abspath(urdf_path))

    link_to_local_meshes_all, link_to_world_meshes_all = build_robot_world_meshes(urdf_path, base_dir)
    T_world = compute_fk_zero(urdf_path)

    # Determine hand links to edit
    hand_links = parse_urdf_for_hand_links(urdf_path, include_palm=args.include_palm)
    if len(hand_links) == 0:
        print("[ERROR] No hand links with collision found.")
        sys.exit(1)
    hand_link_names = [l.name for l in hand_links]

    link_to_local_points: Dict[str, np.ndarray] = {}
    for link in hand_link_names:
        local_meshes = link_to_local_meshes_all.get(link, [])
        if not local_meshes:
            continue
        try:
            union = trimesh.util.concatenate(local_meshes)
            pts, _ = trimesh.sample.sample_surface(union, args.points_per_link)
            link_to_local_points[link] = pts.astype(np.float32)
        except Exception:
            link_to_local_points[link] = sample_points_from_link(local_meshes, points_per_link=args.points_per_link)

    global_points, global_colors, colored_meshes, link_ranges = build_global_points_and_meshes(
        hand_link_names, link_to_world_meshes_all, link_to_local_points, T_world
    )

    keypoints: Dict[str, List[List[float]]] = {}
    if args.existing:
        keypoints.update(load_existing_keypoints(args.existing))
        print(f"[INFO] Loaded existing keypoints from {args.existing}")


    picked_indices = o3d_pick_link_global(global_points, global_colors, colored_meshes)
    if not picked_indices:
        print("[INFO] No points picked in global view. Nothing to save.")
        # Still enforce requirement if requested
        if args.require_fingers:
            missing = [ln for ln in hand_link_names if ln.startswith("link_") and len(keypoints.get(ln, [])) == 0]
            if missing:
                print("[ERROR] Missing keypoints for finger links:")
                for m in missing:
                    print(f"  - {m}")
                sys.exit(2)

        out_path = args.output
        if out_path is None:
            robot_name = os.path.splitext(os.path.basename(urdf_path))[0]
            out_path = os.path.join(base_dir, f"{robot_name}_keypoints.json")
        save_keypoints(out_path, keypoints)
        if args.csv_output:
            with open(args.csv_output, "w") as f:
                f.write("link_name,x,y,z\n")
            print(f"[INFO] Saved CSV (empty) to {args.csv_output}")
        print("Done.")
        return

    csv_rows: List[Tuple[str, float, float, float]] = []
    for idx in picked_indices:
        # Identify link based on range
        link_name = None
        for ln, (s, e) in link_ranges.items():
            if s <= idx < e:
                link_name = ln
                break
        if link_name is None:
            continue
        local_start, _ = link_ranges[link_name]
        local_idx = idx - local_start
        pts_local = link_to_local_points.get(link_name, None)
        if pts_local is None or local_idx < 0 or local_idx >= pts_local.shape[0]:
            continue
        pt_local = pts_local[local_idx]
        keypoints.setdefault(link_name, []).append([float(pt_local[0]), float(pt_local[1]), float(pt_local[2])])
        if args.csv_output:
            pt_world = global_points[idx]
            csv_rows.append((link_name, float(pt_world[0]), float(pt_world[1]), float(pt_world[2])))

    if args.require_fingers:
        missing = [ln for ln in hand_link_names if ln.startswith("link_") and len(keypoints.get(ln, [])) == 0] # XXX
        if missing:
            print("[ERROR] Missing keypoints for finger links:")
            for m in missing:
                print(f"  - {m}")
            sys.exit(2)

    out_path = args.output
    if out_path is None:
        robot_name = os.path.splitext(os.path.basename(urdf_path))[0]
        out_path = os.path.join(base_dir, f"{robot_name}_keypoints.json")
    save_keypoints(out_path, keypoints)

    if args.csv_output:
        import csv
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["link_name", "x", "y", "z"])
            for row in csv_rows:
                writer.writerow(row)
        print(f"[INFO] Saved CSV to {args.csv_output}")

    print("Done.")


if __name__ == "__main__":
    main() 