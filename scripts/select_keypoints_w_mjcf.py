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


# === [ADD] 扩展 CollisionGeometry：增加 mesh_scale ===
@dataclass
class CollisionGeometry:
    kind: str  # 'mesh' or 'box' or 'capsule'
    mesh_path: Optional[str] = None
    box_size: Optional[Tuple[float, float, float]] = None  # (x,y,z) full extents
    capsule_radius: Optional[float] = None                # NEW
    capsule_half_length: Optional[float] = None           # NEW
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_scale: Optional[Tuple[float, float, float]] = None
    

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


# === [ADD] 旋转/格式检测工具 ===
def detect_model_format(path: str) -> str:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return "unknown"
    tag = root.tag.lower()
    if "mujoco" in tag:
        return "mjcf"
    if tag == "robot":
        return "urdf"
    if tag == "sdf":
        return "sdf"
    return "unknown"

def euler_xyz_to_matrix(ex: float, ey: float, ez: float) -> np.ndarray:
    """MuJoCo 的 euler 通常为 x->y->z 顺序。"""
    cx, sx = math.cos(ex), math.sin(ex)
    cy, sy = math.cos(ey), math.sin(ey)
    cz, sz = math.cos(ez), math.sin(ez)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rx @ Ry @ Rz

def quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """MuJoCo 的 quat 默认是 w x y z。"""
    n = math.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])

def axisangle_to_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    theta = math.sqrt(ax*ax + ay*ay + az*az)
    if theta < 1e-12:
        return np.eye(3)
    ux, uy, uz = ax/theta, ay/theta, az/theta
    c, s = math.cos(theta), math.sin(theta)
    C = 1 - c
    return np.array([
        [c+ux*ux*C, ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c+uy*uy*C, uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c+uz*uz*C],
    ])

def matrix_to_rpy_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """把任意旋转矩阵换算到 URDF 的 rpy (roll=X, pitch=Y, yaw=Z) 顺序 Rz*Ry*Rx。"""
    # pitch
    sy = -R[2,0]
    if abs(sy) < 0.999999:
        pitch = math.asin(sy)
        roll  = math.atan2(R[2,1], R[2,2])
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        # 近奇异：用另一套
        pitch = math.copysign(math.pi/2, sy)
        roll  = 0.0
        yaw   = math.atan2(-R[0,1], R[1,1])
    return (roll, pitch, yaw)


# === [ADD] 解析 MJCF 资产 <asset><mesh ...> ===
def _mjcf_parse_assets(root: ET.Element, base_dir: str):
    """返回: name -> {'file': abs_path, 'scale': (sx,sy,sz)}"""
    assets = {}
    for m in root.findall("mesh"):
        name = m.get("name")
        if not name:
            continue
        file_attr = m.get("file", "")
        # 解析 scale
        s_attr = m.get("scale")
        if s_attr:
            try:
                sx, sy, sz = [float(v) for v in s_attr.split()]
            except:
                sx = sy = sz = 1.0
        else:
            sx = sy = sz = 1.0
        # 路径解析
        resolved = file_attr
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.path.dirname(base_dir), "stls", "hand", file_attr)
        assets[name] = {"file": resolved, "scale": (sx, sy, sz)}
    return assets

# === [ADD] 解析 MJCF 节点姿态（body/geom 通用）===
def _mjcf_pose_T(elem: ET.Element) -> np.ndarray:
    pos = elem.get("pos")
    if pos:
        try:
            px, py, pz = [float(v) for v in pos.split()]
        except:
            px = py = pz = 0.0
    else:
        px = py = pz = 0.0

    # 优先 quat，再 euler，再 axisangle
    if elem.get("quat"):
        w, x, y, z = [float(v) for v in elem.get("quat").split()]
        R = quat_wxyz_to_matrix(w, x, y, z)
    elif elem.get("euler"):
        ex, ey, ez = [float(v) for v in elem.get("euler").split()]
        # 如果像度数，自动转弧度
        if max(abs(ex), abs(ey), abs(ez)) > 2*math.pi + 1e-6:
            ex, ey, ez = math.radians(ex), math.radians(ey), math.radians(ez)
        R = euler_xyz_to_matrix(ex, ey, ez)
    elif elem.get("axisangle"):
        ax, ay, az, ang = [float(v) for v in elem.get("axisangle").split()]
        # 归一化旋转轴
        norm = math.sqrt(ax*ax + ay*ay + az*az) + 1e-12
        ux, uy, uz = ax/norm, ay/norm, az/norm
        R = axisangle_to_matrix(ux*ang, uy*ang, uz*ang)
    else:
        R = np.eye(3)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = np.array([px, py, pz], dtype=float)
    return T

# === [ADD] 解析 MJCF：收集 link(=body) -> [CollisionGeometry] ===
def parse_all_links_collisions_mjcf(xml_path: str) -> Dict[str, List[CollisionGeometry]]:
    root = ET.parse(xml_path).getroot()
    base_dir = os.path.dirname(os.path.abspath(xml_path))
    asset_file_path = os.path.join(base_dir, "shared_asset.xml")
    asset_root = ET.parse(asset_file_path).getroot()
    assets = _mjcf_parse_assets(asset_root, base_dir)

    all_links: Dict[str, List[CollisionGeometry]] = {}

    def dfs_body(body_elem: ET.Element):
        name = body_elem.get("name", "")
        # T_body = _mjcf_pose_T(body_elem)
        # 收集 geoms
        colls: List[CollisionGeometry] = []
        for g in body_elem.findall("geom"):
            if g.get("class", "Vizual").endswith("Vizual"):
                continue
            gtype = g.get("type", "mesh")
            print("name: ", name, " gtype: ", gtype)
            # print(body_elem.findall("geom"))
            # 几何相对于 body 的局部位姿
            T_local = _mjcf_pose_T(g)
            # T_local = T_body @ T_local
            R_local = T_local[:3,:3]
            rpy_local = matrix_to_rpy_zyx(R_local)
            xyz_local = tuple(map(float, T_local[:3,3]))

            if gtype == "box" and g.get("size"):
                sx, sy, sz = [float(v) for v in g.get("size").split()]
                # MuJoCo 的 box size 是半长度；Trimesh 需要全尺寸
                box_extents = (2*sx, 2*sy, 2*sz)
                colls.append(CollisionGeometry(
                    kind="box",
                    box_size=box_extents,
                    origin_xyz=xyz_local,
                    origin_rpy=rpy_local,
                ))
            elif gtype == "capsule" and g.get("size"):
                # MJCF capsule: size="radius half_length"
                r, h = [float(v) for v in g.get("size").split()]
                colls.append(CollisionGeometry(
                    kind="capsule",
                    capsule_radius=r,
                    capsule_half_length=h,
                    origin_xyz=xyz_local,
                    origin_rpy=rpy_local,
                ))
            elif gtype == "mesh" and g.get("mesh"):
                mesh_name = g.get("mesh")
                asset = assets.get(mesh_name)
                # 组合 asset.scale 与 geom.scale（若有）
                asx, asy, asz = asset["scale"]
                gs_attr = g.get("scale")
                if gs_attr:
                    try:
                        gsx, gsy, gsz = [float(v) for v in gs_attr.split()]
                    except:
                        gsx = gsy = gsz = 1.0
                else:
                    gsx = gsy = gsz = 1.0
                scale = (asx*gsx, asy*gsy, asz*gsz)
                colls.append(CollisionGeometry(
                    kind="mesh",
                    mesh_path=asset["file"],
                    origin_xyz=xyz_local,
                    origin_rpy=rpy_local,
                    mesh_scale=scale,
                ))
            else:
                # 其他类型（capsule/cylinder/sphere 等）按需扩展
                pass

        if colls:
            all_links[name] = colls
        for child in body_elem.findall("body"):
            dfs_body(child)

    for b in root.findall("body"):
        dfs_body(b)

    return all_links

# === [ADD] 计算 MJCF 的零位 FK：返回 body_name -> T_world ===
def compute_fk_zero_mjcf(xml_path: str) -> Dict[str, np.ndarray]:
    root = ET.parse(xml_path).getroot()
    T_world: Dict[str, np.ndarray] = {}

    def dfs_body(body_elem: ET.Element, T_parent: np.ndarray):
        name = body_elem.get("name", "")
        T_body_local = _mjcf_pose_T(body_elem)
        T_here = T_parent @ T_body_local
        if name:
            T_world[name] = T_here
        for child in body_elem.findall("body"):
            dfs_body(child, T_here)

    # 顶层 body 相对 world
    for b in root.findall("body"):
        dfs_body(b, np.eye(4))
    return T_world


def load_collision_trimesh(base_dir: str, col: CollisionGeometry) -> trimesh.Trimesh:
    if col.kind == "mesh":
        mesh_path = col.mesh_path
        assert mesh_path is not None
        resolved = mesh_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(base_dir, mesh_path)
            print("base_dir: ", base_dir)
        if not os.path.isfile(resolved):
            alt = os.path.join(base_dir, os.path.basename(mesh_path))
            if os.path.isfile(alt):
                resolved = alt
            elif base_dir.split("/")[-1] == (mesh_path.split("/")[0] if mesh_path else ""):
                alt = os.path.join(base_dir, *mesh_path.split("/")[1:])
            if os.path.isfile(alt):
                resolved = alt
            else:
                raise FileNotFoundError(f"Mesh file not found: {mesh_path} (resolved: {resolved})")
        mesh = trimesh.load(resolved, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
        # 应用 mesh_scale（来自 MJCF 的 asset.scale/geom.scale）
        if col.mesh_scale is not None:
            sx, sy, sz = col.mesh_scale
            S = np.eye(4, dtype=float)
            S[0,0], S[1,1], S[2,2] = sx, sy, sz
            mesh = mesh.copy()
            mesh.apply_transform(S)
            
    elif col.kind == "box":
        assert col.box_size is not None
        size = np.array(col.box_size, dtype=float)
        mesh = trimesh.creation.box(extents=size)

    elif col.kind == "capsule":
        assert (col.capsule_radius is not None) and (col.capsule_half_length is not None)
        radius = float(col.capsule_radius)
        half   = float(col.capsule_half_length)
        # trimesh 的 capsule(height=圆柱段长度)，MJCF 的 half_length 是半长
        height = 2.0 * half
        mesh = trimesh.creation.capsule(radius=radius, height=height)

    # 应用局部 origin（保留在 link/body 局部坐标系）
    R = rpy_to_matrix(*col.origin_rpy)
    t = np.array(col.origin_xyz, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh


# === [ADD] MJCF 版：返回 HandLink 列表（按你的命名过滤逻辑）===
def parse_mjcf_for_hand_links(xml_path: str, include_palm: bool=False) -> List[HandLink]:

    def is_hand_name(name: str) -> bool:
        # 你的命名：ff/mf/rf/lf（四指），th（thumb），以及 palm/wrist/forearm
        prefixes = ("robot0:ff", "robot0:mf", "robot0:rf", "robot0:lf", "robot0:th")
        if name.startswith(prefixes):
            return True
        if include_palm and name.startswith("robot0:palm"):
            return True
        return False


    all_coll = parse_all_links_collisions_mjcf(xml_path)
    print("len of all_coll: ", len(all_coll))
    links: List[HandLink] = []
    for name, cols in all_coll.items():
        print("name: ", name)
        if is_hand_name(name):
            links.append(HandLink(name=name, collisions=cols))

    if len([l for l in links if any(k in l.name for k in ["index","middle","ring","thumb"])]) == 0:
        print("[WARN] No finger bodies found in MJCF by naming. Check naming conventions.")
        # 不 exit，允许仅 palm 等

    return links


def parse_urdf_for_hand_links(urdf_path: str, include_palm: bool = False) -> List[HandLink]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # def is_hand_link(name: str) -> bool:
    #     if name.startswith("link_base"): # xarm base link
    #         return False
    #     if name.startswith("link_"):
    #         return True
    #     if include_palm and name in {"palm", "base_link"}:
    #         return True
    #     # exclude xArm links: link1..link6
    #     if name in {"wrist"} and include_palm:
    #         return True
    #     return False
    
    def is_hand_link(name: str) -> bool:
        if name.startswith("index") or name.startswith("middle") or name.startswith("ring") or name.startswith("thumb"):
            return True
        if include_palm and name.startswith("palm"):
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
    # if len([l for l in links if l.name.startswith("link_")]) == 0:
    #     print("[WARN] No Allegro finger links found in URDF. Check naming conventions.")
    #     exit()
    if len([l for l in links if "link" in l.name]) == 0:
        print("[WARN] No Allegro finger links found in URDF. Check naming conventions.")
        exit()
    return links


# === [ADD] 自动分发包装（URDF/MJCF）===
def parse_hand_links_auto(model_path: str, include_palm: bool=False) -> List[HandLink]:
    fmt = detect_model_format(model_path)
    if fmt == "mjcf":
        return parse_mjcf_for_hand_links(model_path, include_palm=include_palm)
    else:
        # 默认按 URDF 走（含 URDF）
        return parse_urdf_for_hand_links(model_path, include_palm=include_palm)


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


# === [REPLACE] build_robot_world_meshes：自动分发 URDF / MJCF ===
def build_robot_world_meshes(model_path: str, base_dir: str) -> Tuple[Dict[str, List[trimesh.Trimesh]], Dict[str, List[o3d.geometry.TriangleMesh]]]:
    fmt = detect_model_format(model_path)
    if fmt == "mjcf":
        all_collisions = parse_all_links_collisions_mjcf(model_path)
        T_world = compute_fk_zero_mjcf(model_path)
    else:
        all_collisions = parse_all_links_collisions(model_path)
        T_world = compute_fk_zero(model_path)

    link_to_local_meshes: Dict[str, List[trimesh.Trimesh]] = {}
    link_to_world_o3d: Dict[str, List[o3d.geometry.TriangleMesh]] = {}

    for link, cols in all_collisions.items():
        local_meshes: List[trimesh.Trimesh] = []
        world_meshes_o3d: List[o3d.geometry.TriangleMesh] = []
        for col in cols:
            try:
                m_local = load_collision_trimesh(base_dir, col)  # link/body 局部
            except Exception as e:
                print(f"[WARN] Failed to load collision for {link}: {e}")
                continue
            local_meshes.append(m_local)
            # 到世界坐标
            T = T_world.get(link, np.eye(4))
            m_world = m_local.copy()
            m_world.apply_transform(T)
            world_meshes_o3d.append(trimesh_to_o3d(m_world))
        if local_meshes:
            link_to_local_meshes[link] = local_meshes
        if world_meshes_o3d:
            link_to_world_o3d[link] = world_meshes_o3d

    return link_to_local_meshes, link_to_world_o3d


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3 = o3d.geometry.TriangleMesh()
    o3.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3.compute_vertex_normals()
    return o3


import re

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return float(r), float(g), float(b)

def hash_color(name: str) -> Tuple[float, float, float]:
    """
    为 link 名生成可区分的颜色（0..1）。
    - 名字哈希决定 base hue（稳定、与不同前缀区分）
    - 若名字以数字结尾（如 *_12），用黄金比例步进分散色相，并交替亮度/饱和度
    """
    # 1) 32-bit 稳定哈希（FNV-1a 简化版）
    h = 2166136261
    for ch in name:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    base = (h / 2**32)  # 0..1

    # 2) 解析末尾序号（index_link_0 -> k=0）
    m = re.search(r'(\d+)$', name)
    k = int(m.group(1)) if m else -1

    # 3) 色相：base + 黄金比例步进（避免相邻色相过近）
    phi = 0.6180339887498949
    hue = (base + (k if k >= 0 else 0) * phi) % 1.0

    # 4) 亮度/饱和度交替：相邻序号即使色相接近也能区分
    if k >= 0:
        # 三态循环能更稳：高亮/中亮/低亮 + 饱和度微调
        mode = k % 3
        if mode == 0:
            sat, val = 0.70, 0.95
        elif mode == 1:
            sat, val = 0.85, 0.80
        else:
            sat, val = 0.65, 0.70
    else:
        # 无序号：给个通用的、较鲜明的默认
        sat, val = 0.70, 0.92

    return _hsv_to_rgb(hue, sat, val)


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


def save_keypoints(json_path: str, keypoints: Dict[str, List[List[float]]]) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(keypoints, f, indent=2)
    print(f"[INFO] Saved keypoints to {json_path}")
    
    
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

    model_path = args.urdf
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    base_dir = os.path.dirname(os.path.abspath(model_path))

    link_to_local_meshes_all, link_to_world_meshes_all = build_robot_world_meshes(model_path, base_dir)
    T_world = compute_fk_zero_mjcf(model_path) if detect_model_format(model_path) == "mjcf" else compute_fk_zero(model_path)

    hand_links = parse_hand_links_auto(model_path, include_palm=args.include_palm)
    
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
            robot_name = os.path.splitext(os.path.basename(model_path))[0]
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
        robot_name = os.path.splitext(os.path.basename(model_path))[0]
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