#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def load_density(path):
    d = torch.load(path, map_location="cpu")
    return {
        "counts": d["counts"].numpy(),  # (4, X, Y, Z)
        "voxel_size": float(d["voxel_size"]),
        "min_bound": d["min_bound"].numpy(),  # (3,)
        "dims": d["dims"],
        "tag": d.get("tag", os.path.basename(path)),
        "fingertip_order": d.get("fingertip_order", ["index", "middle", "ring", "thumb"]),
    }

def dens_to_points(counts, voxel_size, min_bound, thresh=0):
    X, Y, Z = counts.shape
    ii, jj, kk = np.nonzero(counts > thresh)
    vals = counts[ii, jj, kk].astype(np.float32)
    centers = np.stack([ii + 0.5, jj + 0.5, kk + 0.5], axis=1) * voxel_size + min_bound
    return centers, vals

def plot_density(ax, centers, vals, title):
    if centers.shape[0] == 0:
        ax.set_title(f"{title} (no points)")
        return
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    colors = plt.cm.viridis(vals_norm)
    ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=colors, s=6, alpha=0.6, marker='s')
    ax.set_title(title)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    
def plot_density_comparison(ax, centers, vals, max, min, title):
    vals_norm = (vals - min) / (max - min + 1e-6)
    colors = plt.cm.viridis(vals_norm)
    ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=colors, s=6, alpha=0.6, marker='s')
    ax.set_title(title)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    # ax.set_box_aspect([0.25,1,1.5])
    ax.set_box_aspect([1,1,1])

    
    # Add a colorbar showing the density scale
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    norm = Normalize(vmin=min, vmax=max)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    
    # Create a custom colorbar showing density vs transparency
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Density (count)', rotation=270, labelpad=15)
    
    # Add text showing the actual min/max values
    ax.text2D(0.02, 0.98, f'Min: {vals.min()}, Max: {vals.max()}, Mean: {vals.mean().round(0)}', 
              transform=ax.transAxes, fontsize=8, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_density_comparison_alpha(ax, centers, vals, max_val, min_val, title):
    assert centers.shape[0] != 0
    
    alpha_min, alpha_max = 0.1, 1.0
    alpha_vals = alpha_min + (alpha_max - alpha_min) * (vals - min_val) / (max_val - min_val + 1e-6)
    
    # Fast approach: Use a single scatter call with RGBA colors
    # Convert to RGBA where the alpha channel represents density
    rgba_colors = np.zeros((len(centers), 4))
    rgba_colors[:, :3] = [0.2, 0.6, 0.9]  # Blue color for all points
    rgba_colors[:, 3] = alpha_vals  # Alpha channel for density
    
    # Single scatter call - much faster
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
               c=rgba_colors, s=8, marker='s', edgecolors='none')
    
    ax.set_title(f"{title}")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    norm = Normalize(vmin=min_val, vmax=max_val)
    sm = ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])
    
    # Create a custom colorbar showing density vs transparency
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Density (count)', rotation=270, labelpad=15)
    
    # Add text showing the actual min/max values
    ax.text2D(0.02, 0.98, f'Min: {vals.min()}, Max: {vals.max()}', 
              transform=ax.transAxes, fontsize=8, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main(a, b, c, fingertip="all"):
    da, db, dc = load_density(a), load_density(b), load_density(c)
    assert da["dims"] == db["dims"]
    assert abs(da["voxel_size"] - db["voxel_size"]) < 1e-9
    assert np.allclose(da["min_bound"], db["min_bound"])
    assert da["dims"] == dc["dims"]
    assert abs(da["voxel_size"] - dc["voxel_size"]) < 1e-9
    assert np.allclose(da["min_bound"], dc["min_bound"])

    fingers = da["fingertip_order"]
    indexes = range(4) if fingertip == "all" else [fingers.index(fingertip)]

    fig = plt.figure(figsize=(10, 5*len(indexes)))
    for row, fidx in enumerate(indexes):
        ca = da["counts"][fidx]  # (X,Y,Z)
        cb = db["counts"][fidx]
        cc = dc["counts"][fidx]
        cmax = max(ca.max(), cb.max(), cc.max())
        # normalize
        centers_a, vals_a = dens_to_points(ca, da["voxel_size"], da["min_bound"])
        centers_b, vals_b = dens_to_points(cb, db["voxel_size"], db["min_bound"])
        centers_c, vals_c = dens_to_points(cc, dc["voxel_size"], dc["min_bound"])
        max_val = max(vals_a.max(), vals_b.max(), vals_c.max())
        min_val = min(vals_a.min(), vals_b.min(), vals_c.min())
        print(f"max_val: {max_val}, min_val: {min_val}")

        ax1 = fig.add_subplot(len(indexes), 3, row*3+1, projection='3d')
        # plot_density(ax1, centers_a, vals_a, f"{da['tag']} - {fingers[fidx]}")
        plot_density_comparison(ax1, centers_a, vals_a, max_val, min_val, f"reach_only - {fingers[fidx]}")
        # plot_density_comparison_alpha(ax1, centers_a, vals_a, max_val, min_val, f"reach_only - {fingers[fidx]}")

        ax2 = fig.add_subplot(len(indexes), 3, row*3+2, projection='3d')
        # plot_density(ax2, centers_b, vals_b, f"{db['tag']} - {fingers[fidx]}")
        plot_density_comparison(ax2, centers_b, vals_b, max_val, min_val, f"reach_curiosity - {fingers[fidx]}")
        # plot_density_comparison_alpha(ax2, centers_b, vals_b, max_val, min_val, f"reach_curiosity - {fingers[fidx]}")

        ax3 = fig.add_subplot(len(indexes), 3, row*3+3, projection='3d')
        plot_density_comparison(ax3, centers_c, vals_c, max_val, min_val, f"task_curiosity - {fingers[fidx]}")
        # plot_density_comparison_alpha(ax3, centers_c, vals_c, max_val, min_val, f"task_curiosity - {fingers[fidx]}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reach_only", type=str, required=True)
    p.add_argument("--reach_curiosity", type=str, required=True)
    p.add_argument("--task_curiosity", type=str, required=True)
    p.add_argument("--fingertip", type=str, default="all", choices=["all","index","middle","ring","thumb"])
    args = p.parse_args()
    main(args.reach_only, args.reach_curiosity, args.task_curiosity, args.fingertip)
    
# python scripts/vis_exploration_density.py --reach_only logs/PPO/08-22-11-51_xarm_allegro_singulation_ppo_w_curiosity_40_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42/exploration_data_2000.pt --reach_curiosity /home/ruoyi/Work/UniDexFPM/logs/PPO/08-24-23-10_xarm_allegro_singulation_ppo_w_curiosity_40_use_random_strict_pick_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42/exploration_data_2000.pt --task_curiosity /home/ruoyi/Work/UniDexFPM/logs/PPO/08-25-01-05_xarm_allegro_singulation_ppo_w_curiosity_40_use_random_strict_pick_objtype:all_labeltype:box_grid_singulation_objnum:5_objcat:box_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+tilt+slide+neighbor+actionpen_seed42/exploration_data_2000.pt  --fingertip thumb