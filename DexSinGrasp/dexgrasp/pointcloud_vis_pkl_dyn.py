import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Load the pkl file
with open('/data/DexSinGrasp/Logs/Results/results_trajectory_render/0000_seed0_expert2_obj8/pointcloud/pointcloud_010.pkl', 'rb') as f:
    data = pickle.load(f)

# Print all keys in the data dictionary
print("Keys in data dictionary:")
for key in data.keys():
    print(f"- {key}")

# Print shape information for each key
print("\nShape information:")
for key, value in data.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value.shape}")
    elif isinstance(value, list):
        print(f"{key}: list of length {len(value)}")
    else:
        print(f"{key}: {type(value)}")

# Get max time steps from data shape
max_time_steps = data['rendered'].shape[1]

# Create figure with 4 subplots
fig = plt.figure(figsize=(8, 8))

# 3D view
ax1 = fig.add_subplot(221, projection='3d')
# Front view (XZ)
ax2 = fig.add_subplot(222)
# Top view (XY) 
ax3 = fig.add_subplot(223)
# Side view (YZ)
ax4 = fig.add_subplot(224)

# Function to update plots
def update_plots(time_step):
    # Clear all plots
    ax1.cla()
    ax2.cla() 
    ax3.cla()
    ax4.cla()
    
    # Get point cloud data for current time step
    points = data['rendered'][8, time_step]
    # print(points.shape)
    
    # 3D view
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='g', marker='o', s=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')
    ax1.set_xlim([-0.3, 0.3])
    ax1.set_ylim([-0.3, 0.3])
    ax1.set_zlim([0.5, 1.1])
    
    # Front view (XZ)
    ax2.scatter(points[:, 0], points[:, 2], c='g', marker='o', s=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Front View (XZ)')
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([0.5, 1.1])
    ax2.grid(True)
    
    # Top view (XY)
    ax3.scatter(points[:, 0], points[:, 1], c='g', marker='o', s=2)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Top View (XY)')
    ax3.set_xlim([-0.3, 0.3])
    ax3.set_ylim([-0.3, 0.3])
    ax3.grid(True)
    
    # Side view (YZ)
    ax4.scatter(points[:, 1], points[:, 2], c='g', marker='o', s=2)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Side View (YZ)')
    ax4.set_xlim([-0.3, 0.3])
    ax4.set_ylim([0.5, 1.1])
    ax4.grid(True)
    
    plt.suptitle(f'Time Step: {time_step}', y=0.95)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# Display animation
for t in range(max_time_steps):
    update_plots(t)
    # time.sleep(0.01)

plt.show()
