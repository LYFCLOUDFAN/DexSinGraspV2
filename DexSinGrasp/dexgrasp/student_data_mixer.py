import os
import shutil
import random

def mix_student_data():
    base_path = '/data/DexSinGrasp/Logs/Results/results_trajectory_render'
    student_path = os.path.join(base_path, '0000_seed0_student')
    
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist")
        return
        
    # Create student folders
    os.makedirs(os.path.join(student_path, 'pointcloud'), exist_ok=True) 
    os.makedirs(os.path.join(student_path, 'trajectory'), exist_ok=True)

    # Define source folders and number of files to take from each
    source_folders = {
        '0000_seed0_expert2_obj6': 20,  # 20 files
        '0000_seed0_expert2_obj8': 20,  # 20 files
        '0000_seed0_expert3_obj6': 20,  # 20 files
        '0000_seed0_expert3_obj8': 20,  # 20 files
        '0000_seed0_expert2_obj4': 10,  # 10 files
        '0000_seed0_expert3_obj4': 10   # 10 files
    }

    # Track files to copy
    files_to_copy = []
    
    # Select files from each source folder
    for folder, num_files in source_folders.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder} does not exist")
            continue
            
        # Get list of indices (0-19)
        selected_indices = random.sample(range(20), num_files)
        print(selected_indices)
        
        # Add selected files to copy list
        for idx in selected_indices:
            files_to_copy.append({
                'src_folder': folder,
                'idx': idx
            })
    
    # Shuffle files_to_copy to randomize order
    random.shuffle(files_to_copy)

    # exit()
    
    # Copy files to student folder with new indices
    for new_idx, file_info in enumerate(files_to_copy):
        src_folder = file_info['src_folder']
        old_idx = file_info['idx']
        
        # Copy pointcloud file
        src_pc = os.path.join(base_path, src_folder, 'pointcloud', f'pointcloud_{old_idx:03d}.pkl')
        dst_pc = os.path.join(student_path, 'pointcloud', f'pointcloud_{new_idx:03d}.pkl')
        shutil.copy2(src_pc, dst_pc)
        
        # Copy trajectory file
        src_traj = os.path.join(base_path, src_folder, 'trajectory', f'trajectory_{old_idx:03d}.pkl')
        dst_traj = os.path.join(student_path, 'trajectory', f'trajectory_{new_idx:03d}.pkl')
        shutil.copy2(src_traj, dst_traj)
        
    print(f"Created student dataset with {len(files_to_copy)} files")

if __name__ == "__main__":
    mix_student_data()
