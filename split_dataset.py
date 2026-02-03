import os
import shutil
import random
import glob
import numpy as np
from tqdm import tqdm

def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    # Get all .npz files in the source directory
    # We only look for files in the top level based on the `ls` output seen previously
    files = [f for f in os.listdir(source_dir) if f.endswith('.npz')]
    files.sort()  # Sort to ensure deterministic shuffling with seed
    
    print(f"Found {len(files)} .npz files in {source_dir}")
    
    if len(files) == 0:
        print("No .npz files found.")
        return

    # Shuffle files
    random.seed(seed)
    random.shuffle(files)
    
    # Calculate split indices
    total_files = len(files)
    n_train = int(total_files * train_ratio)
    n_val = int(total_files * val_ratio)
    n_test = total_files - n_train - n_val
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Create subdirectories
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(source_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Created directory: {split_dir}")

    # Move files
    def move_files(file_list, destination_name):
        dest_dir = os.path.join(source_dir, destination_name)
        print(f"Moving files to {destination_name}...")
        for f in tqdm(file_list):
            src_path = os.path.join(source_dir, f)
            dst_path = os.path.join(dest_dir, f)
            shutil.move(src_path, dst_path)
            
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')
    
    print("Dataset split completed successfully.")

if __name__ == "__main__":
    dataset_path = "/data0/Hand2GripperDatasets/Hand2Gripper_TrainValTest_Dataset"
    split_dataset(dataset_path)
