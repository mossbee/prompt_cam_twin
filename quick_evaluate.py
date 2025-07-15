#!/usr/bin/env python3
"""
Quick evaluation script for Kaggle environment
"""

import os
import sys

# For Kaggle environment - adjust paths
os.chdir('/kaggle/working/prompt_cam_twin')
sys.path.append('/kaggle/working/prompt_cam_twin')

# Run evaluation
checkpoint_path = "/kaggle/input/nd-twin/checkpoint_stage2_best_epoch_2.pth"  # Adjust this path
config_path = "experiment/config/twin_verification/dinov2/args.yaml"
output_dir = "/kaggle/working/evaluation_results"
data_dir = "data"  # JSON files are in the repo (working directory)

# Create the command - data_dir points to JSON files in working directory
command = f"""
python evaluate_twin_verification.py \\
    --checkpoint {checkpoint_path} \\
    --config {config_path} \\
    --output_dir {output_dir} \\
    --data_dir {data_dir}
"""

print("🚀 Running Twin Face Verification Evaluation...")
print(f"📁 Checkpoint: {checkpoint_path}")
print(f"⚙️  Config: {config_path}")  
print(f"💾 Output: {output_dir}")
print(f"📊 Data Dir: {data_dir}")
print(f"Command: {command}")

# Check if data files exist
test_info_file = os.path.join(data_dir, "test_dataset_infor.json")
test_pairs_file = os.path.join(data_dir, "test_twin_pairs.json")
train_info_file = os.path.join(data_dir, "train_dataset_infor.json")

print(f"\n📋 Checking data files...")
print(f"   Train info: {train_info_file} - {'✅ Found' if os.path.exists(train_info_file) else '❌ Missing'}")
print(f"   Test info: {test_info_file} - {'✅ Found' if os.path.exists(test_info_file) else '❌ Missing'}")
print(f"   Test pairs: {test_pairs_file} - {'✅ Found' if os.path.exists(test_pairs_file) else '❌ Missing'}")
print(f"   Checkpoint: {checkpoint_path} - {'✅ Found' if os.path.exists(checkpoint_path) else '❌ Missing'}")

print(f"\n🎯 Running evaluation...")

# Execute
os.system(command.replace('\\', '').replace('\n', ' '))
