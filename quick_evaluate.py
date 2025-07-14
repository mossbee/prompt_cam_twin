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

# Create the command
command = f"""
python evaluate_twin_verification.py \\
    --checkpoint {checkpoint_path} \\
    --config {config_path} \\
    --output_dir {output_dir}
"""

print("🚀 Running Twin Face Verification Evaluation...")
print(f"Command: {command}")

# Execute
os.system(command.replace('\\', '').replace('\n', ' '))
