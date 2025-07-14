#!/usr/bin/env python3
"""
Simple test to check if the TwinPromptCAM model can do a basic forward pass
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append('/home/mossbee/Work/Kaggle/Prompt_CAM')

def test_basic_forward():
    print("Testing basic forward pass...")
    
    try:
        from model.twin_prompt_cam import TwinPromptCAM, TwinPromptCAMConfig
        
        # Create a simple config
        config = TwinPromptCAMConfig(
            model='dinov2',
            drop_path_rate=0.1,
            vpt_num=1,
            vpt_mode='deep',
            stage1_training=True
        )
        
        # Create model
        print("Creating model...")
        model = TwinPromptCAM(config, num_persons=10)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        person_indices = torch.tensor([0, 1])
        
        print("Testing extract_features...")
        with torch.no_grad():
            features = model.extract_features(images, person_indices)
            print(f"Features shape: {features.shape}")
            
        print("Testing identity classification forward...")
        with torch.no_grad():
            outputs = model(images, images, person_indices, person_indices, mode='identity_classification')
            print(f"Identity outputs shape: {outputs.shape}")
            
        print("Testing verification forward...")
        with torch.no_grad():
            outputs = model(images, images, person_indices, person_indices, mode='verification')
            print(f"Verification outputs shape: {outputs.shape}")
            
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_forward()
