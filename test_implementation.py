#!/usr/bin/env python3
"""
Test script for Twin Face Verification implementation.
This script verifies that all components are properly integrated.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_loading():
    """Test that twin datasets can be loaded properly"""
    print("Testing data loading...")
    
    try:
        from data.twin_dataset import TwinPairDataset, TwinIdentityDataset
        
        # Check if data files exist
        data_files = [
            'data/train_dataset_infor.json',
            'data/train_twin_pairs.json', 
            'data/test_dataset_infor.json',
            'data/test_twin_pairs.json'
        ]
        
        missing_files = [f for f in data_files if not os.path.exists(f)]
        if missing_files:
            print(f"⚠️  Missing data files: {missing_files}")
            print("   Please ensure dataset files are in the data/ directory")
            return False
        
        # Test identity dataset
        identity_dataset = TwinIdentityDataset('data/train_dataset_infor.json')
        print(f"✅ Identity dataset loaded: {len(identity_dataset)} samples")
        print(f"   Number of classes: {identity_dataset.num_classes}")
        
        # Test pair dataset
        pair_dataset = TwinPairDataset(
            'data/train_dataset_infor.json',
            'data/train_twin_pairs.json'
        )
        print(f"✅ Pair dataset loaded: {len(pair_dataset)} pairs")
        
        # Test data loading
        sample = identity_dataset[0]
        print(f"✅ Identity sample shape: {sample['image'].shape}")
        print(f"   Images are loaded directly (224x224 already preprocessed)")
        
        pair_sample = pair_dataset[0]
        print(f"✅ Pair sample shapes: img1={pair_sample['img1'].shape}, img2={pair_sample['img2'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False


def test_model_creation():
    """Test that twin verification model can be created"""
    print("\nTesting model creation...")
    
    try:
        from model.twin_prompt_cam import TwinPromptCAM, TwinPromptCAMConfig
        
        # Create model config
        config = TwinPromptCAMConfig(
            model='dinov2',
            drop_path_rate=0.1,
            stage1_training=True
        )
        
        # Create model
        model = TwinPromptCAM(config, num_persons=356)
        print(f"✅ Model created successfully")
        
        # Test model forward pass
        batch_size = 2
        img1 = torch.randn(batch_size, 3, 224, 224)
        img2 = torch.randn(batch_size, 3, 224, 224)
        person1_idx = torch.randint(0, 356, (batch_size,))
        person2_idx = torch.randint(0, 356, (batch_size,))
        
        # Test identity mode
        with torch.no_grad():
            identity_output = model(img1, img2, person1_idx, person2_idx, mode='identity_classification')
            print(f"✅ Identity output shape: {identity_output.shape}")
        
        # Test verification mode  
        config.stage1_training = False
        model = TwinPromptCAM(config, num_persons=356)
        with torch.no_grad():
            verification_output = model(img1, img2, person1_idx, person2_idx, mode='verification')
            print(f"✅ Verification output shape: {verification_output.shape}")
        
        # Test trainable parameters
        trainable_params = model.get_trainable_parameters()
        total_params = sum(p.numel() for p in trainable_params)
        print(f"✅ Trainable parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test verification metrics"""
    print("\nTesting verification metrics...")
    
    try:
        from utils.verification_metrics import VerificationMetrics, TwinVerificationAnalyzer
        
        # Create dummy data
        scores = torch.randn(100)  # Random scores
        labels = torch.randint(0, 2, (100,))  # Random binary labels
        
        # Test verification metrics
        metrics = VerificationMetrics()
        metrics.update(scores, labels)
        
        results = metrics.compute_all_metrics()
        print(f"✅ Computed metrics: AUC={results['auc']:.3f}, EER={results['eer']:.3f}")
        
        # Test twin analyzer
        analyzer = TwinVerificationAnalyzer()
        is_twin_pairs = [True] * 30 + [False] * 70  # 30% twin pairs
        analyzer.update(scores, labels, is_twin_pairs)
        
        twin_results = analyzer.compute_twin_vs_nontwin_performance()
        print(f"✅ Twin analysis completed: {twin_results['twin_pairs']['count']} twin pairs")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {str(e)}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from utils.misc import load_yaml
        
        config_files = [
            'experiment/config/twin_verification/dinov2/args.yaml',
            'experiment/config/twin_verification/dino/args.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                config = load_yaml(config_file)
                print(f"✅ Loaded config: {config_file}")
                print(f"   Model: {config.get('model', 'N/A')}")
                print(f"   Data: {config.get('data', 'N/A')}")
            else:
                print(f"⚠️  Config file not found: {config_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {str(e)}")
        return False


def run_integration_test():
    """Run a minimal integration test"""
    print("\nRunning integration test...")
    
    try:
        # Test argument parsing
        from main import setup_parser
        parser = setup_parser()
        
        # Test with twin verification arguments
        test_args = [
            '--data', 'twin',
            '--train_type', 'twin_verification',
            '--model', 'dinov2',
            '--stage1_training',
            '--batch_size', '4',
            '--epoch', '2',
            '--lr', '0.001'
        ]
        
        args = parser.parse_args(test_args)
        print(f"✅ Arguments parsed successfully")
        print(f"   Data: {args.data}")
        print(f"   Train type: {args.train_type}")
        print(f"   Stage 1 training: {args.stage1_training}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False


def main():
    print("🚀 Twin Face Verification Implementation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        test_data_loading,
        test_model_creation, 
        test_metrics,
        test_config_loading,
        run_integration_test
    ]
    
    for test_func in tests:
        passed = test_func()
        all_tests_passed = all_tests_passed and passed
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Ensure dataset files are in data/ directory")
        print("2. Run training with: python main.py --config experiment/config/twin_verification/dinov2/args.yaml")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
