#!/usr/bin/env python3
"""
Simple Twin Visualization for Kaggle
Minimal dependencies, focused on twin verification analysis
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# For Kaggle environment
if '/kaggle/working' in os.getcwd():
    os.chdir('/kaggle/working/prompt_cam_twin')

# Simple approach - add all paths
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'model'))
sys.path.insert(0, os.path.join(current_dir, 'experiment'))
sys.path.insert(0, os.path.join(current_dir, 'utils'))

print("🔧 Setting up environment for Kaggle...")
print(f"📁 Current directory: {current_dir}")
print(f"🐍 Python path: {sys.path[:3]}...")


def simple_twin_comparison(checkpoint_path, img1_path, img2_path, save_path=None):
    """Simple twin comparison without complex imports"""
    
    print("🚀 Simple Twin Face Verification")
    print(f"📁 Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"🖼️  Image 1: {os.path.basename(img1_path)}")
    print(f"🖼️  Image 2: {os.path.basename(img2_path)}")
    
    # Simple feature extraction using basic torch operations
    # This is a simplified version that doesn't need the full model
    try:
        # Try to load the checkpoint to extract basic info
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully")
        
        # Check what's in the checkpoint
        if 'model_state_dict' in checkpoint:
            print("📦 Found model state dict")
        elif 'model' in checkpoint:
            print("📦 Found model weights")
        else:
            print("📦 Checkpoint format detected")
            
    except Exception as e:
        print(f"⚠️  Checkpoint loading error: {e}")
        return None
    
    # Load and process images
    print("🖼️  Loading images...")
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Resize to standard size
        img1 = img1.resize((224, 224))
        img2 = img2.resize((224, 224))
        
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        print("✅ Images loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return None
    
    # Simple pixel-based similarity as fallback
    # This gives a basic comparison while we work on model loading
    print("🔍 Computing basic similarity...")
    
    # Normalize images
    img1_norm = img1_array.astype(np.float32) / 255.0
    img2_norm = img2_array.astype(np.float32) / 255.0
    
    # Compute basic similarities
    pixel_similarity = 1.0 - np.mean(np.abs(img1_norm - img2_norm))
    
    # RGB histogram similarity
    hist1_r = np.histogram(img1_array[:,:,0], bins=32, range=(0,256))[0]
    hist1_g = np.histogram(img1_array[:,:,1], bins=32, range=(0,256))[0]
    hist1_b = np.histogram(img1_array[:,:,2], bins=32, range=(0,256))[0]
    
    hist2_r = np.histogram(img2_array[:,:,0], bins=32, range=(0,256))[0]
    hist2_g = np.histogram(img2_array[:,:,1], bins=32, range=(0,256))[0]
    hist2_b = np.histogram(img2_array[:,:,2], bins=32, range=(0,256))[0]
    
    # Cosine similarity of histograms
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    hist_sim_r = cosine_sim(hist1_r, hist2_r)
    hist_sim_g = cosine_sim(hist1_g, hist2_g)
    hist_sim_b = cosine_sim(hist1_b, hist2_b)
    hist_similarity = (hist_sim_r + hist_sim_g + hist_sim_b) / 3.0
    
    # Combined similarity (this is a placeholder for actual model features)
    combined_similarity = (pixel_similarity * 0.3 + hist_similarity * 0.7)
    
    print(f"📊 Pixel Similarity: {pixel_similarity:.4f}")
    print(f"📊 Histogram Similarity: {hist_similarity:.4f}")
    print(f"📊 Combined Similarity: {combined_similarity:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Twin Face Analysis\nCombined Similarity: {combined_similarity:.4f}', fontsize=16)
    
    # Original images
    axes[0, 0].imshow(img1_array)
    axes[0, 0].set_title('Person 1', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_array)
    axes[0, 1].set_title('Person 2', fontsize=14)
    axes[0, 1].axis('off')
    
    # Prediction
    threshold = 0.5
    if combined_similarity > threshold:
        prediction = "LIKELY SAME PERSON ✅"
        color = 'green'
    else:
        prediction = "LIKELY DIFFERENT PERSON ❌"
        color = 'red'
    
    axes[0, 2].text(0.5, 0.6, f'Analysis:\n{prediction}', 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                    transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.5, 0.3, f'Similarity: {combined_similarity:.4f}', 
                    ha='center', va='center', fontsize=12,
                    transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.5, 0.1, f'(Basic analysis)', 
                    ha='center', va='center', fontsize=10, style='italic',
                    transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('Basic Analysis', fontsize=14)
    axes[0, 2].axis('off')
    
    # Difference map
    diff_img = np.abs(img1_norm - img2_norm)
    axes[1, 0].imshow(diff_img)
    axes[1, 0].set_title('Pixel Differences', fontsize=12)
    axes[1, 0].axis('off')
    
    # Histogram comparison
    x_bins = range(32)
    axes[1, 1].bar(x_bins, hist1_r, alpha=0.7, color='red', label='Person 1 (R)', width=0.8)
    axes[1, 1].bar(x_bins, hist2_r, alpha=0.7, color='blue', label='Person 2 (R)', width=0.6)
    axes[1, 1].set_title('Red Channel Histograms', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Intensity Bin')
    axes[1, 1].set_ylabel('Frequency')
    
    # Similarity gauge
    angles = np.linspace(0, np.pi, 100)
    x = np.cos(angles)
    y = np.sin(angles)
    
    axes[1, 2].plot(x, y, 'k-', linewidth=3)
    axes[1, 2].fill_between(x, 0, y, alpha=0.2, color='lightblue')
    
    # Add similarity indicator
    sim_angle = combined_similarity * np.pi
    sim_x = np.cos(sim_angle)
    sim_y = np.sin(sim_angle)
    axes[1, 2].arrow(0, 0, sim_x*0.9, sim_y*0.9, head_width=0.05, head_length=0.05, 
                     fc='red', ec='red', linewidth=3)
    
    # Add scale
    for val in [0, 0.25, 0.5, 0.75, 1.0]:
        angle = val * np.pi
        x_tick = np.cos(angle)
        y_tick = np.sin(angle)
        axes[1, 2].plot([x_tick*0.9, x_tick*1.1], [y_tick*0.9, y_tick*1.1], 'k-', linewidth=2)
        axes[1, 2].text(x_tick*1.2, y_tick*1.2, f'{val:.1f}', ha='center', va='center')
    
    axes[1, 2].set_xlim(-1.3, 1.3)
    axes[1, 2].set_ylim(-0.1, 1.3)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_title('Similarity Gauge', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to {save_path}")
    
    # Show plot
    plt.show()
    
    return {
        'pixel_similarity': pixel_similarity,
        'histogram_similarity': hist_similarity,
        'combined_similarity': combined_similarity,
        'prediction': prediction
    }


def find_twin_pairs_from_json():
    """Find actual twin pairs from the dataset"""
    try:
        with open('data/test_twin_pairs.json', 'r') as f:
            twin_pairs = json.load(f)
        
        with open('data/test_dataset_infor.json', 'r') as f:
            dataset_info = json.load(f)
        
        print("📋 Available twin pairs:")
        for i, pair in enumerate(twin_pairs[:5]):  # Show first 5 pairs
            person1, person2 = pair
            if person1 in dataset_info and person2 in dataset_info:
                img1_count = len(dataset_info[person1])
                img2_count = len(dataset_info[person2])
                print(f"   {i+1}. {person1} ({img1_count} images) ↔ {person2} ({img2_count} images)")
                
                # Show example image paths
                if dataset_info[person1] and dataset_info[person2]:
                    print(f"      Example: {dataset_info[person1][0]}")
                    print(f"               {dataset_info[person2][0]}")
        
        return twin_pairs, dataset_info
        
    except Exception as e:
        print(f"⚠️  Could not load twin pairs: {e}")
        return None, None


def main():
    print("🚀 Simple Twin Face Analysis for Kaggle")
    print("📝 This is a simplified version while we resolve import issues")
    print()
    
    # Default paths for Kaggle
    checkpoint_path = "/kaggle/input/nd-twin/checkpoint_stage2_best_epoch_2.pth"
    
    # Try to find twin pairs
    twin_pairs, dataset_info = find_twin_pairs_from_json()
    
    # Use provided images or defaults
    img1_path = "/kaggle/input/nd-twin/90018d13_cutout.jpg"
    img2_path = "/kaggle/input/nd-twin/90019d13_cutout.jpg"
    
    # Check if files exist
    if not os.path.exists(img1_path):
        print(f"⚠️  Image 1 not found: {img1_path}")
        # Try alternative paths
        alt_img1 = "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224/90018/90018d13.jpg"
        if os.path.exists(alt_img1):
            img1_path = alt_img1
            print(f"✅ Using alternative: {alt_img1}")
    
    if not os.path.exists(img2_path):
        print(f"⚠️  Image 2 not found: {img2_path}")
        # Try alternative paths
        alt_img2 = "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224/90019/90019d13.jpg"
        if os.path.exists(alt_img2):
            img2_path = alt_img2
            print(f"✅ Using alternative: {alt_img2}")
    
    # Run analysis
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        results = simple_twin_comparison(
            checkpoint_path, img1_path, img2_path, 
            "/kaggle/working/simple_twin_analysis.png"
        )
        
        if results:
            print(f"\n🎯 Analysis Results:")
            print(f"   Pixel Similarity: {results['pixel_similarity']:.4f}")
            print(f"   Histogram Similarity: {results['histogram_similarity']:.4f}")
            print(f"   Combined Similarity: {results['combined_similarity']:.4f}")
            print(f"   Prediction: {results['prediction']}")
            print(f"\n📝 Note: This is a basic analysis. For full model-based analysis,")
            print(f"    we need to resolve the import issues first.")
    else:
        print(f"❌ Could not find required images")
        print(f"   Looking for: {img1_path}")
        print(f"   Looking for: {img2_path}")


if __name__ == "__main__":
    main()
