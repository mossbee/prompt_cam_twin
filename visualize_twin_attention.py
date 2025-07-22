#!/usr/bin/env python3
"""
Twin Face Verification Attention Visualization

This script allows you to:
1. Input two images of twins
2. Visualize attention maps and feature differences
3. See what the model focuses on to distinguish between twins
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse
import yaml
from torchvision import transforms
import cv2

# Add project paths
sys.path.append('.')
sys.path.append('model')
sys.path.append('experiment')

from model.twin_prompt_cam import TwinPromptCAM, TwinPromptCAMConfig
from experiment.build_model import get_model


class TwinAttentionVisualizer:
    """Visualize attention maps and feature differences for twin face verification"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Hook for capturing attention maps
        self.attention_maps = {}
        self.feature_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features and attention"""
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn_weights'):
                    # For ViT attention blocks
                    self.attention_maps[name] = module.attn_weights.detach()
                elif len(output) > 1 and output[1] is not None:
                    # For attention outputs with weights
                    self.attention_maps[name] = output[1].detach()
            return hook
        
        def feature_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.feature_maps[name] = output[0].detach()
                else:
                    self.feature_maps[name] = output.detach()
            return hook
        
        # Register hooks for backbone layers
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            
            # For ViT-based models
            if hasattr(backbone, 'blocks'):
                for i, block in enumerate(backbone.blocks):
                    if hasattr(block, 'attn'):
                        block.attn.register_forward_hook(attention_hook(f'attn_block_{i}'))
                    block.register_forward_hook(feature_hook(f'feature_block_{i}'))
            
            # For final features
            backbone.register_forward_hook(feature_hook('backbone_final'))
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        tensor_image = self.transform(image).unsqueeze(0).to(self.device)
        return tensor_image, original_image
    
    def extract_features_and_attention(self, img1_path, img2_path, person1_id=0, person2_id=1):
        """Extract features and attention maps for two images"""
        
        # Clear previous captures
        self.attention_maps.clear()
        self.feature_maps.clear()
        
        # Load and preprocess images
        img1_tensor, img1_original = self.preprocess_image(img1_path)
        img2_tensor, img2_original = self.preprocess_image(img2_path)
        
        with torch.no_grad():
            # Forward pass for first image
            self.attention_maps.clear()
            self.feature_maps.clear()
            person1_tensor = torch.tensor([person1_id], device=self.device)
            features1 = self.model.extract_features(img1_tensor, person1_tensor)
            attention1 = dict(self.attention_maps)
            features_map1 = dict(self.feature_maps)
            
            # Forward pass for second image
            self.attention_maps.clear()
            self.feature_maps.clear()
            person2_tensor = torch.tensor([person2_id], device=self.device)
            features2 = self.model.extract_features(img2_tensor, person2_tensor)
            attention2 = dict(self.attention_maps)
            features_map2 = dict(self.feature_maps)
            
            # Compute similarity
            similarity = F.cosine_similarity(features1, features2).item()
        
        return {
            'img1_original': img1_original,
            'img2_original': img2_original,
            'features1': features1,
            'features2': features2,
            'attention1': attention1,
            'attention2': attention2,
            'features_map1': features_map1,
            'features_map2': features_map2,
            'similarity': similarity
        }
    
    def visualize_attention_map(self, attention_weights, original_image, layer_name=""):
        """Visualize attention map overlaid on original image"""
        
        if attention_weights is None or len(attention_weights.shape) < 3:
            return None
        
        # Average attention across heads and take CLS token attention
        if len(attention_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
            attention = attention_weights[0].mean(0)  # Average across heads
        else:
            attention = attention_weights[0]
        
        # Get attention from CLS token to all patches
        cls_attention = attention[0, 1:]  # Skip CLS token itself
        
        # Reshape to spatial dimensions (14x14 for 224x224 input with patch size 16)
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        
        # Resize to match original image
        attention_map = cv2.resize(attention_map.cpu().numpy(), (224, 224))
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image')
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention_map, cmap='hot', alpha=0.8)
        axes[1].set_title(f'Attention Map - {layer_name}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.6)
        axes[2].set_title(f'Attention Overlay')
        axes[2].axis('off')
        
        return fig
    
    def visualize_feature_difference(self, features1, features2, original_img1, original_img2):
        """Visualize feature differences between two images"""
        
        # Compute feature difference
        feature_diff = torch.abs(features1 - features2)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original images
        axes[0, 0].imshow(original_img1)
        axes[0, 0].set_title('Twin 1')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(original_img2)
        axes[0, 1].set_title('Twin 2')
        axes[0, 1].axis('off')
        
        # Feature similarity heatmap
        similarity_matrix = F.cosine_similarity(
            features1.unsqueeze(2), 
            features2.unsqueeze(1), 
            dim=0
        ).cpu().numpy()
        
        im = axes[0, 2].imshow(similarity_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
        axes[0, 2].set_title('Feature Similarity Matrix')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Feature magnitudes
        feat1_norm = torch.norm(features1, dim=0).cpu().numpy()
        feat2_norm = torch.norm(features2, dim=0).cpu().numpy()
        
        axes[1, 0].bar(range(len(feat1_norm)), feat1_norm, alpha=0.7, label='Twin 1')
        axes[1, 0].set_title('Feature Magnitudes - Twin 1')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Magnitude')
        
        axes[1, 1].bar(range(len(feat2_norm)), feat2_norm, alpha=0.7, label='Twin 2', color='orange')
        axes[1, 1].set_title('Feature Magnitudes - Twin 2')
        axes[1, 1].set_xlabel('Feature Dimension')
        axes[1, 1].set_ylabel('Magnitude')
        
        # Feature difference
        diff_norm = torch.norm(feature_diff, dim=0).cpu().numpy()
        axes[1, 2].bar(range(len(diff_norm)), diff_norm, alpha=0.7, color='red')
        axes[1, 2].set_title('Feature Differences')
        axes[1, 2].set_xlabel('Feature Dimension')
        axes[1, 2].set_ylabel('Difference Magnitude')
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_visualization(self, img1_path, img2_path, person1_id=0, person2_id=1, save_dir=None):
        """Create comprehensive visualization of twin verification"""
        
        print(f"🔍 Analyzing twin images...")
        print(f"   Image 1: {img1_path}")
        print(f"   Image 2: {img2_path}")
        
        # Extract features and attention
        results = self.extract_features_and_attention(img1_path, img2_path, person1_id, person2_id)
        
        print(f"   Similarity Score: {results['similarity']:.4f}")
        
        # Create main comparison figure
        fig_main, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig_main.suptitle(f'Twin Face Verification Analysis (Similarity: {results["similarity"]:.4f})', fontsize=16)
        
        # Original images
        axes[0, 0].imshow(results['img1_original'])
        axes[0, 0].set_title('Twin 1')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['img2_original'])
        axes[0, 1].set_title('Twin 2')
        axes[0, 1].axis('off')
        
        # Attention maps for last layer
        last_layer_key = list(results['attention1'].keys())[-1] if results['attention1'] else None
        
        if last_layer_key and last_layer_key in results['attention1']:
            attention1 = results['attention1'][last_layer_key]
            attention2 = results['attention2'][last_layer_key]
            
            # Process attention maps
            if len(attention1.shape) == 4:
                # [batch, heads, seq_len, seq_len]
                attn = attention1[0].mean(0)[0, 1:]
                grid_size = int((attn.shape[0]) ** 0.5)
                att_map1 = attn.reshape(grid_size, grid_size)
                attn2 = attention2[0].mean(0)[0, 1:]
                att_map2 = attn2.reshape(grid_size, grid_size)
            else:
                attn = attention1[0][0, 1:]
                grid_size = int((attn.shape[0]) ** 0.5)
                att_map1 = attn.reshape(grid_size, grid_size)
                attn2 = attention2[0][0, 1:]
                att_map2 = attn2.reshape(grid_size, grid_size)
            
            # Resize and normalize
            att_map1 = cv2.resize(att_map1.cpu().numpy(), (224, 224))
            att_map2 = cv2.resize(att_map2.cpu().numpy(), (224, 224))
            
            att_map1 = (att_map1 - att_map1.min()) / (att_map1.max() - att_map1.min())
            att_map2 = (att_map2 - att_map2.min()) / (att_map2.max() - att_map2.min())
            
            # Attention overlays
            axes[0, 2].imshow(results['img1_original'])
            axes[0, 2].imshow(att_map1, cmap='hot', alpha=0.6)
            axes[0, 2].set_title('Twin 1 Attention')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(results['img2_original'])
            axes[0, 3].imshow(att_map2, cmap='hot', alpha=0.6)
            axes[0, 3].set_title('Twin 2 Attention')
            axes[0, 3].axis('off')
            
            # Attention difference
            att_diff = np.abs(att_map1 - att_map2)
            im = axes[1, 0].imshow(att_diff, cmap='viridis')
            axes[1, 0].set_title('Attention Difference')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Feature analysis
        features1 = results['features1'][0].cpu().numpy()
        features2 = results['features2'][0].cpu().numpy()
        feature_diff = np.abs(features1 - features2)
        
        # Feature comparison
        axes[1, 1].plot(features1, alpha=0.7, label='Twin 1', linewidth=2)
        axes[1, 1].plot(features2, alpha=0.7, label='Twin 2', linewidth=2)
        axes[1, 1].set_title('Feature Comparison')
        axes[1, 1].set_xlabel('Feature Dimension')
        axes[1, 1].set_ylabel('Feature Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Feature difference
        axes[1, 2].bar(range(len(feature_diff)), feature_diff, alpha=0.7, color='red')
        axes[1, 2].set_title('Feature Differences')
        axes[1, 2].set_xlabel('Feature Dimension')
        axes[1, 2].set_ylabel('Absolute Difference')
        
        # Similarity distribution (mock data for context)
        # In practice, you'd compute this from a validation set
        sim_scores = np.random.normal(results['similarity'], 0.1, 1000)
        axes[1, 3].hist(sim_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 3].axvline(results['similarity'], color='red', linestyle='--', linewidth=2, label=f'Current: {results["similarity"]:.3f}')
        axes[1, 3].set_title('Similarity Distribution')
        axes[1, 3].set_xlabel('Similarity Score')
        axes[1, 3].set_ylabel('Frequency')
        axes[1, 3].legend()
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig_main.savefig(os.path.join(save_dir, 'twin_verification_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_dir}/twin_verification_analysis.png")
        
        return fig_main, results


def load_model_from_checkpoint(checkpoint_path, config_path):
    """Load model from checkpoint"""
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = argparse.Namespace(**config_dict)
    config.stage1_training = False  # For evaluation
    
    # Create model config
    model_config = TwinPromptCAMConfig(
        model=config.model,
        drop_path_rate=getattr(config, 'drop_path_rate', 0.1),
        vpt_num=getattr(config, 'vpt_num', 1),
        vpt_mode=getattr(config, 'vpt_mode', 'deep'),
        stage1_training=False
    )
    
    model = TwinPromptCAM(model_config, num_persons=getattr(config, 'num_persons', 356))
    
    # Load checkpoint
    print(f"📦 Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"⚠️  Strict loading failed, trying non-strict...")
        model.load_state_dict(state_dict, strict=False)
    
    print("✅ Model loaded successfully")
    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize Twin Face Verification Attention')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--img1', type=str, required=True,
                        help='Path to first twin image')
    parser.add_argument('--img2', type=str, required=True,
                        help='Path to second twin image')
    parser.add_argument('--person1_id', type=int, default=0,
                        help='Person ID for first image (for prompts)')
    parser.add_argument('--person2_id', type=int, default=1,
                        help='Person ID for second image (for prompts)')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("🚀 Twin Face Verification Attention Visualization")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"⚙️  Config: {args.config}")
    print(f"🖼️  Image 1: {args.img1}")
    print(f"🖼️  Image 2: {args.img2}")
    print(f"💾 Output: {args.output_dir}")
    
    # Check if images exist
    if not os.path.exists(args.img1):
        print(f"❌ Image 1 not found: {args.img1}")
        return
    if not os.path.exists(args.img2):
        print(f"❌ Image 2 not found: {args.img2}")
        return
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.config)
    
    # Create visualizer
    device = args.device if torch.cuda.is_available() else 'cpu'
    visualizer = TwinAttentionVisualizer(model, device)
    
    # Create visualization
    fig, results = visualizer.create_comprehensive_visualization(
        args.img1, args.img2, 
        args.person1_id, args.person2_id,
        args.output_dir
    )
    
    # Show interactive plot
    plt.show()
    
    print(f"✅ Visualization complete!")
    print(f"📊 Similarity Score: {results['similarity']:.4f}")
    if results['similarity'] > 0.5:
        print("🤝 Model predicts: SAME PERSON (twins verified)")
    else:
        print("❌ Model predicts: DIFFERENT PERSON (twins not verified)")


if __name__ == "__main__":
    main()
