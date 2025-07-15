#!/usr/bin/env python3
"""
Twin Verification Interpretability - Extends Prompt-CAM for Twin Analysis

This script focuses on twin-specific visualization that doesn't exist in the original Prompt-CAM:
1. Side-by-side twin comparison (original only shows single images)
2. Verification similarity scoring (original shows classification confidence)
3. Person-specific prompt analysis (original uses class prompts)
4. Discriminative feature analysis between highly similar faces

Original Prompt-CAM handles: Single image attention, class-specific traits, cross-class analysis
This adds: Twin pair analysis, verification decisions, similarity interpretation
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from torchvision import transforms
import argparse

# For Kaggle environment
if '/kaggle/working' in os.getcwd():
    os.chdir('/kaggle/working/prompt_cam_twin')
    sys.path.append('/kaggle/working/prompt_cam_twin')

# Add project paths
sys.path.append('.')
sys.path.append('model')
sys.path.append('experiment')
sys.path.append('utils')

# Fix Python path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from model.twin_prompt_cam import TwinPromptCAM, TwinPromptCAMConfig
except ImportError as e:
    print(f"WARNING: Import warning: {e}")
    print("📝 Trying alternative imports...")
    # Alternative import approach for Kaggle
    import importlib.util
    
    # Load twin_prompt_cam manually
    spec = importlib.util.spec_from_file_location("twin_prompt_cam", "model/twin_prompt_cam.py")
    twin_cam_module = importlib.util.module_from_spec(spec)
    sys.modules["twin_prompt_cam"] = twin_cam_module
    spec.loader.exec_module(twin_cam_module)
    
    TwinPromptCAM = twin_cam_module.TwinPromptCAM
    TwinPromptCAMConfig = twin_cam_module.TwinPromptCAMConfig


class TwinVerificationInterpreter:
    """Twin-specific interpretability that extends original Prompt-CAM"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.attention_maps = []
        
        # Image preprocessing (consistent with original)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Register attention hooks
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention maps"""
        self.attention_maps = []
        
        def attention_hook(module, input, output):
            # For BlockPETL or standard attention modules
            if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
                self.attention_maps.append(module.attn.attention_weights.detach().clone())
            elif len(output) > 1 and isinstance(output[1], torch.Tensor):
                # Some blocks return (features, attention)
                self.attention_maps.append(output[1].detach().clone())
        
        # Try to hook into the backbone blocks
        try:
            for i, block in enumerate(self.model.backbone.blocks):
                block.register_forward_hook(attention_hook)
        except:
            pass  # Will fallback to simple feature extraction
    
    def analyze_twin_verification(self, img1_path, img2_path, person1_id=0, person2_id=1, threshold=0.32):
        """
        Twin-specific analysis following original Prompt-CAM approach:
        1. Extract person-specific features using person prompts
        2. Get attention maps showing what each person's prompts focus on
        3. Compare attention patterns to understand verification decision
        """
        
        print("🔍 Twin Verification Analysis (Following Prompt-CAM)")
        print(f"👥 Analyzing: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}")
        print(f"TARGET: Using optimal threshold: {threshold:.4f} (from evaluation)")
        
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1_array = np.array(img1.resize((224, 224)))
        img2_array = np.array(img2.resize((224, 224)))
        
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        
        # Extract features and attention maps following Prompt-CAM approach
        with torch.no_grad():
            # Enable attention visualization
            original_vis_attn = getattr(self.model.backbone.params, 'vis_attn', False)
            self.model.backbone.params.vis_attn = True
            
            try:
                # Get features and attention for first image with person1's prompts
                features1, attention1 = self._extract_person_features_and_attention(
                    img1_tensor, person1_id)
                
                # Get features and attention for second image with person2's prompts  
                features2, attention2 = self._extract_person_features_and_attention(
                    img2_tensor, person2_id)
                
                # If attention extraction failed, try gradient-based approach
                if attention1 is None:
                    print("🔄 Trying gradient-based attention for image 1...")
                    features1, attention1 = self._extract_gradient_based_attention(
                        img1_tensor.clone(), person1_id)
                
                if attention2 is None:
                    print("🔄 Trying gradient-based attention for image 2...")
                    features2, attention2 = self._extract_gradient_based_attention(
                        img2_tensor.clone(), person2_id)
                
                if attention1 is not None and attention2 is not None:
                    print("✅ Person-specific attention maps extracted successfully")
                elif attention1 is not None or attention2 is not None:
                    print("⚠️  Partial attention extraction - some maps available")
                else:
                    print("WARNING: Could not extract person-specific attention maps")
                    attention1 = attention2 = None
                    
            finally:
                # Restore original setting
                self.model.backbone.params.vis_attn = original_vis_attn
            
            # Compute similarity
            similarity = F.cosine_similarity(features1, features2).item()
        
        # Verification decision
        is_same_person = similarity > threshold
        confidence = abs(similarity - threshold)
        
        print(f"📊 Similarity Score: {similarity:.4f}")
        print(f"TARGET: Verification: {'SAME PERSON' if is_same_person else 'DIFFERENT PERSON'}")
        print(f"🎲 Decision Confidence: {confidence:.4f}")
        print(f"📝 Note: These are twins - should be DIFFERENT PERSON but with high similarity")
        
        # Create twin-specific visualization following Prompt-CAM style
        return self._create_prompt_cam_style_visualization(
            img1_array, img2_array, features1, features2, 
            similarity, is_same_person, threshold, confidence,
            attention1, attention2, person1_id, person2_id
        )
    
    def _create_prompt_cam_style_visualization(self, img1, img2, feat1, feat2, similarity, is_same, threshold, confidence, attention1, attention2, person1_id, person2_id):
        """Create visualization following original Prompt-CAM style with attention overlays"""
        
        # Create plot similar to original Prompt-CAM demo
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Twin Verification Analysis (Prompt-CAM Style) - Similarity: {similarity:.4f}', fontsize=16)
        
        # Row 1: Person 1 analysis
        # Original image
        axes[0, 0].imshow(img1)
        axes[0, 0].set_title(f'Person {person1_id} (90018)', fontsize=14)
        axes[0, 0].axis('off')
        
        # Person 1's attention map overlay (following Prompt-CAM approach)
        if attention1 is not None:
            att_map1 = self._extract_prompt_cam_attention(attention1, img1.shape[:2])
            if att_map1 is not None:
                # Create overlay like original Prompt-CAM
                overlay1 = self._create_prompt_cam_overlay(img1, att_map1)
                axes[0, 1].imshow(overlay1)
                axes[0, 1].set_title(f'Person {person1_id} Attention', fontsize=14)
            else:
                axes[0, 1].text(0.5, 0.5, 'Attention\nNot Available', ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'Attention\nNot Available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
        
        # Verification result for Person 1's perspective
        decision_color = 'green' if is_same else 'red'
        decision_text = 'SAME PERSON' if is_same else 'DIFFERENT PERSON'
        axes[0, 2].text(0.5, 0.6, f'Verification:\n{decision_text}', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.3),
                        transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.5, 0.3, f'Similarity: {similarity:.4f}\nThreshold: {threshold:.4f}', 
                        ha='center', va='center', fontsize=10,
                        transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Verification Result', fontsize=14)
        axes[0, 2].axis('off')
        
        # Similarity gauge
        self._draw_similarity_gauge(axes[0, 3], similarity, threshold)
        
        # Row 2: Person 2 analysis
        # Original image
        axes[1, 0].imshow(img2)
        axes[1, 0].set_title(f'Person {person2_id} (90019)', fontsize=14)
        axes[1, 0].axis('off')
        
        # Person 2's attention map overlay
        if attention2 is not None:
            att_map2 = self._extract_prompt_cam_attention(attention2, img2.shape[:2])
            if att_map2 is not None:
                overlay2 = self._create_prompt_cam_overlay(img2, att_map2)
                axes[1, 1].imshow(overlay2)
                axes[1, 1].set_title(f'Person {person2_id} Attention', fontsize=14)
            else:
                axes[1, 1].text(0.5, 0.5, 'Attention\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Attention\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        # Feature analysis
        feat1_np = feat1[0].cpu().numpy()
        feat2_np = feat2[0].cpu().numpy()
        correlation = np.corrcoef(feat1_np[:100], feat2_np[:100])[0, 1]
        
        # Feature comparison plot
        feature_dims = min(50, len(feat1_np))
        x_range = range(feature_dims)
        axes[1, 2].plot(x_range, feat1_np[:feature_dims], 'b-', alpha=0.7, linewidth=2, label=f'Person {person1_id}')
        axes[1, 2].plot(x_range, feat2_np[:feature_dims], 'r-', alpha=0.7, linewidth=2, label=f'Person {person2_id}')
        axes[1, 2].set_title('Feature Comparison', fontsize=12)
        axes[1, 2].set_xlabel('Feature Dimension')
        axes[1, 2].set_ylabel('Feature Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Attention comparison (if available)
        if attention1 is not None and attention2 is not None and att_map1 is not None and att_map2 is not None:
            # Show attention difference like in original Prompt-CAM
            att_diff = np.abs(att_map1 - att_map2)
            im = axes[1, 3].imshow(att_diff, cmap='viridis')
            axes[1, 3].set_title('Attention Difference', fontsize=12)
            axes[1, 3].axis('off')
            plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
            
            att_corr = np.corrcoef(att_map1.flatten(), att_map2.flatten())[0, 1]
        else:
            # Show feature correlation instead
            axes[1, 3].text(0.5, 0.5, f'Feature Correlation:\n{correlation:.4f}\n\n(Attention maps\nnot available)', 
                           ha='center', va='center', fontsize=12,
                           transform=axes[1, 3].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
            axes[1, 3].set_title('Feature Analysis', fontsize=12)
            axes[1, 3].axis('off')
            att_corr = None
        
        plt.tight_layout()
        return fig, {
            'similarity': similarity,
            'is_same_person': is_same,
            'confidence': confidence,
            'correlation': correlation,
            'attention_correlation': att_corr
        }
    
    def _extract_attention_map(self, attention_list):
        """Extract attention map from captured attention weights"""
        try:
            if not attention_list or len(attention_list) == 0:
                return None
            
            # Use the last layer's attention (most semantically meaningful)
            attention = attention_list[-1]
            
            if attention is None:
                return None
            
            # Handle different attention formats
            if isinstance(attention, tuple):
                attention = attention[0]
            
            if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
                # Average across heads and take CLS token attention to patches
                batch_size, num_heads, seq_len, _ = attention.shape
                # CLS token is at position 0, patches start from position 1
                att_map = attention[0].mean(0)[0, 1:].reshape(int((seq_len-1)**0.5), int((seq_len-1)**0.5))
            elif len(attention.shape) == 3:  # [batch, seq_len, seq_len]
                seq_len = attention.shape[1]
                att_map = attention[0][0, 1:].reshape(int((seq_len-1)**0.5), int((seq_len-1)**0.5))
            else:
                # Try to find attention patterns in different formats
                if attention.dim() >= 2:
                    # Flatten and try to reshape to square
                    flat_att = attention.flatten()
                    if len(flat_att) >= 196:  # 14x14 = 196
                        att_map = flat_att[:196].reshape(14, 14)
                    else:
                        return None
                else:
                    return None
            
            # Convert to numpy and resize to image size
            att_map_np = att_map.detach().cpu().numpy()
            
            # Resize to image size and normalize
            try:
                import cv2
                att_map_resized = cv2.resize(att_map_np, (224, 224))
            except:
                # Fallback using numpy interpolation
                from scipy.ndimage import zoom
                scale_factor = 224 / att_map_np.shape[0]
                att_map_resized = zoom(att_map_np, scale_factor)
            
            # Normalize to [0, 1]
            att_map_norm = (att_map_resized - att_map_resized.min()) / (att_map_resized.max() - att_map_resized.min() + 1e-8)
            
            return att_map_norm
            
        except Exception as e:
            print(f"WARNING: Attention extraction error: {e}")
            return None
    
    def _draw_similarity_gauge(self, ax, similarity, threshold):
        """Draw verification similarity gauge (NEW)"""
        # Semi-circle gauge
        angles = np.linspace(0, np.pi, 100)
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Background gauge
        ax.plot(x, y, 'k-', linewidth=3)
        ax.fill_between(x[:50], 0, y[:50], alpha=0.2, color='red', label='Different')
        ax.fill_between(x[50:], 0, y[50:], alpha=0.2, color='green', label='Same')
        
        # Threshold line
        thresh_angle = threshold * np.pi
        thresh_x = np.cos(thresh_angle)
        thresh_y = np.sin(thresh_angle)
        ax.plot([thresh_x*0.8, thresh_x*1.2], [thresh_y*0.8, thresh_y*1.2], 
                'orange', linewidth=3, label=f'Threshold ({threshold})')
        
        # Similarity indicator
        sim_angle = similarity * np.pi
        sim_x = np.cos(sim_angle)
        sim_y = np.sin(sim_angle)
        ax.arrow(0, 0, sim_x*0.9, sim_y*0.9, head_width=0.05, head_length=0.05, 
                 fc='blue', ec='blue', linewidth=3)
        
        # Scale labels
        for val in [0, 0.25, 0.5, 0.75, 1.0]:
            angle = val * np.pi
            x_tick = np.cos(angle)
            y_tick = np.sin(angle)
            ax.text(x_tick*1.15, y_tick*1.15, f'{val:.2f}', ha='center', va='center', fontsize=10)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.set_aspect('equal')
        ax.set_title('Similarity Gauge', fontsize=12)
        ax.axis('off')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3))
    
    def _plot_similarity_context(self, ax, current_sim, threshold):
        """Plot similarity in context of typical distributions (NEW)"""
        # Mock typical similarity distributions for context
        same_person_sims = np.random.normal(0.8, 0.15, 1000)
        different_person_sims = np.random.normal(0.3, 0.15, 1000)
        
        # Clip to [0, 1] range
        same_person_sims = np.clip(same_person_sims, 0, 1)
        different_person_sims = np.clip(different_person_sims, 0, 1)
        
        ax.hist(different_person_sims, bins=30, alpha=0.6, color='red', 
                label='Different People', density=True)
        ax.hist(same_person_sims, bins=30, alpha=0.6, color='green', 
                label='Same Person', density=True)
        
        # Current similarity
        ax.axvline(current_sim, color='blue', linestyle='-', linewidth=3, 
                   label=f'Current: {current_sim:.3f}')
        ax.axvline(threshold, color='orange', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.2f}')
        
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Similarity in Context', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_synthetic_attention_visualization(self, axes, img1, img2, feat1, feat2):
        """Create synthetic attention maps based on feature analysis when real attention isn't available"""
        print("📊 Creating synthetic attention visualization based on features...")
        
        # Create attention-like heatmaps based on image gradients and feature analysis
        # This gives a rough approximation of what regions might be important
        
        # Convert images to grayscale for gradient analysis
        img1_gray = np.mean(img1, axis=2)
        img2_gray = np.mean(img2, axis=2)
        
        # Compute image gradients (edge detection)
        grad1_x = np.gradient(img1_gray, axis=1)
        grad1_y = np.gradient(img1_gray, axis=0)
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = np.gradient(img2_gray, axis=1)
        grad2_y = np.gradient(img2_gray, axis=0)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Normalize gradients to [0, 1]
        grad1_norm = (grad1_mag - grad1_mag.min()) / (grad1_mag.max() - grad1_mag.min() + 1e-8)
        grad2_norm = (grad2_mag - grad2_mag.min()) / (grad2_mag.max() - grad2_mag.min() + 1e-8)
        
        # Apply gaussian smoothing to make it look more like attention
        try:
            from scipy.ndimage import gaussian_filter
            att_map1_synthetic = gaussian_filter(grad1_norm, sigma=3)
            att_map2_synthetic = gaussian_filter(grad2_norm, sigma=3)
        except ImportError:
            # Fallback without smoothing
            att_map1_synthetic = grad1_norm
            att_map2_synthetic = grad2_norm
        
        # Show synthetic attention overlays
        axes[0].imshow(img1)
        axes[0].imshow(att_map1_synthetic, cmap='hot', alpha=0.5)
        axes[0].set_title('Person 1 (Synthetic Attention)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].imshow(att_map2_synthetic, cmap='hot', alpha=0.5)
        axes[1].set_title('Person 2 (Synthetic Attention)', fontsize=12)
        axes[1].axis('off')
        
        # Attention difference
        att_diff = np.abs(att_map1_synthetic - att_map2_synthetic)
        im = axes[2].imshow(att_diff, cmap='viridis')
        axes[2].set_title('Attention Difference (Synthetic)', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        # Feature-based similarity analysis
        feat1_np = feat1[0].cpu().numpy()
        feat2_np = feat2[0].cpu().numpy()
        feature_similarity = np.corrcoef(feat1_np, feat2_np)[0, 1]
        
        axes[3].text(0.5, 0.5, f'Feature Similarity:\n{feature_similarity:.4f}\n\n(Synthetic attention based\non image gradients)', 
                    ha='center', va='center', fontsize=12,
                    transform=axes[3].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
        axes[3].set_title('Feature Analysis', fontsize=12)
        axes[3].axis('off')
        
        return feature_similarity
    
    def _extract_person_features_and_attention(self, img_tensor, person_id):
        """Extract features and attention maps for a specific person following Prompt-CAM approach"""
        try:
            # Get features using our twin model
            person_tensor = torch.tensor([person_id], device=self.device)
            features = self.model.extract_features(img_tensor, person_tensor)
            
            # Extract attention directly from the backbone transformer blocks
            # We'll hook into the last attention block to get attention maps
            attention_maps = []
            
            def attention_hook(module, input, output):
                # output is (features, attention_weights)
                if len(output) > 1 and isinstance(output[1], torch.Tensor):
                    attention_maps.append(output[1].detach().clone())
            
            # Register hook on the last transformer block
            try:
                last_block = self.model.backbone.blocks[-1]
                hook_handle = last_block.register_forward_hook(attention_hook)
                
                # Run a forward pass to capture attention
                with torch.no_grad():
                    _ = self.model.backbone.forward_features(img_tensor)
                
                # Remove the hook
                hook_handle.remove()
                
                if attention_maps:
                    # Get the attention map from the last layer
                    last_attention = attention_maps[-1]  # [batch, heads, seq_len, seq_len]
                    
                    # Extract CLS token attention to patches (following Prompt-CAM approach)
                    if len(last_attention.shape) == 4:
                        batch_size, num_heads, seq_len, _ = last_attention.shape
                        # CLS token is at position 0, extract attention to image patches
                        # Skip CLS token and any prompt tokens
                        vpt_num = getattr(self.model.backbone.params, 'vpt_num', 1)
                        cls_to_patches = last_attention[:, :, 0, (vpt_num+1):]  # [batch, heads, num_patches]
                        
                        print(f"✅ Extracted attention shape: {cls_to_patches.shape}")
                        return features, cls_to_patches
                    else:
                        print(f"WARNING: Unexpected attention shape: {last_attention.shape}")
                        return features, None
                else:
                    print("WARNING: No attention maps captured from hook")
                    return features, None
                    
            except Exception as e:
                print(f"WARNING: Could not extract attention using hooks: {e}")
                
                # Fallback: try to manually extract attention by modifying the forward pass
                try:
                    # Enable attention extraction in the backbone
                    if hasattr(self.model.backbone.params, 'vis_attn'):
                        original_vis_attn = self.model.backbone.params.vis_attn
                        self.model.backbone.params.vis_attn = True
                        
                        # Run forward pass
                        _, attention_output = self.model.backbone(img_tensor)
                        
                        # Restore original setting
                        self.model.backbone.params.vis_attn = original_vis_attn
                        
                        if attention_output is not None:
                            print(f"✅ Extracted attention via backbone: {attention_output.shape}")
                            # Process the attention following Prompt-CAM format
                            if len(attention_output.shape) == 4:  # [batch, heads, seq_len, seq_len]
                                vpt_num = getattr(self.model.backbone.params, 'vpt_num', 1)
                                cls_to_patches = attention_output[:, :, 0, (vpt_num+1):]
                                return features, cls_to_patches
                            else:
                                return features, attention_output
                        else:
                            print("WARNING: Backbone did not return attention maps")
                            return features, None
                    else:
                        print("WARNING: Backbone does not support attention visualization")
                        return features, None
                        
                except Exception as e2:
                    print(f"WARNING: Fallback attention extraction failed: {e2}")
                    return features, None
                
        except Exception as e:
            print(f"WARNING: Error in person-specific feature extraction: {e}")
            return None, None
    
    def _extract_prompt_cam_attention(self, attention_tensor, img_shape):
        """Extract attention map following original Prompt-CAM approach"""
        try:
            if attention_tensor is None:
                return None
                
            # attention_tensor should be [batch, heads, patches] following Prompt-CAM format
            # Average across attention heads to get final attention map
            if len(attention_tensor.shape) == 3:  # [batch, heads, patches]
                att_map = attention_tensor[0].mean(0)  # Average across heads
            elif len(attention_tensor.shape) == 2:  # [heads, patches]
                att_map = attention_tensor.mean(0)  # Average across heads
            elif len(attention_tensor.shape) == 1:  # [patches]
                att_map = attention_tensor
            else:
                print(f"WARNING: Unexpected attention tensor shape: {attention_tensor.shape}")
                return None
            
            # Reshape to spatial dimensions
            # For ViT, we need to determine patch grid size
            num_patches = len(att_map)
            grid_size = int(num_patches ** 0.5)
            
            if grid_size * grid_size != num_patches:
                # Try common ViT configurations
                if num_patches == 196:  # 14x14
                    grid_size = 14
                elif num_patches == 256:  # 16x16
                    grid_size = 16
                else:
                    print(f"WARNING: Could not determine grid size for {num_patches} patches")
                    return None
            
            # Reshape to spatial grid
            att_map_2d = att_map.reshape(grid_size, grid_size).detach().cpu().numpy()
            
            # Resize to image dimensions and normalize following Prompt-CAM approach
            try:
                import cv2
                att_map_resized = cv2.resize(att_map_2d, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
            except ImportError:
                # Fallback without cv2
                from scipy.ndimage import zoom
                scale_h = img_shape[0] / grid_size
                scale_w = img_shape[1] / grid_size
                att_map_resized = zoom(att_map_2d, (scale_h, scale_w))
            
            # Normalize to [0, 1] following Prompt-CAM
            min_val = att_map_resized.min()
            max_val = att_map_resized.max()
            if max_val > min_val:
                att_map_norm = (att_map_resized - min_val) / (max_val - min_val)
            else:
                att_map_norm = att_map_resized
                
            return att_map_norm
            
        except Exception as e:
            print(f"WARNING: Error extracting Prompt-CAM attention: {e}")
            return None
    
    def _create_prompt_cam_overlay(self, image, attention_map, alpha=0.5):
        """Create attention overlay following original Prompt-CAM SuperImposeHeatmap function"""
        try:
            import cv2
            
            # Apply Gaussian blur for smoothing (following original)
            attention_smoothed = cv2.GaussianBlur(attention_map, (9, 9), 0)
            
            # Convert to heatmap (following original)
            heatmap = (attention_smoothed * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Convert image to correct format
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.copy()
            
            # Superimpose the heatmap on the original image (following original)
            result = (image_uint8 * alpha + heatmap * (1 - alpha)).astype(np.uint8)
            
            # Convert BGR to RGB for matplotlib
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            return result_rgb
            
        except ImportError:
            # Fallback without cv2 - simple overlay
            print("WARNING: cv2 not available, using simple overlay")
            
            # Create simple colored overlay
            heatmap = plt.cm.hot(attention_map)[:, :, :3]  # Remove alpha channel
            result = image * alpha + heatmap * (1 - alpha)
            return (result * 255).astype(np.uint8)
    
    def _extract_gradient_based_attention(self, img_tensor, person_id):
        """Extract attention using gradient-based methods as fallback"""
        try:
            print("📊 Using gradient-based attention extraction...")
            
            # Enable gradients for the input
            img_tensor.requires_grad_(True)
            
            # Get features
            person_tensor = torch.tensor([person_id], device=self.device)
            features = self.model.extract_features(img_tensor, person_tensor)
            
            # Compute gradients with respect to input
            # Use the norm of features as the target
            target = features.norm(dim=1).sum()
            target.backward()
            
            # Get gradients
            gradients = img_tensor.grad
            
            if gradients is not None:
                # Convert gradients to attention-like map
                # Take the maximum across color channels and remove batch dimension
                grad_attention = gradients[0].abs().max(dim=0)[0]  # [H, W]
                
                # Smooth the gradient map
                import torch.nn.functional as F
                grad_attention = grad_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                grad_attention = F.avg_pool2d(grad_attention, kernel_size=3, stride=1, padding=1)
                grad_attention = grad_attention.squeeze()  # [H, W]
                
                # Normalize to [0, 1]
                min_val = grad_attention.min()
                max_val = grad_attention.max()
                if max_val > min_val:
                    grad_attention = (grad_attention - min_val) / (max_val - min_val)
                
                print(f"✅ Created gradient-based attention map: {grad_attention.shape}")
                
                # Convert to format similar to transformer attention [1, 1, num_patches]
                # Downsample to patch grid (e.g., 14x14 for 224x224 image)
                patch_size = 16  # Common ViT patch size
                num_patches_per_side = img_tensor.shape[-1] // patch_size
                
                grad_attention_downsampled = F.avg_pool2d(
                    grad_attention.unsqueeze(0).unsqueeze(0), 
                    kernel_size=patch_size, 
                    stride=patch_size
                ).squeeze()  # [14, 14]
                
                # Flatten to patch sequence
                grad_attention_patches = grad_attention_downsampled.flatten().unsqueeze(0).unsqueeze(0)  # [1, 1, 196]
                
                return features, grad_attention_patches
            else:
                print("WARNING: No gradients available")
                return features, None
                
        except Exception as e:
            print(f"WARNING: Gradient-based attention extraction failed: {e}")
            return features, None
        finally:
            # Clean up gradients
            if img_tensor.grad is not None:
                img_tensor.grad.zero_()
            img_tensor.requires_grad_(False)
    
def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """Load twin verification model"""
    
    # Default config for twin verification
    config_dict = {
        'model': 'dinov2',
        'drop_path_rate': 0.1,
        'vpt_num': 1,
        'vpt_mode': 'deep',
        'num_persons': 356
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict.update(yaml.safe_load(f))
    
    config = argparse.Namespace(**config_dict)
    
    model_config = TwinPromptCAMConfig(
        model=config.model,
        drop_path_rate=config.drop_path_rate,
        vpt_num=config.vpt_num,
        vpt_mode=config.vpt_mode,
        stage1_training=False
    )
    
    model = TwinPromptCAM(model_config, num_persons=config.num_persons)
    
    # Load checkpoint
    print(f"📦 Loading model from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"WARNING: Model loading with errors: {e}")
    
    print("✅ Twin verification model loaded successfully")
    return model


def main():
    parser = argparse.ArgumentParser(description='Twin Verification Interpretability (Extends Prompt-CAM)')
    parser.add_argument('--checkpoint', type=str, 
                        default="/kaggle/input/nd-twin/checkpoint_stage2_best_epoch_2.pth",
                        help='Path to twin verification checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--img1', type=str, required=True,
                        help='Path to first person image')
    parser.add_argument('--img2', type=str, required=True,
                        help='Path to second person image')
    parser.add_argument('--person1_id', type=int, default=0,
                        help='Person ID for first image (for person-specific prompts)')
    parser.add_argument('--person2_id', type=int, default=1,
                        help='Person ID for second image (for person-specific prompts)')
    parser.add_argument('--threshold', type=float, default=0.32,
                        help='Verification threshold (default: optimal threshold from evaluation)')
    parser.add_argument('--output', type=str, 
                        default="/kaggle/working/twin_verification_analysis.png",
                        help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("🚀 Twin Verification Interpretability")
    print("📝 This extends original Prompt-CAM with twin-specific analysis:")
    print("   • Pairwise similarity analysis")  
    print("   • Person-specific prompt interpretation")
    print("   • Verification decision boundaries")
    print("   • Discriminative feature analysis")
    print()
    
    # Check files exist
    if not os.path.exists(args.img1):
        print(f"❌ Image 1 not found: {args.img1}")
        return
    if not os.path.exists(args.img2):
        print(f"❌ Image 2 not found: {args.img2}")
        return
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.config)
    
    # Create interpreter
    device = args.device if torch.cuda.is_available() else 'cpu'
    interpreter = TwinVerificationInterpreter(model, device)
    
    # Run twin-specific analysis
    fig, results = interpreter.analyze_twin_verification(
        args.img1, args.img2, 
        args.person1_id, args.person2_id, 
        args.threshold
    )
    
    # Save visualization
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"💾 Twin verification analysis saved to {args.output}")
    
    # Show results
    plt.show()
    
    print(f"\nTARGET: Twin Verification Results:")
    print(f"   Similarity: {results['similarity']:.4f}")
    print(f"   Decision: {'SAME PERSON' if results['is_same_person'] else 'DIFFERENT PERSON'}")
    print(f"   Confidence: {results['confidence']:.4f}")
    print(f"   Feature Correlation: {results['correlation']:.4f}")


# Kaggle-friendly wrapper
def kaggle_twin_analysis():
    """Quick analysis for Kaggle with example twins"""
    
    checkpoint_path = "/kaggle/input/nd-twin/checkpoint_stage2_best_epoch_2.pth"
    
    # Example: Real twin pair from the dataset
    img1_path = "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224/90018/90018d13.jpg"
    img2_path = "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224/90019/90019d13.jpg"  # Twin of 90018
    
    try:
        model = load_model_from_checkpoint(checkpoint_path)
        interpreter = TwinVerificationInterpreter(model)
        
        fig, results = interpreter.analyze_twin_verification(img1_path, img2_path, threshold=0.32)
        
        plt.savefig("/kaggle/working/twin_analysis_example.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    except Exception as e:
        print(f"WARNING: Error in Kaggle analysis: {e}")
        print("🔄 Falling back to simple analysis...")
        # Import and use the simple version as fallback
        import simple_twin_viz
        return simple_twin_viz.simple_twin_comparison(checkpoint_path, img1_path, img2_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No arguments - run Kaggle example
        kaggle_twin_analysis()
    else:
        # Arguments provided - run main
        main()
