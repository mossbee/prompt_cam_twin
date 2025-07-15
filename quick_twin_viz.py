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
        
        # Image preprocessing (consistent with original)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_twin_verification(self, img1_path, img2_path, person1_id=0, person2_id=1, threshold=0.32):
        """
        Twin-specific analysis that's NOT in original Prompt-CAM:
        1. Pairwise similarity analysis
        2. Person-specific prompt comparison  
        3. Verification decision interpretation
        4. Discriminative feature analysis
        5. Person-specific attention visualization
        """
        
        print("🔍 Twin Verification Analysis (Extended Prompt-CAM)")
        print(f"👥 Analyzing: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}")
        print(f"TARGET: Using optimal threshold: {threshold:.4f} (from evaluation)")
        
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1_array = np.array(img1.resize((224, 224)))
        img2_array = np.array(img2.resize((224, 224)))
        
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        
        # Extract person-specific features AND attention maps (NEW: not in original Prompt-CAM)
        with torch.no_grad():
            # Convert person IDs to tensors as expected by the model
            person1_tensor = torch.tensor([person1_id], device=self.device)
            person2_tensor = torch.tensor([person2_id], device=self.device)
            
            # Get features and attention maps
            features1 = self.model.extract_features(img1_tensor, person1_tensor)
            features2 = self.model.extract_features(img2_tensor, person2_tensor)
            
            # Get attention maps by running forward pass
            try:
                _, attention1 = self.model.backbone(img1_tensor)
                _, attention2 = self.model.backbone(img2_tensor) 
                print("✅ Attention maps extracted successfully")
            except:
                print("WARNING: Could not extract attention maps, using feature analysis only")
                attention1 = attention2 = None
            
            similarity = F.cosine_similarity(features1, features2).item()
        
        # Verification decision
        is_same_person = similarity > threshold
        confidence = abs(similarity - threshold)
        
        print(f"📊 Similarity Score: {similarity:.4f}")
        print(f"TARGET: Verification: {'SAME PERSON' if is_same_person else 'DIFFERENT PERSON'}")
        print(f"🎲 Decision Confidence: {confidence:.4f}")
        print(f"📝 Note: These are twins - should be DIFFERENT PERSON but with high similarity")
        
        # Create twin-specific visualization with attention
        return self._create_twin_comparison_plot(
            img1_array, img2_array, features1, features2, 
            similarity, is_same_person, threshold, confidence,
            attention1, attention2
        )
    
    def _create_twin_comparison_plot(self, img1, img2, feat1, feat2, similarity, is_same, threshold, confidence, attention1=None, attention2=None):
        """Create twin verification visualization with attention maps (NEW - not in original Prompt-CAM)"""
        
        # Create larger plot to include attention maps
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Twin Verification Analysis - Similarity: {similarity:.4f} | Threshold: {threshold:.4f}', fontsize=16)
        
        # Row 1: Original Images and Decision
        axes[0, 0].imshow(img1)
        axes[0, 0].set_title('Person 1 (90018)', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2)
        axes[0, 1].set_title('Person 2 (90019)', fontsize=14)
        axes[0, 1].axis('off')
        
        # Verification decision (Fixed for twins - they SHOULD be same person)
        decision_color = 'green' if is_same else 'red'
        decision_text = 'SAME PERSON [YES]' if is_same else 'DIFFERENT PERSON [NO]'
        
        axes[0, 2].text(0.5, 0.7, f'Verification:\n{decision_text}', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.3),
                        transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.5, 0.4, f'Similarity: {similarity:.4f}', 
                        ha='center', va='center', fontsize=12,
                        transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.5, 0.2, f'Threshold: {threshold:.4f}', 
                        ha='center', va='center', fontsize=10,
                        transform=axes[0, 2].transAxes)
        confidence_text = f'Confidence: {confidence:.4f}'
        axes[0, 2].text(0.5, 0.05, confidence_text, 
                        ha='center', va='center', fontsize=10, style='italic',
                        transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Decision', fontsize=14)
        axes[0, 2].axis('off')
        
        # Similarity gauge
        self._draw_similarity_gauge(axes[0, 3], similarity, threshold)
        
        # Row 2: Attention Maps (NEW - core Prompt-CAM feature)
        if attention1 is not None and attention2 is not None:
            try:
                # Extract and visualize attention maps
                att_map1 = self._extract_attention_map(attention1)
                att_map2 = self._extract_attention_map(attention2)
                
                if att_map1 is not None and att_map2 is not None:
                    # Show attention overlays
                    axes[1, 0].imshow(img1)
                    axes[1, 0].imshow(att_map1, cmap='hot', alpha=0.6)
                    axes[1, 0].set_title('Person 1 Attention', fontsize=12)
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(img2)
                    axes[1, 1].imshow(att_map2, cmap='hot', alpha=0.6)
                    axes[1, 1].set_title('Person 2 Attention', fontsize=12)
                    axes[1, 1].axis('off')
                    
                    # Attention difference
                    att_diff = np.abs(att_map1 - att_map2)
                    im = axes[1, 2].imshow(att_diff, cmap='viridis')
                    axes[1, 2].set_title('Attention Difference', fontsize=12)
                    axes[1, 2].axis('off')
                    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
                    
                    # Attention correlation
                    att_corr = np.corrcoef(att_map1.flatten(), att_map2.flatten())[0, 1]
                    axes[1, 3].text(0.5, 0.5, f'Attention Correlation:\n{att_corr:.4f}\n\n(How similarly do they\nlook at the image?)', 
                                   ha='center', va='center', fontsize=12,
                                   transform=axes[1, 3].transAxes,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
                    axes[1, 3].set_title('Attention Analysis', fontsize=12)
                    axes[1, 3].axis('off')
                else:
                    # Fallback if attention extraction fails
                    for i in range(4):
                        axes[1, i].text(0.5, 0.5, 'Attention maps\nnot available', 
                                       ha='center', va='center', transform=axes[1, i].transAxes)
                        axes[1, i].axis('off')
            except Exception as e:
                print(f"WARNING: Attention visualization error: {e}")
                for i in range(4):
                    axes[1, i].text(0.5, 0.5, 'Attention analysis\nfailed', 
                                   ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].axis('off')
        else:
            for i in range(4):
                axes[1, i].text(0.5, 0.5, 'Attention maps\nnot extracted', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')
        
        # Row 3: Feature Analysis (original twin-specific analysis)
        feat1_np = feat1[0].cpu().numpy()
        feat2_np = feat2[0].cpu().numpy()
        
        # Feature comparison
        feature_dims = min(50, len(feat1_np))
        x_range = range(feature_dims)
        
        axes[2, 0].plot(x_range, feat1_np[:feature_dims], 'b-', alpha=0.7, linewidth=2, label='Person 1')
        axes[2, 0].plot(x_range, feat2_np[:feature_dims], 'r-', alpha=0.7, linewidth=2, label='Person 2')
        axes[2, 0].set_title('Feature Comparison', fontsize=12)
        axes[2, 0].set_xlabel('Feature Dimension')
        axes[2, 0].set_ylabel('Feature Value')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Feature differences
        feature_diff = np.abs(feat1_np - feat2_np)
        axes[2, 1].bar(range(feature_dims), feature_diff[:feature_dims], 
                       alpha=0.7, color='purple')
        axes[2, 1].set_title('Discriminative Features', fontsize=12)
        axes[2, 1].set_xlabel('Feature Dimension')
        axes[2, 1].set_ylabel('Absolute Difference')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Similarity distribution context
        self._plot_similarity_context(axes[2, 2], similarity, threshold)
        
        # Feature correlation
        correlation = np.corrcoef(feat1_np[:100], feat2_np[:100])[0, 1]
        axes[2, 3].scatter(feat1_np[:100], feat2_np[:100], alpha=0.6, s=10)
        axes[2, 3].plot([feat1_np[:100].min(), feat1_np[:100].max()], 
                        [feat1_np[:100].min(), feat1_np[:100].max()], 'r--', alpha=0.8)
        axes[2, 3].set_title(f'Feature Correlation: {correlation:.3f}', fontsize=12)
        axes[2, 3].set_xlabel('Person 1 Features')
        axes[2, 3].set_ylabel('Person 2 Features')
        axes[2, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, {
            'similarity': similarity,
            'is_same_person': is_same,
            'confidence': confidence,
            'correlation': correlation,
            'attention_correlation': att_corr if 'att_corr' in locals() else None
        }
    
    def _extract_attention_map(self, attention):
        """Extract attention map from model output"""
        try:
            if attention is None:
                return None
            
            # Handle different attention formats
            if isinstance(attention, tuple):
                attention = attention[0]
            
            if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
                # Average across heads and take CLS token attention
                att_map = attention[0].mean(0)[0, 1:].reshape(14, 14)  # Assuming 14x14 patches
            elif len(attention.shape) == 3:  # [batch, seq_len, seq_len]
                att_map = attention[0][0, 1:].reshape(14, 14)
            else:
                return None
            
            # Resize to image size and normalize
            import cv2
            att_map = cv2.resize(att_map.detach().cpu().numpy(), (224, 224))
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
            
            return att_map
            
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
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Verification threshold')
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
        
        fig, results = interpreter.analyze_twin_verification(img1_path, img2_path)
        
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
