import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vision_transformer import VisionTransformer
from model.vpt import VPT


class TwinPromptCAM(nn.Module):
    """
    Twin Face Verification model using Prompt-CAM approach.
    
    Architecture:
    1. Shared ViT backbone with person-specific prompts
    2. Feature extraction for each image using corresponding person prompts
    3. Similarity computation between extracted features
    4. Binary classification (same/different person)
    """
    
    def __init__(self, params, num_persons=356):
        super().__init__()
        self.params = params
        self.num_persons = num_persons
        
        # Model-specific configurations
        if params.model == 'dinov2':
            self.depth = 12
            self.embed_dim = 768
            self.patch_size = 14
        elif params.model == 'dino':
            self.depth = 12
            self.embed_dim = 768
            self.patch_size = 16
        else:
            # Default values
            self.depth = 12
            self.embed_dim = 768
            self.patch_size = 16
        
        # Initialize ViT backbone (frozen)
        self.backbone = self._build_backbone()
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Person-specific prompts (similar to VPT but for persons instead of classes)
        self.person_prompts = VPT(params, 
                                 depth=self.depth,
                                 patch_size=self.patch_size,
                                 embed_dim=self.embed_dim)
        
        # Override VPT to have person-specific prompts
        self._init_person_prompts()
        
        # Shared projection head for feature extraction
        self.feature_dim = 256
        self.feature_projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Similarity computation network
        self.similarity_net = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # For identity classification (Stage 1)
        if hasattr(params, 'stage1_training') and params.stage1_training:
            self.identity_classifier = nn.Linear(self.feature_dim, num_persons)
    
    def _build_backbone(self):
        """Build the ViT backbone based on config - using existing infrastructure"""
        import timm
        
        # Use the existing model creation approach
        if self.params.model == 'dinov2':
            model = timm.create_model("vit_base_patch14_dinov2_petl", 
                                    drop_path_rate=self.params.drop_path_rate,
                                    pretrained=False,
                                    params=self.params)
            # Note: In actual usage, you'd load pretrained weights here
            # if not visualize:
            #     model.load_pretrained('pretrained_weights/dinov2_vitb14_pretrain.pth')
        elif self.params.model == 'dino':
            model = timm.create_model("vit_base_patch16_dino_petl", 
                                    drop_path_rate=self.params.drop_path_rate,
                                    pretrained=False,
                                    params=self.params)
            # Note: In actual usage, you'd load pretrained weights here
            # if not visualize:
            #     model.load_pretrained('pretrained_weights/dino_vitbase16_pretrain.pth')
        else:
            raise NotImplementedError(f"Model {self.params.model} not implemented")
        
        # Remove the classification head since we'll use our own
        model.reset_classifier(0)
        
        return model
    
    def _init_person_prompts(self):
        """Initialize person-specific prompts"""
        # Modify VPT to have num_persons prompts instead of vpt_num
        val = 0.02  # Small initialization
        self.person_prompts.prompt_embeddings = nn.Parameter(
            torch.zeros(self.depth, self.num_persons, self.embed_dim)
        )
        nn.init.uniform_(self.person_prompts.prompt_embeddings.data, -val, val)
    
    def extract_features(self, images, person_indices):
        """
        Extract person-specific features from images.
        
        Args:
            images: Batch of images [B, C, H, W]
            person_indices: Person indices for selecting prompts [B]
        
        Returns:
            features: Extracted features [B, feature_dim]
        """
        batch_size = images.size(0)
        
        # Since all images in a batch should use the same person prompt for proper batching,
        # we'll use the first person's index for the entire batch
        # In practice, the dataloader should ensure all images in a batch belong to the same person
        if len(person_indices) > 0:
            person_idx = person_indices[0].item()
            
            # Temporarily set the VPT prompts to the person-specific prompt
            original_prompts = self.backbone.vpt.prompt_embeddings.data.clone()
            
            # Replicate the person-specific prompt across all VPT layers and positions
            person_prompt = self.person_prompts.prompt_embeddings[:, person_idx:person_idx+1, :].clone()
            self.backbone.vpt.prompt_embeddings.data = person_prompt
            
            try:
                # Forward through backbone with person-specific prompts
                x, _ = self.backbone.forward_features(images)
                
                # For prompt_cam mode, x contains the prompt outputs
                if self.params.train_type == 'prompt_cam':
                    # Use the first prompt token as the feature representation
                    cls_features = x[:, 0, :]  # [B, embed_dim]
                else:
                    # Fallback: use CLS token
                    cls_features = x[:, 0, :]  # [B, embed_dim]
                    
            finally:
                # Always restore original prompts
                self.backbone.vpt.prompt_embeddings.data = original_prompts
        else:
            # If no person indices provided, use standard forward pass
            x, _ = self.backbone.forward_features(images)
            cls_features = x[:, 0, :]  # [B, embed_dim]
        
        # Project to feature space
        features = self.feature_projector(cls_features)
        
        return features
    
    def forward(self, img1, img2, person1_idx, person2_idx, mode='verification'):
        """
        Forward pass for twin verification.
        
        Args:
            img1, img2: Input image pairs [B, C, H, W]
            person1_idx, person2_idx: Person indices [B]
            mode: 'verification' or 'identity_classification'
        
        Returns:
            For verification: similarity scores [B, 1]
            For identity: classification logits [B, num_persons]
        """
        if mode == 'identity_classification':
            # Stage 1: Identity classification
            features1 = self.extract_features(img1, person1_idx)
            logits = self.identity_classifier(features1)
            return logits
        
        elif mode == 'verification':
            # Stage 2: Verification
            features1 = self.extract_features(img1, person1_idx)
            features2 = self.extract_features(img2, person2_idx)
            
            # Concatenate features
            combined_features = torch.cat([features1, features2], dim=1)
            
            # Compute similarity
            similarity = self.similarity_net(combined_features)
            
            return similarity
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_attention_maps(self, image, person_idx, layer_idx=-1):
        """
        Extract attention maps for interpretability.
        
        Args:
            image: Input image [1, C, H, W]
            person_idx: Person index [1]
            layer_idx: Which layer to extract attention from (-1 for last layer)
        
        Returns:
            attention_maps: Multi-head attention maps
        """
        with torch.no_grad():
            # For now, return a placeholder since attention map extraction
            # requires more complex integration with the backbone architecture
            # This can be implemented later when the basic functionality works
            
            # Extract basic features first
            features = self.extract_features(image, torch.tensor([person_idx]))
            
            # Return dummy attention maps for now
            H, W = image.shape[-2:]
            patch_size = self.patch_size
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            
            # Create dummy attention map
            attention_map = torch.ones(1, num_patches_h, num_patches_w)
            
            return attention_map
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained"""
        trainable_params = []
        
        # Person prompts
        trainable_params.extend(list(self.person_prompts.parameters()))
        
        # Feature projector
        trainable_params.extend(list(self.feature_projector.parameters()))
        
        # Similarity network
        trainable_params.extend(list(self.similarity_net.parameters()))
        
        # Identity classifier (if exists)
        if hasattr(self, 'identity_classifier'):
            trainable_params.extend(list(self.identity_classifier.parameters()))
        
        return trainable_params


class TwinPromptCAMConfig:
    """Configuration class for TwinPromptCAM"""
    
    def __init__(self, model='dinov2', drop_path_rate=0.1, 
                 vpt_num=1, vpt_mode='deep', stage1_training=False):
        self.model = model
        self.drop_path_rate = drop_path_rate
        self.vpt_num = vpt_num
        self.vpt_mode = vpt_mode
        self.stage1_training = stage1_training
        
        # Required attributes for vision transformer compatibility
        self.train_type = 'prompt_cam'  # Use prompt_cam training type
        self.vpt_dropout = 0.1  # Default VPT dropout
        self.vpt_layer = None   # Use all layers by default
