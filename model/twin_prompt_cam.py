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
        
        # Initialize ViT backbone (frozen)
        self.backbone = self._build_backbone()
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Person-specific prompts (similar to VPT but for persons instead of classes)
        self.person_prompts = VPT(params, 
                                 depth=self.backbone.depth,
                                 patch_size=self.backbone.patch_embed.patch_size,
                                 embed_dim=self.backbone.embed_dim)
        
        # Override VPT to have person-specific prompts
        self._init_person_prompts()
        
        # Shared projection head for feature extraction
        self.feature_dim = 256
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, self.feature_dim),
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
        """Build the ViT backbone based on config"""
        # This should match the existing model building logic
        if self.params.model == 'dinov2':
            model = VisionTransformer(
                img_size=224,
                patch_size=14,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                num_classes=0,  # No classification head
                drop_path_rate=self.params.drop_path_rate
            )
        elif self.params.model == 'dino':
            model = VisionTransformer(
                img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                num_classes=0,
                drop_path_rate=self.params.drop_path_rate
            )
        else:
            raise NotImplementedError(f"Model {self.params.model} not implemented")
        
        return model
    
    def _init_person_prompts(self):
        """Initialize person-specific prompts"""
        # Modify VPT to have num_persons prompts instead of vpt_num
        val = 0.02  # Small initialization
        self.person_prompts.prompt_embeddings = nn.Parameter(
            torch.zeros(self.backbone.depth, self.num_persons, self.backbone.embed_dim)
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
        
        # Prepare prompts for each image based on person indices
        prompts = []
        for layer_idx in range(self.backbone.depth):
            layer_prompts = []
            for batch_idx in range(batch_size):
                person_idx = person_indices[batch_idx]
                person_prompt = self.person_prompts.prompt_embeddings[layer_idx, person_idx:person_idx+1]
                layer_prompts.append(person_prompt)
            prompts.append(torch.cat(layer_prompts, dim=0))
        
        # Forward through backbone with person-specific prompts
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        
        # Add prompts to each layer
        for layer_idx, layer in enumerate(self.backbone.blocks):
            # Add person-specific prompts to this layer
            layer_prompt = prompts[layer_idx]  # [B, 1, embed_dim]
            x = torch.cat([layer_prompt, x], dim=1)
            
            # Forward through layer
            x = layer(x)
            
            # Remove prompts from output (keep only image patches + CLS)
            x = x[:, 1:]  # Remove the first token (prompt)
        
        x = self.backbone.norm(x)
        
        # Use CLS token for feature extraction
        cls_features = x[:, 0]  # [B, embed_dim]
        
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
            # Similar to extract_features but return attention maps
            x = self.backbone.patch_embed(image)
            x = self.backbone._pos_embed(x)
            
            target_layer = self.backbone.blocks[layer_idx]
            
            # Add person-specific prompt
            person_prompt = self.person_prompts.prompt_embeddings[layer_idx, person_idx:person_idx+1]
            x = torch.cat([person_prompt, x], dim=1)
            
            # Get attention maps from the target layer
            # This would need to be implemented based on the specific ViT architecture
            attention_maps = target_layer.get_attention_maps(x)
            
            return attention_maps
    
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
