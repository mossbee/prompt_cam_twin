# Twin Face Verification with Prompt-CAM Implementation Plan

## Overview
Adapt Prompt-CAM from multi-class fine-grained classification to twin face verification (binary classification for face pairs). The core idea: use person-specific prompts to extract discriminative facial features, then compare features from two images to determine if they belong to the same person.

## Architecture Adaptation

### Core Approach
- **Original Prompt-CAM**: Class-specific prompts → Multi-class classification
- **Twin Verification**: Person-specific prompts → Feature extraction → Pairwise comparison

### Model Architecture
```
Image1 → ViT + Person-specific Prompts → Features1
Image2 → ViT + Person-specific Prompts → Features2
Features1, Features2 → Similarity Network → Binary Classification (Same/Different)
```

## Key Implementation Components

### 1. Data Loading & Preprocessing (`experiment/build_loader.py`)
- **New Dataset Class**: `TwinPairDataset`
  - Loads image pairs with same/different labels
  - Handles positive pairs (same person) and negative pairs (different people, especially twins)
  - **Image preprocessing**: Minimal - images are already 224x224, only tensor conversion and normalization
- **Data Augmentation**: None - images are already preprocessed and ready
- **Balanced Sampling**: Equal positive/negative pairs per batch

### 2. Model Architecture (`model/`)
- **New File**: `twin_prompt_cam.py`
  - Extends existing VPT architecture
  - Person-specific prompts (356 prompts for training identities)
  - Dual-image processing capability
  - Feature extraction and similarity computation
- **Similarity Network**: Simple MLP for comparing extracted features
- **Shared ViT backbone**: Keep frozen, only train prompts + similarity head

### 3. Training Strategy (`engine/trainer.py`)
- **Contrastive Learning**: 
  - Positive pairs: same person images
  - Hard negative pairs: twin sibling images (most challenging)
  - Easy negative pairs: random different people
- **Loss Function**: Binary cross-entropy or contrastive loss
- **Verification Metrics**: Equal Error Rate (EER), ROC-AUC, precision/recall

### 4. Configuration (`experiment/config/`)
- **New Config Directory**: `twin_verification/`
  - Model configs for different ViT backbones (DINO, DINOv2)
  - Training hyperparameters optimized for verification task
  - Multi-environment support (local/Kaggle)

## Project Structure Modifications

### New Files to Create
```
model/twin_prompt_cam.py          # Main verification model
data/twin_dataset.py              # Dataset class for twin pairs
experiment/twin_verification/     # Config directory
├── dino/args.yaml
├── dinov2/args.yaml
engine/verification_trainer.py   # Specialized trainer for verification
utils/verification_metrics.py    # Verification-specific metrics
```

### Files to Modify
```
experiment/build_model.py         # Add twin verification model
experiment/build_loader.py        # Add twin pair data loading
main.py                          # Add twin verification mode
```

## Training Pipeline

### Stage 1: Person-Specific Prompt Learning
- Train prompts to extract person-specific features
- Use original identity classification loss as auxiliary task
- Goal: Each prompt learns to focus on distinguishing features of one person

### Stage 2: Verification Head Training
- Freeze learned prompts
- Train similarity network on image pairs
- Focus on same/different classification

### Stage 3: End-to-End Fine-tuning (Optional)
- Joint training of prompts + similarity network
- Lower learning rate for prompts

## Resource Utilization

### Multi-GPU Strategy
- **Data Parallel**: Distribute batches across GPUs
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Mixed Precision**: FP16 training for memory efficiency

### Checkpointing Strategy
- **Every Epoch**: Full model checkpoint for Kaggle 12h timeout
- **Best Model**: Save based on validation EER
- **Resume Capability**: Load from any checkpoint seamlessly

### Memory Optimization
- **Frozen ViT**: Only store gradients for prompts + heads
- **Dynamic Batch Size**: Adjust based on available memory
- **Efficient Data Loading**: Prefetch and multi-worker data loading

## Experiment Tracking

### Three-Mode Support
1. **MLFlow** (Local): Automatic logging to local server
2. **WandB** (Kaggle): Use WANDB_API_KEY from secrets
3. **No Tracking**: Minimal overhead option

### Logged Metrics
- Training: Loss, accuracy, learning rate
- Validation: EER, ROC-AUC, precision, recall
- Interpretability: Attention visualizations
- System: GPU utilization, memory usage

## Interpretability Features

### Attention Visualization
- **Same Person**: Show which facial regions prompts focus on for correct matches
- **Different People**: Visualize distinguishing features between twins
- **Error Analysis**: Understand misclassification patterns

### Progressive Trait Discovery
- Adapt greedy head masking from original Prompt-CAM
- Identify minimum set of facial features needed for distinction

## Implementation Priority

### Phase 1 (Core Functionality)
1. Twin dataset implementation
2. Basic verification model
3. Training pipeline
4. Checkpointing system

### Phase 2 (Optimization)
1. Multi-GPU training
2. Memory optimizations
3. Hyperparameter tuning
4. Advanced data augmentation

### Phase 3 (Analysis)
1. Interpretability tools
2. Extensive evaluation
3. Error analysis
4. Visualization dashboard

## Key Design Decisions

### Why This Approach Works
- **Prompt Constraint**: Forces model to focus on genuine distinguishing features
- **Frozen ViT**: Leverages powerful pre-trained representations
- **Person-Specific Learning**: Each prompt specializes in one identity's traits
- **Pairwise Comparison**: Natural fit for verification task

### Risk Mitigation
- **Overfitting**: Strong regularization through frozen backbone
- **Data Imbalance**: Careful positive/negative pair sampling
- **Twin Confusion**: Hard negative mining with twin pairs
- **Resource Limits**: Efficient training with minimal parameters

This plan provides a clear roadmap for implementing twin face verification while leveraging the existing Prompt-CAM codebase efficiently.
