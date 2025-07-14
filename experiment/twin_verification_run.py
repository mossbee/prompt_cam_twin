import os
import torch
from experiment.build_model import get_model
from experiment.build_loader import get_loader
from engine.verification_trainer import VerificationTrainer
from utils.global_var import OUTPUT_DIR
from timm.utils import get_outdir
from utils.log_utils import logging_env_setup
from utils.misc import method_name
from datetime import datetime
import yaml
from utils.setup_logging import get_logger

logger = get_logger("TwinVerificationRun")


def twin_verification_run(params):
    """Main training function for twin face verification"""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Setup output directory
    method = f"twin_{params.model}_{params.train_type}"
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    output_dir = os.path.join(OUTPUT_DIR, "twin_verification", method, start_time)
    params.output_dir = get_outdir(output_dir)
    
    # Save parameters
    params_text = yaml.safe_dump(params.__dict__, default_flow_style=False)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(params_text)
    
    # Setup logging
    logging_env_setup(params)
    
    logger.info("=== Twin Face Verification Training ===")
    logger.info(f"Model: {params.model}")
    logger.info(f"Stage 1 Training: {params.stage1_training}")
    logger.info(f"Output Directory: {output_dir}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_loader(params, logger)
    
    # Create model
    logger.info("Creating model...")
    model, tune_parameters, model_grad_params_no_head = get_model(params)
    
    # Create trainer
    trainer = VerificationTrainer(model, params)
    
    # Load checkpoint if specified
    if hasattr(params, 'resume_from') and params.resume_from:
        logger.info(f"Resuming from checkpoint: {params.resume_from}")
        trainer.load_checkpoint(params.resume_from)
    
    try:
        if params.stage1_training:
            # Stage 1: Identity Classification
            logger.info("Starting Stage 1: Identity Classification Training")
            trainer.train_stage1_identity(train_loader, val_loader, params.stage1_epochs)
            
            # Save Stage 1 checkpoint
            stage1_checkpoint = trainer._save_checkpoint(params.stage1_epochs - 1, 'stage1_final')
            logger.info(f"Stage 1 completed. Checkpoint saved: {stage1_checkpoint}")
            
            # Switch to Stage 2 datasets (verification pairs)
            logger.info("Switching to Stage 2 datasets...")
            params.stage1_training = False  # Switch to verification mode
            train_loader, val_loader, test_loader = get_loader(params, logger)
        
        # Stage 2: Verification Training
        logger.info("Starting Stage 2: Verification Training")
        trainer.train_stage2_verification(train_loader, val_loader, params.stage2_epochs)
        
        # Final evaluation
        logger.info("Training completed successfully!")
        if trainer.best_model_path:
            logger.info(f"Best model saved at: {trainer.best_model_path}")
            logger.info(f"Best validation AUC: {trainer.best_val_auc:.4f}")
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise e
    
    finally:
        # Clean up tracking
        trainer.close_tracking()
    
    return trainer.best_model_path, trainer.best_val_auc


def run_twin_verification_inference(model_path, test_loader, params):
    """Run inference on test set with trained model"""
    
    logger.info("=== Twin Face Verification Inference ===")
    
    # Load model
    model, _, _ = get_model(params)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create trainer for inference utilities
    trainer = VerificationTrainer(model, params)
    
    # Run evaluation
    with torch.no_grad():
        test_metrics = trainer._validate_verification_epoch(test_loader)
    
    logger.info("Test Results:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"AUC: {test_metrics['auc']:.4f}")
    logger.info(f"EER: {test_metrics['eer']:.4f}")
    logger.info(f"F1: {test_metrics['f1']:.4f}")
    
    return test_metrics
