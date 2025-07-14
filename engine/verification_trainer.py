import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
from utils.verification_metrics import VerificationMetrics, TwinVerificationAnalyzer
from utils.setup_logging import get_logger

# Optional imports for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = get_logger("TwinVerificationTrainer")


class VerificationTrainer:
    """Trainer for twin face verification task"""
    
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Get trainable parameters
        if hasattr(self.model, 'module'):
            trainable_params = self.model.module.get_trainable_parameters()
        else:
            trainable_params = self.model.get_trainable_parameters()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(trainable_params)
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.verification_criterion = nn.BCEWithLogitsLoss()
        self.identity_criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision training
        self.scaler = GradScaler() if getattr(params, 'use_amp', False) else None
        
        # Metrics
        self.train_metrics = VerificationMetrics()
        self.val_metrics = VerificationMetrics()
        self.twin_analyzer = TwinVerificationAnalyzer()
        
        # Training state
        self.epoch = 0
        self.best_val_auc = 0.0
        self.best_model_path = None
        
        # Experiment tracking
        self._setup_tracking()
    
    def _setup_optimizer(self, trainable_params):
        """Setup optimizer"""
        lr = self.params.lr
        weight_decay = getattr(self.params, 'weight_decay', 0.01)  # Default weight decay
        
        if self.params.optimizer == 'adam':
            return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif self.params.optimizer == 'adamw':
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif self.params.optimizer == 'sgd':
            return optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.params.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if hasattr(self.params, 'scheduler') and self.params.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.epoch)
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def _setup_tracking(self):
        """Setup experiment tracking"""
        self.tracking_mode = getattr(self.params, 'tracking_mode', 'none')
        
        if self.tracking_mode == 'mlflow':
            if MLFLOW_AVAILABLE:
                mlflow.start_run()
                mlflow.log_params(vars(self.params))
                logger.info("MLFlow tracking initialized")
            else:
                logger.warning("MLFlow not available, switching to no tracking")
                self.tracking_mode = 'none'
        elif self.tracking_mode == 'wandb':
            if WANDB_AVAILABLE:
                wandb.init(
                    project="twin-face-verification",
                    entity=getattr(self.params, 'wandb_entity', None),
                    config=vars(self.params)
                )
                logger.info("WandB tracking initialized")
            else:
                logger.warning("WandB not available, switching to no tracking")
                self.tracking_mode = 'none'
    
    def train_stage1_identity(self, train_loader, val_loader, num_epochs):
        """
        Stage 1: Train person-specific prompts with identity classification
        """
        logger.info("Starting Stage 1: Identity Classification Training")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_acc = self._train_identity_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self._validate_identity_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics({
                'stage': 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Save checkpoint
            if epoch % self.params.save_freq == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(epoch, 'stage1')
            
            logger.info(f"Stage 1 Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def train_stage2_verification(self, train_loader, val_loader, num_epochs):
        """
        Stage 2: Train verification head with pair classification
        """
        logger.info("Starting Stage 2: Verification Training")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self._train_verification_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate_verification_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best model
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_model_path = self._save_checkpoint(epoch, 'stage2_best')
            
            # Log metrics
            metrics_to_log = {
                'stage': 2,
                'epoch': epoch,
                'train_loss': train_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            metrics_to_log.update({f'val_{k}': v for k, v in val_metrics.items()})
            self._log_metrics(metrics_to_log)
            
            # Save regular checkpoint
            if epoch % self.params.save_freq == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(epoch, 'stage2')
            
            logger.info(f"Stage 2 Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                       f"Val AUC: {val_metrics['auc']:.4f}, Val EER: {val_metrics['eer']:.4f}")
    
    def _train_identity_epoch(self, train_loader):
        """Train one epoch for identity classification"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            person_indices = batch['label']  # Same as labels for identity task
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images, images, person_indices, person_indices, 
                                       mode='identity_classification')
                    loss = self.identity_criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, images, person_indices, person_indices, 
                                   mode='identity_classification')
                loss = self.identity_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_identity_epoch(self, val_loader):
        """Validate one epoch for identity classification"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                person_indices = batch['label']
                
                outputs = self.model(images, images, person_indices, person_indices, 
                                   mode='identity_classification')
                loss = self.identity_criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _train_verification_epoch(self, train_loader):
        """Train one epoch for verification"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            labels = batch['label'].to(self.device).float()
            person1_idx = batch['person1_idx'].to(self.device)
            person2_idx = batch['person2_idx'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(img1, img2, person1_idx, person2_idx, 
                                       mode='verification')
                    loss = self.verification_criterion(outputs.squeeze(), labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(img1, img2, person1_idx, person2_idx, 
                                   mode='verification')
                loss = self.verification_criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update metrics
            self.train_metrics.update(outputs.squeeze(), labels.int())
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def _validate_verification_epoch(self, val_loader):
        """Validate one epoch for verification"""
        self.model.eval()
        self.val_metrics.reset()
        self.twin_analyzer.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                labels = batch['label'].to(self.device)
                person1_idx = batch['person1_idx'].to(self.device)
                person2_idx = batch['person2_idx'].to(self.device)
                is_twin_pairs = batch['is_twin_pair']
                
                outputs = self.model(img1, img2, person1_idx, person2_idx, 
                                   mode='verification')
                
                # Update metrics
                self.val_metrics.update(outputs.squeeze(), labels.int())
                self.twin_analyzer.update(outputs.squeeze(), labels.int(), is_twin_pairs)
        
        metrics = self.val_metrics.compute_all_metrics()
        twin_analysis = self.twin_analyzer.compute_twin_vs_nontwin_performance()
        
        # Add twin analysis to metrics
        metrics.update({
            'twin_accuracy': twin_analysis['twin_pairs']['accuracy'],
            'twin_auc': twin_analysis['twin_pairs']['auc'],
            'twin_count': twin_analysis['twin_pairs']['count']
        })
        
        return metrics
    
    def _save_checkpoint(self, epoch, stage):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'params': vars(self.params)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.params.output_dir, f'checkpoint_{stage}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.epoch = checkpoint['epoch']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def _log_metrics(self, metrics):
        """Log metrics to tracking system"""
        if self.tracking_mode == 'mlflow' and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=self.epoch)
        elif self.tracking_mode == 'wandb' and WANDB_AVAILABLE:
            wandb.log(metrics, step=self.epoch)
    
    def close_tracking(self):
        """Close experiment tracking"""
        if self.tracking_mode == 'mlflow' and MLFLOW_AVAILABLE:
            mlflow.end_run()
        elif self.tracking_mode == 'wandb' and WANDB_AVAILABLE:
            wandb.finish()
