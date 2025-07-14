#!/usr/bin/env python3
"""
Evaluation script for Twin Face Verification model.
Tests model performance on test set and finds optimal threshold.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse
import yaml
import os
import sys
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.twin_prompt_cam import TwinPromptCAM, TwinPromptCAMConfig
from experiment.build_loader import get_loader
from utils.verification_metrics import VerificationMetrics
from utils.setup_logging import setup_logging


class TwinVerificationEvaluator:
    """Comprehensive evaluator for twin face verification model"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    def evaluate_on_testset(self, test_loader, save_dir=None):
        """
        Comprehensive evaluation on test set
        
        Returns:
            results: Dictionary containing all evaluation metrics
        """
        print("🔍 Evaluating model on test set...")
        
        all_similarities = []
        all_labels = []
        all_person1_ids = []
        all_person2_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Handle different batch formats
                if isinstance(batch, dict):
                    img1 = batch['img1'].to(self.device)
                    img2 = batch['img2'].to(self.device)
                    labels = batch['label'].to(self.device)
                    person1_idx = batch['person1_idx'].to(self.device)
                    person2_idx = batch['person2_idx'].to(self.device)
                else:
                    # Handle tuple format if needed
                    img1, img2, labels, person1_idx, person2_idx = batch
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    labels = labels.to(self.device)
                    person1_idx = person1_idx.to(self.device)
                    person2_idx = person2_idx.to(self.device)
                
                # Get similarity scores
                similarities = self.model(img1, img2, person1_idx, person2_idx, mode='verification')
                similarities = torch.sigmoid(similarities).squeeze()
                
                # Store results
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_person1_ids.extend(person1_idx.cpu().numpy())
                all_person2_ids.extend(person2_idx.cpu().numpy())
        
        # Convert to numpy arrays
        similarities = np.array(all_similarities)
        labels = np.array(all_labels)
        
        print(f"📊 Collected {len(similarities)} test samples")
        print(f"   Positive pairs: {np.sum(labels == 1)} ({np.mean(labels)*100:.1f}%)")
        print(f"   Negative pairs: {np.sum(labels == 0)} ({(1-np.mean(labels))*100:.1f}%)")
        
        # Comprehensive evaluation
        results = self._compute_comprehensive_metrics(similarities, labels)
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(similarities, labels)
        results['optimal_threshold'] = optimal_threshold
        
        # Compute metrics at optimal threshold
        optimal_predictions = (similarities >= optimal_threshold).astype(int)
        results['optimal_accuracy'] = accuracy_score(labels, optimal_predictions)
        results['optimal_precision'] = self._safe_precision_score(labels, optimal_predictions)
        results['optimal_recall'] = self._safe_recall_score(labels, optimal_predictions)
        results['optimal_f1'] = self._safe_f1_score(labels, optimal_predictions)
        
        # Create visualizations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._create_evaluation_plots(similarities, labels, optimal_threshold, save_dir)
            self._save_detailed_results(results, similarities, labels, all_person1_ids, all_person2_ids, save_dir)
        
        return results
    
    def _compute_comprehensive_metrics(self, similarities, labels):
        """Compute comprehensive evaluation metrics"""
        
        # ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, similarities)
        pr_auc = auc(recall, precision)
        
        # Equal Error Rate (EER)
        eer, eer_threshold = self._compute_eer(fpr, tpr, roc_thresholds)
        
        # Different threshold strategies
        thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_results = {}
        
        for thresh in thresholds_to_test:
            pred = (similarities >= thresh).astype(int)
            acc = accuracy_score(labels, pred)
            prec = self._safe_precision_score(labels, pred)
            rec = self._safe_recall_score(labels, pred)
            f1 = self._safe_f1_score(labels, pred)
            
            threshold_results[f'thresh_{thresh}'] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            }
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'threshold_results': threshold_results,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def _find_optimal_threshold(self, similarities, labels):
        """Find optimal threshold using multiple criteria"""
        
        # Method 1: Youden's J statistic (maximal difference between TPR and FPR)
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        youden_threshold = thresholds[optimal_idx]
        
        # Method 2: Closest to top-left corner in ROC space
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        closest_idx = np.argmin(distances)
        closest_threshold = thresholds[closest_idx]
        
        # Method 3: F1-score maximization
        f1_scores = []
        test_thresholds = np.linspace(0.1, 0.9, 81)
        
        for thresh in test_thresholds:
            pred = (similarities >= thresh).astype(int)
            f1 = self._safe_f1_score(labels, pred)
            f1_scores.append(f1)
        
        f1_optimal_idx = np.argmax(f1_scores)
        f1_threshold = test_thresholds[f1_optimal_idx]
        
        print(f"🎯 Threshold Analysis:")
        print(f"   Youden's J: {youden_threshold:.3f}")
        print(f"   Closest to (0,1): {closest_threshold:.3f}")
        print(f"   F1-optimal: {f1_threshold:.3f}")
        
        # Use F1-optimal as default
        return f1_threshold
    
    def _compute_eer(self, fpr, tpr, thresholds):
        """Compute Equal Error Rate"""
        fnr = 1 - tpr
        abs_diff = np.abs(fpr - fnr)
        min_idx = np.argmin(abs_diff)
        eer = (fpr[min_idx] + fnr[min_idx]) / 2
        eer_threshold = thresholds[min_idx]
        return eer, eer_threshold
    
    def _safe_precision_score(self, y_true, y_pred):
        """Compute precision with handling for edge cases"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _safe_recall_score(self, y_true, y_pred):
        """Compute recall with handling for edge cases"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _safe_f1_score(self, y_true, y_pred):
        """Compute F1-score with handling for edge cases"""
        precision = self._safe_precision_score(y_true, y_pred)
        recall = self._safe_recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _create_evaluation_plots(self, similarities, labels, optimal_threshold, save_dir):
        """Create comprehensive evaluation plots"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Twin Face Verification - Model Evaluation', fontsize=16)
        
        # 1. ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, similarities)
        pr_auc = auc(recall, precision)
        
        axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Similarity Score Distribution
        axes[0, 2].hist(similarities[labels == 0], bins=50, alpha=0.7, label='Different Persons', density=True)
        axes[0, 2].hist(similarities[labels == 1], bins=50, alpha=0.7, label='Same Person', density=True)
        axes[0, 2].axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
        axes[0, 2].set_xlabel('Similarity Score')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Similarity Score Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Confusion Matrix
        predictions = (similarities >= optimal_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # 5. Threshold vs Metrics
        test_thresholds = np.linspace(0.1, 0.9, 81)
        accuracies = []
        f1_scores = []
        
        for thresh in test_thresholds:
            pred = (similarities >= thresh).astype(int)
            acc = accuracy_score(labels, pred)
            f1 = self._safe_f1_score(labels, pred)
            accuracies.append(acc)
            f1_scores.append(f1)
        
        axes[1, 1].plot(test_thresholds, accuracies, label='Accuracy')
        axes[1, 1].plot(test_thresholds, f1_scores, label='F1-Score')
        axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--', label='Optimal')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Threshold vs Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Error Analysis
        fp_similarities = similarities[(labels == 0) & (predictions == 1)]
        fn_similarities = similarities[(labels == 1) & (predictions == 0)]
        
        if len(fp_similarities) > 0:
            axes[1, 2].hist(fp_similarities, bins=20, alpha=0.7, label=f'False Positives ({len(fp_similarities)})')
        if len(fn_similarities) > 0:
            axes[1, 2].hist(fn_similarities, bins=20, alpha=0.7, label=f'False Negatives ({len(fn_similarities)})')
        
        axes[1, 2].axvline(optimal_threshold, color='red', linestyle='--', label='Threshold')
        axes[1, 2].set_xlabel('Similarity Score')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Error Analysis')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plots saved to {save_dir}/evaluation_plots.png")
    
    def _save_detailed_results(self, results, similarities, labels, person1_ids, person2_ids, save_dir):
        """Save detailed results and analysis"""
        
        # Save numerical results
        results_file = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
                else:
                    json_results[key] = float(value) if isinstance(value, np.number) else value
            json.dump(json_results, f, indent=2)
        
        # Save detailed predictions
        predictions_file = os.path.join(save_dir, 'detailed_predictions.csv')
        import pandas as pd
        
        df = pd.DataFrame({
            'person1_id': person1_ids,
            'person2_id': person2_ids,
            'true_label': labels,
            'similarity_score': similarities,
            'prediction': (similarities >= results['optimal_threshold']).astype(int),
            'correct': (labels == (similarities >= results['optimal_threshold']).astype(int))
        })
        
        df.to_csv(predictions_file, index=False)
        
        print(f"💾 Detailed results saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Twin Face Verification Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with evaluation settings
    config_dict['stage1_training'] = False
    config_dict['batch_size'] = args.batch_size
    config_dict['test_batch_size'] = args.batch_size
    
    # Convert to namespace
    config = argparse.Namespace(**config_dict)
    
    print("🚀 Twin Face Verification Model Evaluation")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"⚙️  Config: {args.config}")
    print(f"💾 Output: {args.output_dir}")
    
    # Load model
    model_config = TwinPromptCAMConfig(
        model=config.model,
        drop_path_rate=getattr(config, 'drop_path_rate', 0.1),
        vpt_num=getattr(config, 'vpt_num', 1),
        vpt_mode=getattr(config, 'vpt_mode', 'deep'),
        stage1_training=False
    )
    
    model = TwinPromptCAM(model_config, num_persons=getattr(config, 'num_persons', 356))
    
    # Load checkpoint
    print(f"📦 Loading checkpoint from {args.checkpoint}")
    try:
        # First try with weights_only=True (safe)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"⚠️  Safe loading failed, trying with weights_only=False...")
        # Fallback to weights_only=False (less safe but needed for some checkpoints)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            # Assume the entire checkpoint is the state dict
            model_state_dict = checkpoint
    else:
        model_state_dict = checkpoint
    
    # Load the state dict
    try:
        model.load_state_dict(model_state_dict)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model state dict: {e}")
        print("🔍 Available keys in checkpoint:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"   - {key}")
        return
    
    # Setup data loader for test set
    print("📊 Loading test dataset...")
    
    # Create a simple logger mock for the evaluation
    class SimpleLogger:
        def info(self, msg):
            print(f"ℹ️  {msg}")
        def warning(self, msg):
            print(f"⚠️  {msg}")
        def error(self, msg):
            print(f"❌ {msg}")
    
    logger = SimpleLogger()
    _, _, test_loader = get_loader(config, logger)
    
    if test_loader is None:
        print("❌ No test set found! Make sure your dataset has test data.")
        return
    
    # Create evaluator and run evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = TwinVerificationEvaluator(model, device)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_on_testset(test_loader, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 EVALUATION SUMMARY")
    print("="*60)
    print(f"📊 ROC AUC: {results['roc_auc']:.4f}")
    print(f"📊 PR AUC: {results['pr_auc']:.4f}")
    print(f"⚖️  Equal Error Rate: {results['eer']:.4f}")
    print(f"🎯 Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"✅ Accuracy @ Optimal: {results['optimal_accuracy']:.4f}")
    print(f"🎯 Precision @ Optimal: {results['optimal_precision']:.4f}")
    print(f"🔄 Recall @ Optimal: {results['optimal_recall']:.4f}")
    print(f"🏆 F1-Score @ Optimal: {results['optimal_f1']:.4f}")
    print("="*60)
    
    print(f"\n📁 Detailed results saved to: {args.output_dir}/")
    print("   - evaluation_plots.png: Comprehensive visualization")
    print("   - evaluation_results.json: Numerical results")
    print("   - detailed_predictions.csv: Per-sample predictions")


if __name__ == '__main__':
    main()
