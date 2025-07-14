import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class VerificationMetrics:
    """Metrics for face verification task"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.scores = []
        self.labels = []
        self.predictions = []
    
    def update(self, scores, labels, threshold=0.5):
        """
        Update metrics with batch of predictions
        
        Args:
            scores: Raw similarity scores from model [B, 1] or [B]
            labels: Ground truth labels [B] (1 for same person, 0 for different)
            threshold: Threshold for binary classification
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Flatten if needed
        scores = scores.flatten()
        labels = labels.flatten()
        
        # Apply sigmoid to convert to probabilities
        probs = self._sigmoid(scores)
        preds = (probs > threshold).astype(int)
        
        self.scores.extend(probs.tolist())
        self.labels.extend(labels.tolist())
        self.predictions.extend(preds.tolist())
    
    def _sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def compute_eer(self):
        """Compute Equal Error Rate (EER)"""
        if len(self.scores) == 0:
            return 0.0
        
        fpr, tpr, thresholds = roc_curve(self.labels, self.scores)
        fnr = 1 - tpr
        
        # Find threshold where FPR = FNR
        eer_threshold_idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = fpr[eer_threshold_idx]
        
        return eer, thresholds[eer_threshold_idx]
    
    def compute_auc(self):
        """Compute Area Under ROC Curve"""
        if len(self.scores) == 0:
            return 0.0
        
        return roc_auc_score(self.labels, self.scores)
    
    def compute_accuracy(self):
        """Compute accuracy with current predictions"""
        if len(self.predictions) == 0:
            return 0.0
        
        return accuracy_score(self.labels, self.predictions)
    
    def compute_precision_recall(self):
        """Compute precision and recall"""
        if len(self.predictions) == 0:
            return 0.0, 0.0
        
        precision = precision_score(self.labels, self.predictions, zero_division=0)
        recall = recall_score(self.labels, self.predictions, zero_division=0)
        
        return precision, recall
    
    def compute_f1(self):
        """Compute F1 score"""
        if len(self.predictions) == 0:
            return 0.0
        
        return f1_score(self.labels, self.predictions, zero_division=0)
    
    def compute_optimal_threshold(self):
        """Find optimal threshold based on F1 score"""
        if len(self.scores) == 0:
            return 0.5
        
        precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return optimal_threshold, f1_scores[optimal_idx]
    
    def compute_all_metrics(self):
        """Compute all metrics and return as dictionary"""
        if len(self.scores) == 0:
            return {
                'accuracy': 0.0,
                'auc': 0.0,
                'eer': 1.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'optimal_threshold': 0.5
            }
        
        eer, eer_threshold = self.compute_eer()
        auc = self.compute_auc()
        accuracy = self.compute_accuracy()
        precision, recall = self.compute_precision_recall()
        f1 = self.compute_f1()
        optimal_threshold, optimal_f1 = self.compute_optimal_threshold()
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1
        }
    
    def get_summary_string(self):
        """Get formatted string of all metrics"""
        metrics = self.compute_all_metrics()
        
        summary = (
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"AUC: {metrics['auc']:.4f} | "
            f"EER: {metrics['eer']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )
        
        return summary


class TwinVerificationAnalyzer:
    """Analyzer for twin-specific verification performance"""
    
    def __init__(self):
        self.twin_pairs_results = []
        self.non_twin_results = []
        self.all_results = []
    
    def update(self, scores, labels, is_twin_pairs):
        """
        Update with batch results
        
        Args:
            scores: Similarity scores
            labels: Ground truth labels
            is_twin_pairs: Boolean array indicating if pair are twins
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().flatten()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy().flatten()
        
        # Separate twin pairs from non-twin pairs
        for i, (score, label, is_twin) in enumerate(zip(scores, labels, is_twin_pairs)):
            result = {'score': score, 'label': label}
            
            self.all_results.append(result)
            
            if is_twin and label == 0:  # Different people who are twins
                self.twin_pairs_results.append(result)
            else:
                self.non_twin_results.append(result)
    
    def compute_twin_vs_nontwin_performance(self):
        """Compare performance on twin pairs vs non-twin pairs"""
        
        def compute_metrics_for_subset(results):
            if not results:
                return {'accuracy': 0.0, 'auc': 0.0, 'count': 0}
            
            scores = [r['score'] for r in results]
            labels = [r['label'] for r in results]
            
            # Apply sigmoid to scores
            probs = 1 / (1 + np.exp(-np.array(scores)))
            preds = (probs > 0.5).astype(int)
            
            accuracy = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
            
            return {
                'accuracy': accuracy,
                'auc': auc,
                'count': len(results)
            }
        
        twin_metrics = compute_metrics_for_subset(self.twin_pairs_results)
        non_twin_metrics = compute_metrics_for_subset(self.non_twin_results)
        all_metrics = compute_metrics_for_subset(self.all_results)
        
        return {
            'twin_pairs': twin_metrics,
            'non_twin_pairs': non_twin_metrics,
            'overall': all_metrics
        }
    
    def reset(self):
        """Reset all accumulated results"""
        self.twin_pairs_results = []
        self.non_twin_results = []
        self.all_results = []
