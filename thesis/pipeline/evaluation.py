"""
V7 Pipeline - Unified Evaluation Module
Comprehensive metrics for comparing all solutions

Metrics:
1. Per-stage F1, Precision, Recall, Accuracy
2. Pipeline end-to-end F1
3. Per-class performance (especially rare classes)
4. Parameter efficiency
5. Training time and convergence
6. Inference speed
"""

import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
import numpy as np
import time
from typing import Dict, List
import json


class MetricsCalculator:
    """Calculate comprehensive metrics for model evaluation"""
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, num_classes=None, average='macro'):
        """
        Calculate standard classification metrics
        
        Args:
            y_true: True labels [N]
            y_pred: Predicted labels [N]
            num_classes: Number of classes (for per-class metrics)
            average: 'macro', 'micro', 'weighted'
        
        Returns:
            Dict with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        if num_classes:
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
            
            metrics['f1_per_class'] = f1_per_class.tolist()
            metrics['precision_per_class'] = precision_per_class.tolist()
            metrics['recall_per_class'] = recall_per_class.tolist()
        
        return metrics
    
    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred, num_classes=None):
        """Calculate confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes) if num_classes else None)
        return cm
    
    @staticmethod
    def get_classification_report(y_true, y_pred, target_names=None):
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


class PipelineEvaluator:
    """
    Evaluate full hierarchical pipeline
    
    Compares:
    - Baseline v6 reproduction
    - Solution 1: Conv-Adapter
    - Solution 2: Multi-Stage Ensemble
    - Solution 3: Hybrid (Adapter + Ensemble)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_model(self, model, dataloader, stage_name='pipeline'):
        """
        Evaluate a single model on dataloader
        
        Returns:
            Dict with metrics and predictions
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['block'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                if hasattr(model, 'forward'):
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                else:
                    logits = model(x)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predictions
                if logits.dim() == 1:
                    # Binary (sigmoid)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                else:
                    # Multi-class
                    preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Calculate metrics
        num_classes = len(np.unique(all_labels))
        metrics = self.metrics_calc.calculate_classification_metrics(
            all_labels, all_preds, num_classes=num_classes
        )
        
        # Add inference metrics
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['total_samples'] = len(all_labels)
        
        # Confusion matrix
        cm = self.metrics_calc.calculate_confusion_matrix(all_labels, all_preds, num_classes)
        metrics['confusion_matrix'] = cm.tolist()
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels,
            'stage': stage_name
        }
    
    def evaluate_hierarchical_pipeline(self, pipeline, dataloader):
        """
        Evaluate full hierarchical pipeline with stage-wise metrics
        
        Returns:
            Dict with metrics for each stage + overall
        """
        pipeline.eval()
        
        all_stage1_preds = []
        all_stage2_preds = []
        all_final_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['block'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions with intermediates
                final_pred, intermediates = pipeline(x, return_intermediates=True)
                
                all_stage1_preds.append(intermediates.get('stage1_pred', None))
                all_stage2_preds.append(intermediates.get('stage2_pred', None))
                all_final_preds.append(final_pred.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate
        all_final_preds = torch.cat(all_final_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Overall pipeline metrics
        overall_metrics = self.metrics_calc.calculate_classification_metrics(
            all_labels, all_final_preds, num_classes=10
        )
        
        # Confusion matrix
        cm = self.metrics_calc.calculate_confusion_matrix(all_labels, all_final_preds, num_classes=10)
        overall_metrics['confusion_matrix'] = cm.tolist()
        
        return {
            'overall': overall_metrics,
            'predictions': all_final_preds,
            'labels': all_labels
        }
    
    def compare_solutions(self, models_dict, dataloader):
        """
        Compare multiple solutions side-by-side
        
        Args:
            models_dict: Dict of {name: model}
            dataloader: Validation/test dataloader
        
        Returns:
            Comparative analysis dict
        """
        results = {}
        
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            if 'pipeline' in name.lower() or 'hierarchical' in name.lower():
                results[name] = self.evaluate_hierarchical_pipeline(model, dataloader)
            else:
                results[name] = self.evaluate_model(model, dataloader, stage_name=name)
        
        # Comparative summary
        summary = {
            'f1_scores': {name: res['metrics']['f1_macro'] if 'metrics' in res else res['overall']['f1_macro'] 
                         for name, res in results.items()},
            'accuracies': {name: res['metrics']['accuracy'] if 'metrics' in res else res['overall']['accuracy']
                          for name, res in results.items()},
        }
        
        # Best model
        best_model = max(summary['f1_scores'].items(), key=lambda x: x[1])
        summary['best_model'] = {'name': best_model[0], 'f1_score': best_model[1]}
        
        return {
            'detailed_results': results,
            'summary': summary
        }
    
    def parameter_efficiency_analysis(self, models_dict):
        """
        Analyze parameter efficiency of different solutions
        
        Returns:
            Dict with parameter counts and efficiency metrics
        """
        analysis = {}
        
        for name, model in models_dict.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            analysis[name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': total_params - trainable_params,
                'percentage_trainable': 100 * trainable_params / total_params if total_params > 0 else 0
            }
            
            # Check for adapter-specific analysis
            if hasattr(model, 'num_parameters_analysis'):
                analysis[name]['adapter_analysis'] = model.num_parameters_analysis()
        
        return analysis


def save_evaluation_results(results, output_path):
    """Save evaluation results to JSON"""
    # Convert numpy arrays to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_evaluation_results(input_path):
    """Load evaluation results from JSON"""
    with open(input_path, 'r') as f:
        return json.load(f)


# Partition type names for readable reports
PARTITION_NAMES = [
    'NONE',      # 0
    'HORZ',      # 1
    'VERT',      # 2
    'SPLIT',     # 3
    'HORZ_A',    # 4
    'HORZ_B',    # 5
    'VERT_A',    # 6
    'VERT_B',    # 7
    'HORZ_4',    # 8
    'VERT_4'     # 9
]


def print_evaluation_report(results, partition_names=PARTITION_NAMES):
    """Print human-readable evaluation report"""
    print("=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    if 'summary' in results:
        print("\nSUMMARY:")
        print(f"Best Model: {results['summary']['best_model']['name']}")
        print(f"Best F1: {results['summary']['best_model']['f1_score']:.4f}")
        
        print("\nF1 Scores Comparison:")
        for name, f1 in results['summary']['f1_scores'].items():
            print(f"  {name:30s}: {f1:.4f}")
    
    if 'detailed_results' in results:
        for name, res in results['detailed_results'].items():
            print(f"\n{'=' * 80}")
            print(f"MODEL: {name}")
            print(f"{'=' * 80}")
            
            metrics = res.get('metrics', res.get('overall', {}))
            
            print(f"\nOverall Metrics:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"  F1 (macro): {metrics.get('f1_macro', 0):.4f}")
            print(f"  Precision:  {metrics.get('precision_macro', 0):.4f}")
            print(f"  Recall:     {metrics.get('recall_macro', 0):.4f}")
            
            if 'f1_per_class' in metrics:
                print(f"\nPer-Class F1 Scores:")
                for i, f1 in enumerate(metrics['f1_per_class']):
                    class_name = partition_names[i] if i < len(partition_names) else f"Class_{i}"
                    print(f"  {class_name:10s}: {f1:.4f}")
