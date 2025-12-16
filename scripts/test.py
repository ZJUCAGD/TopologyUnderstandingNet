"""
Test and Evaluation Script for Persistence Diagram Significance Detection

Main Features:
1. Model testing and performance evaluation
2. Prediction result visualization
3. Error analysis and case studies
4. Model inference and batch prediction
5. Result export and report generation
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import PersistenceSignificanceNet
from dataset import PersistenceDataset, collate_fn
import shutil
from utils import (
    load_checkpoint, visualize_persistence_diagram, plot_confusion_matrix,
    compute_persistence_statistics, save_predictions, create_experiment_dir
)


class ModelEvaluator:
    """Model Evaluator"""
    
    def __init__(self, model_path: str, config_path: str, device: str = None):
        """
        Args:
            model_path: Model checkpoint path
            config_path: Configuration file path
            device: Computing device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.results = {
            'predictions': [],
            'targets': [],
            'probabilities': [],
            'sample_info': []
        }
        
    def _load_model(self, model_path: str):
        """Load model"""
        model_config = self.config['model']
        model = PersistenceSignificanceNet(
            pd_input_dim=model_config['pd_input_dim'],
            pc_input_dim=model_config['pc_input_dim'],
            aux_dim=model_config['aux_dim'],
            hidden_dim=model_config['hidden_dim'],
            fusion_dim=model_config['fusion_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded: {model_path}")
        
        return model
    
    def evaluate_dataset(self, dataset, data_loader):
        """Evaluate dataset"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_sample_info = []
        sample_names = [] 
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pd_points = batch['pd_points'].to(self.device)
                point_clouds = batch['point_cloud'].to(self.device)
                targets = batch['significance_labels'].to(self.device)
                
                # forward pass
                aux_features = batch.get('aux_features', None)
                if aux_features is not None:
                    aux_features = aux_features.to(self.device)
                
                try:
                    outputs = self.model(pd_points, point_clouds, aux_features)
                    probabilities = torch.softmax(outputs, dim=-1)
                    predictions = torch.argmax(outputs, dim=-1)
                except torch.cuda.OutOfMemoryError:
                    print(f"Batch {batch_idx} out of memory, trying sample-by-sample processing...")
                    
                    batch_predictions = []
                    batch_probabilities = []
                    batch_size = pd_points.shape[0]
                    
                    for i in range(batch_size):
                        single_pd = pd_points[i:i+1]
                        single_pc = point_clouds[i:i+1]
                        single_aux = aux_features[i:i+1] if aux_features is not None else None
                        
                        single_output = self.model(single_pd, single_pc, single_aux)
                        single_prob = torch.softmax(single_output, dim=-1)
                        single_pred = torch.argmax(single_output, dim=-1)
                        
                        batch_predictions.append(single_pred)
                        batch_probabilities.append(single_prob)
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    predictions = torch.cat(batch_predictions, dim=0)
                    probabilities = torch.cat(batch_probabilities, dim=0)
                
                batch_size = pd_points.shape[0]
                for i in range(batch_size):
                    if 'sample_names' in batch and i < len(batch['sample_names']):
                        sample_name = batch['sample_names'][i]
                    else:
                        sample_name = f'sample_{batch_idx}_{i}'
                    
                    num_points = batch['num_pd_points'][i].item()
                    
                    sample_predictions = predictions[i, :num_points]
                    sample_targets = targets[i, :num_points]
                    sample_probabilities = probabilities[i, :num_points]
                    
                    for point_idx in range(num_points):
                        sample_names.append(f'{sample_name}_point_{point_idx}')
                    
                    all_predictions.extend(sample_predictions.cpu().numpy())
                    all_targets.extend(sample_targets.cpu().numpy())
                    all_probabilities.extend(sample_probabilities.cpu().numpy())
                    
                    sample_info = {
                        'sample_name': sample_name,
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'num_points': num_points,
                        'pd_points': pd_points[i, :num_points].cpu().numpy()
                    }
                    all_sample_info.append(sample_info)
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        metrics = self._compute_metrics(predictions, targets, probabilities)
        
        self.results = {
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'sample_info': all_sample_info,
            'sample_names': sample_names,
            'metrics': metrics
        }
        
        return metrics
    
    def _compute_metrics(self, targets: np.ndarray, predictions: np.ndarray, 
                        probabilities: np.ndarray) -> dict:
        """Compute evaluation metrics"""
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, average='binary', pos_label=1)
        metrics['recall'] = recall_score(targets, predictions, average='binary', pos_label=1)
        metrics['f1'] = f1_score(targets, predictions, average='binary', pos_label=1)
        
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro')
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro')
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro')
        
        if len(np.unique(targets)) > 1: 
            metrics['roc_auc'] = roc_auc_score(targets, probabilities[:, 1])
        
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        unique, counts = np.unique(targets, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
        
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        metrics['prediction_distribution'] = dict(zip(unique_pred.tolist(), counts_pred.tolist()))
        
        return metrics
    
    def analyze_errors(self, save_dir: str = None):
        """Error analysis"""
        if len(self.results['predictions']) == 0:
            print("Please run evaluation first")
            return
        
        predictions = self.results['predictions']
        targets = self.results['targets']
        probabilities = self.results['probabilities']
        sample_names = self.results.get('sample_names', [])
        
        # find error predictions
        errors = predictions != targets
        error_indices = np.where(errors)[0]
        
        print(f"Total samples: {len(targets)}")
        print(f"Error predictions: {len(error_indices)}")
        print(f"Error rate: {len(error_indices) / len(targets) * 100:.2f}%")
        
        if len(error_indices) == 0:
            print("No error predictions!")
            
        
        # Print sample names with prediction errors
        print("\nSamples with prediction errors:")
        error_sample_names = set()
        for idx in error_indices:
            if idx < len(sample_names):
                sample_name = sample_names[idx]
                base_sample_name = '_'.join(sample_name.split('_')[:-2]) if '_point_' in sample_name else sample_name
                error_sample_names.add(base_sample_name)
        
        for sample_name in sorted(error_sample_names):
            print(f"  - {sample_name}")
        
        print(f"Total {len(error_sample_names)} samples with prediction errors")
        
        # analyze error types
        false_positives = error_indices[(targets[error_indices] == 0) & (predictions[error_indices] == 1)]
        false_negatives = error_indices[(targets[error_indices] == 1) & (predictions[error_indices] == 0)]
        
        # calculate TP and TN correctly using confusion matrix logic
        true_positives = np.sum((targets == 1) & (predictions == 1))
        true_negatives = np.sum((targets == 0) & (predictions == 0))
        
        print(f"True Positives (TP): {true_positives}")
        print(f"True Negatives (TN): {true_negatives}")
        print(f"False Positives (FP): {len(false_positives)}")
        print(f"False Negatives (FN): {len(false_negatives)}")
        
        # analyze error confidence
        error_confidences = np.max(probabilities[error_indices], axis=1)
        #print(f"Mean confidence of errors: {np.mean(error_confidences):.4f}")
        #print(f"Confidence std of errors: {np.std(error_confidences):.4f}")
        
        # visualize error analysis
        if save_dir:
            self._plot_error_analysis(error_indices, false_positives, false_negatives, 
                                    error_confidences, save_dir)
        
        return {
            'total_errors': len(error_indices),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'error_confidence_mean': float(np.mean(error_confidences)),
            'error_confidence_std': float(np.std(error_confidences)),
            'error_sample_names': list(error_sample_names)
        }
    
    def _plot_error_analysis(self, error_indices, false_positives, false_negatives, 
                           error_confidences, save_dir):
        """Plot error analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # plot error type distribution
        error_types = ['False Positives', 'False Negatives']
        error_counts = [len(false_positives), len(false_negatives)]
        axes[0, 0].bar(error_types, error_counts, color=['red', 'orange'])
        axes[0, 0].set_title('Error Type Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # plot error confidence distribution
        axes[0, 1].hist(error_confidences, bins=20, alpha=0.7, color='red')
        axes[0, 1].set_title('Confidence Distribution of Errors')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        
        # plot confusion matrix
        cm = confusion_matrix(self.results['targets'], self.results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['Non-significant', 'Significant'],
                   yticklabels=['Non-significant', 'Significant'])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # plot ROC curve
        if len(np.unique(self.results['targets'])) > 1:
            fpr, tpr, _ = roc_curve(self.results['targets'], self.results['probabilities'][:, 1])
            auc = roc_auc_score(self.results['targets'], self.results['probabilities'][:, 1])
            axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curve')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_predictions(self, num_samples: int = 10, save_dir: str = None):
        """Visualize predictions"""
        if not self.results['sample_info']:
            print("Please run evaluation first")
            return
        
        # plot random samples with predictions
        sample_indices = np.random.choice(len(self.results['sample_info']), 
                                        min(num_samples, len(self.results['sample_info'])), 
                                        replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            sample_info = self.results['sample_info'][sample_idx]
            pd_points = sample_info['pd_points']
            num_points = sample_info['num_points']
            sample_name = sample_info.get('sample_name', f'sample_{sample_idx}') 
            
            start_idx = sum(info['num_points'] for info in self.results['sample_info'][:sample_idx])
            end_idx = start_idx + num_points
            
            sample_predictions = self.results['predictions'][start_idx:end_idx]
            sample_targets = self.results['targets'][start_idx:end_idx]
            sample_probabilities = self.results['probabilities'][start_idx:end_idx]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            non_origin_mask = ~((pd_points[:, 0] == 0) & (pd_points[:, 1] == 0))
            filtered_points = pd_points[non_origin_mask]

            ax0 = axes[0]
            if sample_targets is not None:
                filtered_targets = sample_targets[non_origin_mask]
                significant_mask = filtered_targets == 1
                non_significant_mask = filtered_targets == 0
                
                if np.any(non_significant_mask):
                    ax0.scatter(filtered_points[non_significant_mask, 0], filtered_points[non_significant_mask, 1], 
                               c='lightblue', alpha=0.8, s=30, label='Noise')
                
                if np.any(significant_mask):
                    ax0.scatter(filtered_points[significant_mask, 0], filtered_points[significant_mask, 1], 
                               c='red', alpha=1, s=50, label='Significant')
                
                ax0.legend()
            else:
                ax0.scatter(filtered_points[:, 0], filtered_points[:, 1], c='blue', alpha=0.8, s=30)
            
            ax0.set_xlabel('Birth')
            ax0.set_ylabel('Death')
            ax0.set_title(f'{sample_name} - Ground Truth')  
            ax0.grid(True, alpha=0.3)
            
            ax1 = axes[1]
            if sample_predictions is not None:
                filtered_predictions = sample_predictions[non_origin_mask]
                significant_mask = filtered_predictions == 1
                non_significant_mask = filtered_predictions == 0
                
                if np.any(non_significant_mask):
                    ax1.scatter(filtered_points[non_significant_mask, 0], filtered_points[non_significant_mask, 1], 
                               c='blue', alpha=0.8, s=30, label='Noise')
                
                if np.any(significant_mask):
                    ax1.scatter(filtered_points[significant_mask, 0], filtered_points[significant_mask, 1], 
                               c='red', alpha=1, s=30, label='Significant')
                
                ax1.legend()
            else:
                ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], c='blue', alpha=0.6, s=30)
            
            ax1.set_xlabel('Birth')
            ax1.set_ylabel('Death')
            ax1.set_title(f'{sample_name} - Predictions')
            ax1.grid(True, alpha=0.3)
            '''
            filtered_probabilities = sample_probabilities[non_origin_mask]
            confidences = np.max(filtered_probabilities, axis=1)
            scatter = axes[2].scatter(filtered_points[:, 0], filtered_points[:, 1], 
                                    c=confidences, cmap='viridis', s=50)
            axes[2].set_xlabel('Birth')
            axes[2].set_ylabel('Death')
            axes[2].set_title(f'Sample {sample_idx} - Confidence')
            plt.colorbar(scatter, ax=axes[2])
            '''

            for ax in axes:
                max_val = max(pd_points.max(), pd_points[:, 1].max())
                min_val = min(pd_points.min(), pd_points[:, 0].min())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            plt.tight_layout()
            
            if save_dir:
                safe_sample_name = sample_name.replace('/', '_').replace('\\', '_').replace(':', '_')
                plt.savefig(os.path.join(save_dir, f'{safe_sample_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def generate_report(self, save_path: str):
        """Generate evaluation report"""
        if not self.results['metrics']:
            print("Please run evaluation first")
            return
        
        metrics = self.results['metrics']
        
        report = {
            'model_performance': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'roc_auc': metrics.get('roc_auc', 'N/A')
            },
            'detailed_metrics': {
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'f1_macro': metrics['f1_macro']
            },
            'data_statistics': {
                'total_samples': len(self.results['targets']),
                'class_distribution': metrics['class_distribution'],
                'prediction_distribution': metrics['prediction_distribution']
            },
            'confusion_matrix': metrics['confusion_matrix']
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print report summary
        print("\n" + "="*60)
        print("Model Evaluation Report")
        print("="*60)
        print(f"F1 Score: {metrics['f1']:.5f}")
        print(f"Accuracy: {metrics['accuracy']:.5f}")
        print(f"Precision: {metrics['precision']:.5f}")
        print(f"Recall: {metrics['recall']:.5f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.5f}")
        print(f"Total Samples: {len(self.results['targets'])}")
        print(f"Class Distribution: {metrics['class_distribution']}")
        print("="*60)
        
        return report
    
    def predict_single_sample(self, pd_points: np.ndarray, point_cloud: np.ndarray = None, 
                            aux_features: np.ndarray = None) -> dict:
        """Predict a single sample"""
        self.model.eval()
        
        with torch.no_grad():
            batch_size = 1
            max_pd_points = self.config['data']['max_pd_points']
            max_pc_points = self.config['data']['max_pc_points']
            
            if len(pd_points) > max_pd_points:
                pd_points = pd_points[:max_pd_points]
            
            pd_input = np.zeros((batch_size, max_pd_points, 2))
            pd_input[0, :len(pd_points)] = pd_points
            pd_input = torch.FloatTensor(pd_input).to(self.device)

            pc_input = None
            if point_cloud is not None:
                if len(point_cloud) > max_pc_points:
                    indices = np.random.choice(len(point_cloud), max_pc_points, replace=False)
                    point_cloud = point_cloud[indices]
                
                pc_input = np.zeros((batch_size, max_pc_points, point_cloud.shape[1]))
                pc_input[0, :len(point_cloud)] = point_cloud
                pc_input = torch.FloatTensor(pc_input).to(self.device)
            
            aux_input = None
            if aux_features is not None:
                aux_input = torch.FloatTensor(aux_features).unsqueeze(0).to(self.device)

            outputs = self.model(pd_input, pc_input, aux_input)
            probabilities = F.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)

            num_points = len(pd_points)
            result_probs = probabilities[0, :num_points].cpu().numpy()
            result_preds = predictions[0, :num_points].cpu().numpy()
            
            return {
                'predictions': result_preds,
                'probabilities': result_probs,
                'confidence': np.max(result_probs, axis=1),
                'num_significant': np.sum(result_preds == 1),
                'significance_ratio': np.mean(result_preds == 1)
            }


def main():
    parser = argparse.ArgumentParser(description='Test persistence diagram significance detection model')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--data_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--type', type=str, required=True, help='Test type (test_planar, test_CAD, test_porous, test_ZIF)')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (recommended 1 to avoid memory issues)')
    parser.add_argument('--visualize', action='store_true', help='Whether to generate visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of visualization samples')
    
    args = parser.parse_args()
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()

    
    # Validate type parameter
    valid_types = {'test_planar', 'test_CAD', 'test_porous', 'test_ZIF'}
    if args.type not in valid_types:
        raise ValueError("type must be one of test_planar, test_CAD, test_porous, test_ZIF")

    test_dir = os.path.join(args.data_dir,'processed','test')
    src_dir = os.path.join(args.data_dir,'processed',args.type)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    if os.path.exists(src_dir):
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(test_dir, item)
            (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
        
    evaluator = ModelEvaluator(args.model, args.config)
    
    # load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    original_max_pc_points = config['data']['max_pc_points']
    if torch.cuda.is_available():

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 10:  
            print(f"Detected small GPU memory ({gpu_memory_gb:.1f}GB), switching to CPU computation")
            # Force use CPU
            evaluator.device = 'cpu'
            evaluator.model = evaluator.model.cpu()
            print("Switched to CPU computation mode")
    
    test_dataset = PersistenceDataset(
        data_dir=args.data_dir,
        split='test',
        max_pd_points=config['data']['max_pd_points'],
        max_pc_points=config['data']['max_pc_points'],
        normalize=config['data']['normalize'],
        synthetic_ratio=config['data'].get('synthetic_ratio', 0.0)
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # evaluate the model
    data_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,  
        pin_memory=False  
    )
    
    metrics = evaluator.evaluate_dataset(test_dataset, data_loader)
    

    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    report = evaluator.generate_report(report_path)

    error_analysis = evaluator.analyze_errors(args.output_dir)
    
    # save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.pkl')
    save_predictions(evaluator.results, predictions_path)
    
    # visualize predictions
    if args.visualize:
        evaluator.visualize_predictions(args.num_vis_samples, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
        main()