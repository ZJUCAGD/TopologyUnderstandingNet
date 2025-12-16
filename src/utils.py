"""
Utility Functions Module
Utility Functions for Persistence Diagram Significance Detection

Contains various utility functions needed during training:
1. Early stopping mechanism
2. Metric tracking
3. Model saving and loading
4. Data visualization
5. Other helper functions
"""

import os
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any


class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to tolerate
            min_delta: Minimum improvement threshold
            mode: 'min' for monitoring metrics that should decrease, 'max' for metrics that should increase
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class MetricTracker:
    """Metric tracker"""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        """Get average of metrics"""
        if key not in self.metrics:
            return 0.0
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        return np.mean(values)
    
    def get_latest(self, key: str) -> float:
        """Get latest value of metrics"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.metrics[key][-1]
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
    
    def save(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        """Load metrics from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.metrics = json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save training checkpoint"""
    import time
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded: {filepath}, Epoch: {epoch}")
    return epoch, metrics


def compute_class_weights(labels: List[np.ndarray], num_classes: int = 2) -> np.ndarray:
    """
    Compute class weights to handle class imbalance
    
    Args:
        labels: List of labels, each element is a label array for a sample
        num_classes: Number of classes
        
    Returns:
        class_weights: Class weight array
    """
    class_counts = np.zeros(num_classes)
    total_samples = 0
    
    for label_array in labels:
        unique, counts = np.unique(label_array, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                class_counts[cls] += count
                total_samples += count
    
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights


def visualize_persistence_diagram(pd_points: np.ndarray, significance_labels: np.ndarray = None, 
                                title: str = "Persistence Diagram", save_path: str = None):
    """
    Visualize persistence diagram
    
    Args:
        pd_points: Persistence diagram points [N, 2] (birth, death)
        significance_labels: Significance labels [N]
        title: Plot title
        save_path: Save path
    """
    plt.figure(figsize=(8, 8))
    
    tolerance = 1e-10
    non_origin_mask = ~((np.abs(pd_points[:, 0]) < tolerance) & (np.abs(pd_points[:, 1]) < tolerance))
    filtered_points = pd_points[non_origin_mask]
    filtered_labels = significance_labels[non_origin_mask] if significance_labels is not None else None

    if filtered_labels is not None:
        # color points by significance
        significant_mask = filtered_labels == 1
        non_significant_mask = filtered_labels == 0
        
        if np.any(non_significant_mask):
            plt.scatter(filtered_points[non_significant_mask, 0], filtered_points[non_significant_mask, 1], 
                       c='lightblue', alpha=0.6, s=30, label='Non-significant')
        
        if np.any(significant_mask):
            plt.scatter(filtered_points[significant_mask, 0], filtered_points[significant_mask, 1], 
                       c='red', alpha=0.8, s=50, label='Significant')
        
        plt.legend()
    else:
        plt.scatter(filtered_points[:, 0], filtered_points[:, 1], c='blue', alpha=0.6, s=30)
    
    max_val = max(pd_points.max(), pd_points[:, 1].max())
    min_val = min(pd_points.min(), pd_points[:, 0].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Diagonal')
    
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]],
                        save_path: str = None):
    """
    Plot training curves
    
    Args:
        train_losses: Training loss list
        val_losses: Validation loss list
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        save_path: Save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # accuracy curves
    if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
        axes[0, 1].plot(train_metrics['accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(val_metrics['accuracy'], label='Val Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # F1 score curves
    if 'f1' in train_metrics and 'f1' in val_metrics:
        axes[1, 0].plot(train_metrics['f1'], label='Train F1', color='blue')
        axes[1, 0].plot(val_metrics['f1'], label='Val F1', color='red')
        axes[1, 0].set_title('F1 Score Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # precision and recall curves
    if 'precision' in train_metrics and 'recall' in train_metrics:
        axes[1, 1].plot(train_metrics['precision'], label='Train Precision', color='green')
        axes[1, 1].plot(train_metrics['recall'], label='Train Recall', color='orange')
        if 'precision' in val_metrics and 'recall' in val_metrics:
            axes[1, 1].plot(val_metrics['precision'], label='Val Precision', color='green', linestyle='--')
            axes[1, 1].plot(val_metrics['recall'], label='Val Recall', color='orange', linestyle='--')
        axes[1, 1].set_title('Precision & Recall Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Save path
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or ['Non-significant', 'Significant'],
                yticklabels=class_names or ['Non-significant', 'Significant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_persistence_statistics(pd_points: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of persistence diagram
    
    Args:
        pd_points: Persistence diagram points [N, 2] (birth, death)
        
    Returns:
        stats: Statistics dictionary
    """
    if len(pd_points) == 0:
        return {
            'num_points': 0,
            'mean_birth': 0.0,
            'mean_death': 0.0,
            'mean_persistence': 0.0,
            'max_persistence': 0.0,
            'persistence_std': 0.0
        }
    
    births = pd_points[:, 0]
    deaths = pd_points[:, 1]
    persistences = deaths - births
    
    stats = {
        'num_points': len(pd_points),
        'mean_birth': float(np.mean(births)),
        'mean_death': float(np.mean(deaths)),
        'mean_persistence': float(np.mean(persistences)),
        'max_persistence': float(np.max(persistences)),
        'persistence_std': float(np.std(persistences)),
        'birth_range': float(np.max(births) - np.min(births)),
        'death_range': float(np.max(deaths) - np.min(deaths))
    }
    
    return stats


def normalize_persistence_diagram(pd_points: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict]:
    """
    Normalize persistence diagram
    
    Args:
        pd_points: Persistence diagram points [N, 2]
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        normalized_points: Normalized points
        norm_params: Normalization parameters
    """
    if len(pd_points) == 0:
        return pd_points, {}
    
    if method == 'standard':
        mean = np.mean(pd_points, axis=0)
        std = np.std(pd_points, axis=0)
        std[std == 0] = 1.0 
        normalized_points = (pd_points - mean) / std
        norm_params = {'mean': mean, 'std': std, 'method': method}
        
    elif method == 'minmax':
        min_vals = np.min(pd_points, axis=0)
        max_vals = np.max(pd_points, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0 
        normalized_points = (pd_points - min_vals) / range_vals
        norm_params = {'min': min_vals, 'max': max_vals, 'method': method}
        
    elif method == 'robust':
        median = np.median(pd_points, axis=0)
        mad = np.median(np.abs(pd_points - median), axis=0)
        mad[mad == 0] = 1.0
        normalized_points = (pd_points - median) / mad
        norm_params = {'median': median, 'mad': mad, 'method': method}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_points, norm_params


def denormalize_persistence_diagram(normalized_points: np.ndarray, norm_params: Dict) -> np.ndarray:
    """
    Denormalize persistence diagram
    
    Args:
        normalized_points: Normalized points
        norm_params: Normalization parameters
        
    Returns:
        original_points: Points in original scale
    """
    if len(normalized_points) == 0:
        return normalized_points
    
    method = norm_params['method']
    
    if method == 'standard':
        return normalized_points * norm_params['std'] + norm_params['mean']
    elif method == 'minmax':
        return normalized_points * (norm_params['max'] - norm_params['min']) + norm_params['min']
    elif method == 'robust':
        return normalized_points * norm_params['mad'] + norm_params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def save_predictions(predictions: Dict, filepath: str):
    """Save predictions"""
    with open(filepath, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved: {filepath}")


def load_predictions(filepath: str) -> Dict:
    """Load predictions"""
    with open(filepath, 'rb') as f:
        predictions = pickle.load(f)
    print(f"Predictions loaded: {filepath}")
    return predictions


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    Create experiment directory
    
    Args:
        base_dir: Base directory
        experiment_name: Experiment name, if None use timestamp
        
    Returns:
        experiment_dir: Experiment directory path
    """
    if experiment_name is None:
        from datetime import datetime
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'results'), exist_ok=True)
    
    return experiment_dir


def log_model_info(model, logger=None):
    """Log model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = f"""
Model Information:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)
"""
    
    if logger:
        logger.info(info)
    else:
        print(info)


def set_random_seed(seed: int):
    """Set random seed"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data augmentation functions
def augment_persistence_diagram(pd_points: np.ndarray, noise_std: float = 0.01, 
                               rotation_angle: float = 0.0) -> np.ndarray:
    """
    Persistence diagram data augmentation
    
    Args:
        pd_points: Persistence diagram points [N, 2]
        noise_std: Noise standard deviation
        rotation_angle: Rotation angle (radians)
        
    Returns:
        augmented_points: Augmented points
    """
    augmented = pd_points.copy()
    
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented += noise
    
    if rotation_angle != 0:
        center = np.mean(augmented, axis=0)
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        centered = augmented - center
        rotated = centered @ rotation_matrix.T
        augmented = rotated + center
    
    augmented[:, 1] = np.maximum(augmented[:, 1], augmented[:, 0])
    
    return augmented



def setup_logging(level='INFO'):
    """
    Set up logging
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger('PersistenceTraining')
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger
