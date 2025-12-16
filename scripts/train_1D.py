"""
Training Script for Persistence Diagram Significance Detection
Training Script for Persistence Diagram Significance Detection

Main Functions:
1. Model training and validation
2. Loss function and optimizer configuration
3. Learning rate scheduling
4. Model saving and loading
5. Training process monitoring and logging
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import PersistenceSignificanceNet, initialize_weights, count_parameters
from dataset import create_data_loaders
from utils import EarlyStopping, MetricTracker, save_checkpoint, load_checkpoint


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Suitable for handling class imbalance between significant and non-significant points
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OriginConstrainedLoss(nn.Module):
    """
    Loss function with origin constraint
    Applies strong constraints on the (0,0) point in the persistence diagram to ensure it is classified as insignificant
    """
    def __init__(self, base_criterion, constraint_weight=10.0):
        super(OriginConstrainedLoss, self).__init__()
        self.base_criterion = base_criterion
        self.constraint_weight = constraint_weight
        
    def forward(self, outputs, targets, pd_points):
        """
        Args:
            outputs: [batch_size, num_points, num_classes] - Model output logits
            targets: [batch_size, num_points] - Ground truth labels
            pd_points: [batch_size, num_points, 2] - Persistence diagram point coordinates
        """
        # Compute base loss
        if isinstance(self.base_criterion, WeightedFocalLoss):
            # [batch_size, num_points, num_classes]
            base_loss = self.base_criterion(outputs, targets)
        else:
            # [batch_size*num_points, num_classes]
            base_loss = self.base_criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        

        tolerance = 1e-10 
        origin_mask = (pd_points[:, :, 0] == 0.0) & (pd_points[:, :, 1] == 0.0)
        
        if not origin_mask.any():
            origin_mask = (torch.abs(pd_points[:, :, 0]) < tolerance) & (torch.abs(pd_points[:, :, 1]) < tolerance)
        
        if origin_mask.any():
            origin_outputs = outputs[origin_mask]  # [num_origin_points, num_classes]
            origin_targets = torch.zeros(origin_outputs.size(0), dtype=torch.long, device=outputs.device)
            
            constraint_loss = F.cross_entropy(origin_outputs, origin_targets)
            
            # total loss
            total_loss = base_loss + self.constraint_weight * constraint_loss
            
            return total_loss
        else:
            return base_loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss that considers sample weights
    """
    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, inputs, targets, sample_weights=None):
        batch_size, seq_len, num_classes = inputs.shape
        inputs = inputs.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
        targets = targets.view(-1)  # [batch_size * seq_len]
        
        # Ensure class_weights is on the same device as inputs
        weight = self.class_weights.to(inputs.device) if self.class_weights is not None else None
        
        # compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if sample_weights is not None:
            sample_weights = sample_weights.view(-1)
            focal_loss = focal_loss * sample_weights
        
        return focal_loss.mean()


class Trainer:
    """Trainer class"""
    
    def __init__(self, config):
        self.config = config
        
        # Intelligent device selection: Check GPU memory and decide to use GPU or CPU
        self.device = self._select_device()
        print(f"Using device: {self.device}")
        
        # Set random seed
        self._set_seed(config['seed'])
        
        # Create output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._build_data_loaders()
        
        # Initialize loss function and optimizer
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Initialize training tools
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            min_delta=config['training']['early_stopping_min_delta'],
            mode='max'  # Higher F1 score is better
        )
        self.metric_tracker = MetricTracker()
        
        # Initialize logger
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
        
        # Training status
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0
        self.best_train_f1 = 0.0
        self.patience_counter = 0
        # Lists to record training and validation losses
        self.train_losses = []
        self.val_losses = []
        
    def _select_device(self):
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU training")
            return torch.device('cpu')
        
        try:
            # Get GPU memory information
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            #print(f"Total GPU memory: {gpu_memory:.2f} GB")
            
            # Check current GPU memory usage
            torch.cuda.empty_cache()  # Clear cache
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            free_memory = gpu_memory - allocated_memory - cached_memory
            
            #print(f"Allocated GPU memory: {allocated_memory:.2f} GB")
            #print(f"Cached GPU memory: {cached_memory:.2f} GB") 
            #print(f"Free GPU memory: {free_memory:.2f} GB")
            
            # If free memory is less than 2GB, switch to CPU
            if free_memory < 2.0:
                print(f"Insufficient GPU memory ({free_memory:.2f} GB < 2.0 GB), switching to CPU training")
                return torch.device('cpu')
            else:
                print(f"GPU memory sufficient, using GPU for training")
                return torch.device('cuda')
                
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            print("Switching to CPU training")
            return torch.device('cpu')
        
    def _set_seed(self, seed):
        """Set random seed"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _build_model(self):
        """Build model"""
        model_config = self.config['model']
        model = PersistenceSignificanceNet(
            pd_input_dim=model_config['pd_input_dim'],
            pc_input_dim=model_config['pc_input_dim'],
            aux_dim=model_config['aux_dim'],
            hidden_dim=model_config['hidden_dim'],
            fusion_dim=model_config['fusion_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        # Initialize weights
        initialize_weights(model)
        
        print(f"Number of model parameters: {count_parameters(model):,}")
        return model
        
    def _build_data_loaders(self):
        """Build data loaders"""
        data_config = self.config['data']
        return create_data_loaders(
            data_dir=data_config['data_dir'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            max_pd_points=data_config['max_pd_points'],
            max_pc_points=data_config['max_pc_points'],
            normalize=data_config['normalize'],
            synthetic_ratio=data_config['synthetic_ratio']
        )
        
    def _build_criterion(self):
        """Build loss function"""
        loss_config = self.config['loss']
        
        # Build base loss function
        base_criterion = None
        if loss_config['type'] == 'cross_entropy':
            class_weights = None
            if loss_config['class_weights']:
                class_weights = torch.FloatTensor(loss_config['class_weights']).to(self.device)
            base_criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        elif loss_config['type'] == 'focal':
            base_criterion = FocalLoss(
                alpha=loss_config['focal_alpha'],
                gamma=loss_config['focal_gamma']
            )
            
        elif loss_config['type'] == 'weighted_focal':
            class_weights = None
            if loss_config['class_weights']:
                class_weights = torch.FloatTensor(loss_config['class_weights']).to(self.device)
            base_criterion = WeightedFocalLoss(
                alpha=loss_config['focal_alpha'],
                gamma=loss_config['focal_gamma'],
                class_weights=class_weights
            )
        else:
            raise ValueError(f"Unknown loss function type: {loss_config['type']}")
        
        # Check if origin constraint is enabled
        if loss_config.get('origin_constraint', False):
            #print('Enabling origin constraint')
            #constraint_weight = loss_config.get('origin_constraint_weight', 10.0)
            #return OriginConstrainedLoss(base_criterion, constraint_weight)
            return base_criterion
        else:
            return base_criterion
            
    def _build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
            
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif sched_config['type'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config['factor'],
                patience=sched_config['patience']
            )
        else:
            return None
            
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                pd_points = batch['pd_points'].to(self.device)
                point_cloud = batch['point_cloud'].to(self.device)
                significance_labels = batch['significance_labels'].to(self.device)
                num_pd_points = batch['num_pd_points'].to(self.device)
                
                aux_features = None
                if 'aux_features' in batch:
                    aux_features = batch['aux_features'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(pd_points, point_cloud, aux_features)
                
                # Compute loss (only for valid PD points)
                loss = self._compute_loss(outputs, significance_labels, num_pd_points, pd_points)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                batch_metrics = self._compute_metrics(outputs, significance_labels, num_pd_points)
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
                
                # Print progress
                #if batch_idx % self.config['training']['log_interval'] == 0:
                #    print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                #          f'Loss: {loss.item():.4f}, Acc: {batch_metrics["accuracy"]:.4f}')
                    
            except torch.cuda.OutOfMemoryError:
                print(f"GPU out of memory, switching to CPU for training batch {batch_idx}")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Switch to CPU
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
                
                # reprocess batch on GPU
                pd_points = batch['pd_points'].to(self.device)
                point_cloud = batch['point_cloud'].to(self.device)
                significance_labels = batch['significance_labels'].to(self.device)
                num_pd_points = batch['num_pd_points'].to(self.device)
                
                aux_features = None
                if 'aux_features' in batch:
                    aux_features = batch['aux_features'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(pd_points, point_cloud, aux_features)
                
                # Compute loss (only for valid PD points)
                loss = self._compute_loss(outputs, significance_labels, num_pd_points, pd_points)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                batch_metrics = self._compute_metrics(outputs, significance_labels, num_pd_points)
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
                
                print(f"Successfully processed batch {batch_idx} on CPU")
        
        # Average metrics
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_loss, epoch_metrics
        
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    pd_points = batch['pd_points'].to(self.device)
                    point_cloud = batch['point_cloud'].to(self.device)
                    significance_labels = batch['significance_labels'].to(self.device)
                    num_pd_points = batch['num_pd_points'].to(self.device)
                    
                    aux_features = None
                    if 'aux_features' in batch:
                        aux_features = batch['aux_features'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(pd_points, point_cloud, aux_features)
                    
                    # Compute loss (only for valid PD points)
                    loss = self._compute_loss(outputs, significance_labels, num_pd_points, pd_points)
                    
                    # Statistics
                    epoch_loss += loss.item()
                    batch_metrics = self._compute_metrics(outputs, significance_labels, num_pd_points)
                    for key in epoch_metrics:
                        epoch_metrics[key] += batch_metrics[key]
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"GPU out of memory, switching to CPU for validation batch {batch_idx}")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Switch to CPU temporarily
                    original_device = self.device
                    self.device = torch.device('cpu')
                    self.model = self.model.to(self.device)
                    
                    # Move data to CPU
                    pd_points = batch['pd_points'].to(self.device)
                    point_cloud = batch['point_cloud'].to(self.device)
                    significance_labels = batch['significance_labels'].to(self.device)
                    num_pd_points = batch['num_pd_points'].to(self.device)
                    
                    aux_features = None
                    if 'aux_features' in batch:
                        aux_features = batch['aux_features'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(pd_points, point_cloud, aux_features)
                    
                    # Compute loss (only for valid PD points)
                    loss = self._compute_loss(outputs, significance_labels, num_pd_points, pd_points)
                    
                    # Statistics
                    epoch_loss += loss.item()
                    batch_metrics = self._compute_metrics(outputs, significance_labels, num_pd_points)
                    for key in epoch_metrics:
                        epoch_metrics[key] += batch_metrics[key]
                    
                    # Switch back to original device
                    self.device = original_device
                    self.model = self.model.to(self.device)
        
        # Average metrics
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_loss, epoch_metrics
        
    def _compute_loss(self, outputs, targets, num_points, pd_points=None):
        """Compute loss, only for valid points"""
        batch_size, max_points, num_classes = outputs.shape
        
        # Create mask, only for valid points
        mask = torch.zeros(batch_size, max_points, dtype=torch.bool, device=self.device)
        for i, n in enumerate(num_points):
            mask[i, :n] = True
        
        # Select only valid points
        valid_outputs = outputs[mask]  # [num_valid_points, num_classes]
        valid_targets = targets[mask]  # [num_valid_points]
        
        if len(valid_outputs) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute loss
        if isinstance(self.criterion, OriginConstrainedLoss):
            # OriginConstrainedLoss requires persistence diagram point coordinate information
            if pd_points is None:
                raise ValueError("OriginConstrainedLoss requires persistence diagram point coordinate information")
            return self.criterion(outputs, targets, pd_points)
        elif isinstance(self.criterion, WeightedFocalLoss):
            valid_outputs = valid_outputs.unsqueeze(0)  # [1, num_valid_points, num_classes]
            valid_targets = valid_targets.unsqueeze(0)  # [1, num_valid_points]
            return self.criterion(valid_outputs, valid_targets)
        else:
            return self.criterion(valid_outputs, valid_targets)
        
    def _compute_metrics(self, outputs, targets, num_points):
        """Compute evaluation metrics"""
        batch_size, max_points, num_classes = outputs.shape
        
        # Create mask, only for valid points
        mask = torch.zeros(batch_size, max_points, dtype=torch.bool, device=self.device)
        for i, n in enumerate(num_points):
            mask[i, :n] = True
        
        # Select only valid points
        valid_outputs = outputs[mask]  # [num_valid_points, num_classes]
        valid_targets = targets[mask]  # [num_valid_points]
        
        if len(valid_outputs) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Predictions
        predictions = torch.argmax(valid_outputs, dim=1)
        
        # Compute metrics
        correct = (predictions == valid_targets).float()
        accuracy = correct.mean().item()
        
        # Compute precision, recall, f1 (only for significant class, i.e., class 1)
        tp = ((predictions == 1) & (valid_targets == 1)).float().sum().item()
        fp = ((predictions == 1) & (valid_targets == 0)).float().sum().item()
        fn = ((predictions == 0) & (valid_targets == 1)).float().sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            train_loss, train_metrics = self.train_epoch()
            
            val_loss, val_metrics = self.validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            epoch_time = time.time() - start_time
            self._log_epoch(epoch, train_loss, train_metrics, val_loss, val_metrics, epoch_time)
            
            # Save best model
            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint('best_model_1D.pth', is_best=True)
            else:
                self.patience_counter += 1
            
            # Save checkpoint periodically
            #if (epoch + 1) % self.config['training']['save_interval'] == 0:
            #    self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping check
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                break
        
        
        # Plot loss curve
        self._plot_loss_curve()
        
        print("Training completed!")
        self.writer.close()
        
    def _log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, epoch_time):
        """Log epoch information"""
        print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
        
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            self.writer.add_scalar(f'Metrics/Train_{metric}', train_metrics[metric], epoch)
            self.writer.add_scalar(f'Metrics/Val_{metric}', val_metrics[metric], epoch)
        
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
    def _save_checkpoint(self, filename, is_best=False, is_last=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }
        
        filepath = os.path.join(self.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"----------------Saving best model: {filepath}----------------")
        elif is_last:
            print(f"Saving last epoch model: {filepath}")
            
    def _plot_loss_curve(self):
        """Plot training and validation loss curves and save"""
        # Create result folder if it doesn't exist
        result_dir = os.path.join(self.output_dir, 'train_loss_result')
        os.makedirs(result_dir, exist_ok=True)
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='train Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='val Loss')
        #plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        save_path = os.path.join(result_dir, 'loss_curve_1D.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Loss curve saved to: {save_path}")


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train persistence diagram significance detection model')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume training')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    trainer = Trainer(config)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.best_val_f1 = checkpoint['best_val_f1']
        print(f"Resuming training from epoch {trainer.current_epoch}")
    
    trainer.train()


if __name__ == "__main__":
    main()