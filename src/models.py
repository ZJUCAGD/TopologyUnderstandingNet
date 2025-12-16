"""
Persistence Diagram Significance Detection Neural Networks
Persistence Diagram Significance Detection Neural Networks

Mainly includes:
1. PersistenceSignificanceNet - Main multimodal fusion network
2. PersistenceDiagramEncoder - Persistence diagram encoder
3. PointCloudEncoder - Point cloud encoder  
4. MultiModalFusion - Multimodal feature fusion layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class PersistenceDiagramEncoder(nn.Module):
    """
    Persistence Diagram Encoder
    Encodes points (birth, death) in persistence diagram into feature vectors
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=128):
        super(PersistenceDiagramEncoder, self).__init__()
        
        self.point_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 4D: birth, death, persistence, birth_death_ratio
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, pd_points):
        """
        Args:
            pd_points: [batch_size, num_points, 2] - (birth, death) coordinates
        Returns:
            point_features: [batch_size, num_points, output_dim] - features for each point
            global_features: [batch_size, output_dim] - global features
        """
        batch_size, num_points, _ = pd_points.shape
        
        # Extract birth, death, persistence, birth_death_ratio
        birth = pd_points[:, :, 0:1]  # [batch_size, num_points, 1]
        death = pd_points[:, :, 1:2]  # [batch_size, num_points, 1]
        persistence = death - birth   # [batch_size, num_points, 1]
        
        max_finite_value = 100.0
        persistence = torch.clamp(persistence, min=0.0, max=max_finite_value)
        birth = torch.clamp(birth, min=0.0, max=max_finite_value)
        death = torch.clamp(death, min=0.0, max=max_finite_value)
        
        epsilon = 1e-6
        birth_safe = torch.clamp(birth, min=epsilon)
        birth_death_ratio = torch.log(death / birth_safe)
        
        # Combine all features
        enhanced_features = torch.cat([
            birth, death, persistence, birth_death_ratio
        ], dim=-1)  # [batch_size, num_points, 4]
        
        enhanced_features = torch.nan_to_num(enhanced_features, nan=0.0, posinf=max_finite_value, neginf=0.0)
        
        reshaped_features = enhanced_features.view(-1, enhanced_features.size(-1))
        
        # Point-level feature extraction
        if reshaped_features.size(0) == 1 and self.training:
            self.point_mlp.eval()
            with torch.no_grad():
                point_features = self.point_mlp(reshaped_features)  # [batch_size * num_points, output_dim]
            self.point_mlp.train()
        else:
            point_features = self.point_mlp(reshaped_features)  # [batch_size * num_points, output_dim]
        point_features = point_features.view(batch_size, num_points, -1)  # [batch_size, num_points, output_dim]
        
        # Self-attention mechanism
        attended_features, _ = self.attention(point_features, point_features, point_features)
        
        # Global feature aggregation
        global_features = torch.mean(attended_features, dim=1)  # [batch_size, output_dim]
        
        return attended_features, global_features


class PointCloudEncoder(nn.Module):
    """
    Point Cloud Encoder (similar to PointNet architecture)
    Processes raw point cloud data and extracts geometric features
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
        super(PointCloudEncoder, self).__init__()
        
        # Point-level feature extraction
        self.point_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim)
        )
        
        # Global feature extraction
        self.global_conv = nn.Sequential(
            nn.Conv1d(output_dim, output_dim * 2, 1),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Conv1d(output_dim * 2, output_dim, 1),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: [batch_size, num_points, input_dim] - raw point cloud
        Returns:
            point_features: [batch_size, num_points, output_dim] - features for each point
            global_features: [batch_size, output_dim] - global features
        """
        x = point_cloud.transpose(1, 2)  # [batch_size, input_dim, num_points]
        
        # Point-level feature extraction
        if x.size(0) == 1 and self.training:
            self.point_conv.eval()
            with torch.no_grad():
                point_features = self.point_conv(x)  # [batch_size, output_dim, num_points]
            self.point_conv.train()
        else:
            point_features = self.point_conv(x)  # [batch_size, output_dim, num_points]
        
        # Global feature extraction
        if point_features.size(0) == 1 and self.training:
            self.global_conv.eval()
            with torch.no_grad():
                global_features = self.global_conv(point_features)  # [batch_size, output_dim, num_points]
            self.global_conv.train()
        else:
            global_features = self.global_conv(point_features)  # [batch_size, output_dim, num_points]
        global_features = torch.max(global_features, dim=2)[0]  # [batch_size, output_dim]
        
        # [batch_size, num_points, output_dim]
        point_features = point_features.transpose(1, 2)
        
        return point_features, global_features


class MultiModalFusion(nn.Module):
    """
    Multimodal Feature Fusion Layer
    Fuses persistence diagram features, point cloud features, and other auxiliary features
    """
    def __init__(self, pd_dim=128, pc_dim=128, aux_dim=0, fusion_dim=256):
        super(MultiModalFusion, self).__init__()
        
        self.pd_dim = pd_dim
        self.pc_dim = pc_dim
        self.aux_dim = aux_dim
        self.fusion_dim = fusion_dim
        
        self.pd_proj = nn.Linear(pd_dim, fusion_dim // 2)
        self.pc_proj = nn.Linear(pc_dim, fusion_dim // 2)
        
        if aux_dim > 0:
            self.aux_proj = nn.Linear(aux_dim, fusion_dim // 2)
            total_dim = fusion_dim + fusion_dim // 2 
        else:
            total_dim = fusion_dim  # 256
            
        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),  # 320 -> 256 (when aux_features are provided)
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim)
        )
        
    def forward(self, pd_features, pc_features, aux_features=None):
        """
        Args:
            pd_features: [batch_size, pd_dim] - persistence diagram global features
            pc_features: [batch_size, pc_dim] - point cloud global features  
            aux_features: [batch_size, aux_dim] - auxiliary features (optional)
        Returns:
            fused_features: [batch_size, fusion_dim] - fused features
        """
        # Project persistence diagram and point cloud features
        pd_proj = self.pd_proj(pd_features)  # [batch_size, fusion_dim//2]
        pc_proj = self.pc_proj(pc_features)  # [batch_size, fusion_dim//2]
        
        # Concatenate projected features
        fused = torch.cat([pd_proj, pc_proj], dim=1)  # [batch_size, fusion_dim]
        
        # If auxiliary features are provided, project and concatenate
        if aux_features is not None and self.aux_dim > 0:
            aux_proj = self.aux_proj(aux_features)  # [batch_size, fusion_dim//4]
            fused = torch.cat([fused, aux_proj], dim=1)  # [batch_size, fusion_dim + fusion_dim//4]

        if fused.size(0) == 1 and self.training:
            self.fusion_net.eval()
            with torch.no_grad():
                fused_features = self.fusion_net(fused)
            self.fusion_net.train()
        else:
            fused_features = self.fusion_net(fused)
        
        return fused_features


class PersistenceSignificanceNet(nn.Module):
    """
    Persistence Diagram Significance Detection Network
    Main multimodal fusion network for predicting significance of each point in persistence diagram
    """
    def __init__(self, 
                 pd_input_dim=2,
                 pc_input_dim=3, 
                 aux_dim=0,
                 hidden_dim=128,
                 fusion_dim=256,
                 num_classes=2):
        super(PersistenceSignificanceNet, self).__init__()
        
        # Encoders for Persistence Diagram and Point Cloud
        self.pd_encoder = PersistenceDiagramEncoder(
            input_dim=pd_input_dim, 
            output_dim=hidden_dim
        )
        self.pc_encoder = PointCloudEncoder(
            input_dim=pc_input_dim,
            output_dim=hidden_dim
        )
        
        # Multi-Modal Fusion
        self.fusion = MultiModalFusion(
            pd_dim=hidden_dim,
            pc_dim=hidden_dim, 
            aux_dim=aux_dim,
            fusion_dim=fusion_dim
        )
        
        # Point-Level Classifier - Predict Significance for Each PD Point
        self.point_classifier = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
    def forward(self, pd_points, point_cloud, aux_features=None):
        """
        Args:
            pd_points: [batch_size, num_pd_points, 2] - persistence diagram point coordinates
            point_cloud: [batch_size, num_pc_points, 3] - raw point cloud
            aux_features: [batch_size, aux_dim] - auxiliary features (optional)
        Returns:
            significance_logits: [batch_size, num_pd_points, num_classes] - significance prediction for each persistence diagram point
        """
        batch_size, num_pd_points, _ = pd_points.shape
        
        # Encode Persistence Diagram and Point Cloud
        pd_point_features, pd_global_features = self.pd_encoder(pd_points)
        pc_point_features, pc_global_features = self.pc_encoder(point_cloud)
        
        # Multi-Modal Fusion
        fused_global_features = self.fusion(pd_global_features, pc_global_features, aux_features)
        
        # Broadcast Fused Global Features to All PD Points
        fused_global_expanded = fused_global_features.unsqueeze(1).expand(-1, num_pd_points, -1)
        
        # Combine Point Features with Fused Global Features
        combined_features = torch.cat([pd_point_features, fused_global_expanded], dim=-1)
        combined_features = combined_features.view(-1, combined_features.size(-1))
        
        # Predict Significance
        if combined_features.size(0) == 1 and self.training:
            self.point_classifier.eval()
            with torch.no_grad():
                significance_logits = self.point_classifier(combined_features)
            self.point_classifier.train()
        else:
            significance_logits = self.point_classifier(combined_features)
        
        significance_logits = significance_logits.view(batch_size, num_pd_points, -1)
        
        return significance_logits
    
    def predict_significance(self, pd_points, point_cloud, aux_features=None, threshold=0.5):
        """
        Predict significant points (inference mode)
        
        Returns:
            predictions: [batch_size, num_pd_points] - binary predictions (0: insignificant, 1: significant)
            probabilities: [batch_size, num_pd_points] - significance probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(pd_points, point_cloud, aux_features)
            probabilities = F.softmax(logits, dim=-1)[:, :, 1]  # 取显著类的概率
            predictions = (probabilities > threshold).long()
            
        return predictions, probabilities


# Auxiliary Functions
def count_parameters(model):
    """Count the number of model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PersistenceSignificanceNet(
        pd_input_dim=2,
        pc_input_dim=3,
        aux_dim=10,  # Assume 10-dimensional auxiliary features
        hidden_dim=128,
        fusion_dim=256,
        num_classes=2
    ).to(device)
    
    # Initialize weights
    initialize_weights(model)
    
    print(f"Number of model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    num_pd_points = 20
    num_pc_points = 1000
    
    # Simulate data
    pd_points = torch.randn(batch_size, num_pd_points, 2).to(device)
    point_cloud = torch.randn(batch_size, num_pc_points, 3).to(device)
    aux_features = torch.randn(batch_size, 10).to(device)
    
    # Forward pass
    output = model(pd_points, point_cloud, aux_features)
    print(f"Output shape: {output.shape}")
    
    # Test prediction function
    predictions, probabilities = model.predict_significance(pd_points, point_cloud, aux_features)
    print(f"Prediction shape: {predictions.shape}")

    print(f"Probability shape: {probabilities.shape}")

