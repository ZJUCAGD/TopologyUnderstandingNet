"""
Custom data conversion script
Convert txt format point cloud and persistence diagram data to .pkl format required for training
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_point_cloud_from_txt(file_path: str) -> np.ndarray:
    """
    Load point cloud data from txt file
    
    Args:
        file_path: Point cloud file path
        
    Returns:
        point_cloud: Point cloud data [n_points, 3]
        
    Supported formats:
    - One point per line, coordinates separated by spaces or tabs
    - Example: x1 y1 z1
           x2 y2 z2
    """
    try:
        point_cloud = np.loadtxt(file_path)
        
        # ensure 3D point cloud
        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(1, -1)
        
        if point_cloud.shape[1] == 2:
            # Add z=0 dimension for 2D points
            z_coords = np.zeros((point_cloud.shape[0], 1))
            point_cloud = np.hstack([point_cloud, z_coords])
        elif point_cloud.shape[1] != 3:
            raise ValueError(f"Point cloud data should be 2D or 3D, but got {point_cloud.shape[1]} dimensions")
            
        #print(f"Successfully loaded point cloud: {point_cloud.shape[0]} points, {point_cloud.shape[1]} dimensions")
        return point_cloud.astype(np.float64)
        
    except Exception as e:
        raise ValueError(f"Failed to load point cloud file {file_path}: {e}")

def load_persistence_diagram_from_txt(file_path: str) -> np.ndarray:
    """
    Load persistence diagram data from txt file
    
    Args:
        file_path: Persistence diagram file path
        
    Returns:
        pd_points: Persistence diagram points [n_points, 2] (birth, death)
        
    Supported formats:
    - One persistence diagram point per line, birth and death separated by spaces or tabs
    - Example: birth1 death1
           birth2 death2
    """
    try:
        pd_points = np.loadtxt(file_path)
        
        # ensure 2D array
        if pd_points.ndim == 1:
            pd_points = pd_points.reshape(1, -1)
            
        if pd_points.shape[1] != 2:
            raise ValueError(f"Persistence diagram data should be 2 columns (birth, death), but got {pd_points.shape[1]} columns")
            
        # ensure death >= birth
        invalid_points = pd_points[:, 1] < pd_points[:, 0]
        if np.any(invalid_points):
            print(f"Warning: Found {np.sum(invalid_points)} invalid points (death < birth), will automatically correct")
            pd_points[invalid_points, 1] = pd_points[invalid_points, 0]
            
        #print(f"Successfully loaded persistence diagram: {pd_points.shape[0]} points")
        return pd_points.astype(np.float64)
        
    except Exception as e:
        raise ValueError(f"Failed to load persistence diagram file {file_path}: {e}")

def estimate_pointcloud_noise_level(point_cloud: np.ndarray, k: int = 10) -> dict:
    """
    Estimate noise level of point cloud

    Args:
        point_cloud (np.ndarray): Nx3 point cloud
        k (int): Number of neighbors for KNN

    Returns:
        dict: Dictionary containing various noise estimation metrics
    """
    if len(point_cloud) < k:
        return {
            'knn_distance_std': 0.0,
            'pca_eigenvalue_ratio': 0.0,
            'density_variation': 0.0
        }

    # 1. KNN distance standard deviation
    try:
        neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_cloud)
        distances, _ = neighbors.kneighbors(point_cloud)
        mean_distances = distances.mean(axis=1)
        knn_distance_std = np.std(mean_distances)
    except Exception:
        knn_distance_std = 0.0

    # 2. Local PCA eigenvalue ratio
    try:
        pca = PCA(n_components=3)
        pca.fit(point_cloud)
        eigenvalues = pca.explained_variance_
        # min_eigenvalue / max_eigenvalue
        pca_eigenvalue_ratio = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0.0
    except Exception:
        pca_eigenvalue_ratio = 0.0

    # 3. Density variation
    try:
        density_proxy = 1.0 / (mean_distances + 1e-6)
        density_variation = np.std(density_proxy) / np.mean(density_proxy) if np.mean(density_proxy) > 0 else 0.0
    except Exception:
        density_variation = 0.0

    return {
        'knn_distance_std': knn_distance_std,
        'pca_eigenvalue_ratio': pca_eigenvalue_ratio,
        'density_variation': density_variation
    }

def compute_roundbox_of_pointcloud(point_cloud: np.ndarray) -> dict:
    """
    Compute round bounding box of point cloud

    Args:
        point_cloud (np.ndarray): Nx3 point cloud

    Returns:
        dict: Dictionary containing bounding box dimensions
    """
    if point_cloud is None or len(point_cloud) == 0:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    try:
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        dimensions = max_coords - min_coords
        return {'x': dimensions[0], 'y': dimensions[1], 'z': dimensions[2]}
    except Exception:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}

def generate_auxiliary_features(pd_points: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
    """
    Generate auxiliary features (consistent with original data generator)
    
    Args:
        pd_points: Persistence diagram points
        point_cloud: Point cloud data
        
    Returns:
        aux_features: Auxiliary features [14]
    """
    features = []
    
    # Persistence diagram statistics features (5)
    if pd_points is not None and len(pd_points) > 0:
        persistences = pd_points[:, 1] - pd_points[:, 0]
        features.extend([
            len(pd_points),  # Number of points
            np.mean(persistences),  # Mean persistence
            np.std(persistences),   # Persistence standard deviation
            np.max(persistences),   # Maximum persistence
            np.mean(pd_points[:, 0]),  # Mean birth time
        ])
    else:
        features.extend([0] * 5)

    # Point cloud statistics features (3)
    if point_cloud is not None and len(point_cloud) > 0:
        features.extend([
            len(point_cloud),  # Number of points in point cloud
            np.mean(np.std(point_cloud, axis=0)),  # Mean standard deviation of dimensions
            np.mean(np.linalg.norm(point_cloud, axis=1)),  # Mean distance to origin
        ])
        
        # Noise features (3)
        noise_levels = estimate_pointcloud_noise_level(point_cloud)
        features.extend([
            noise_levels['knn_distance_std'],  # KNN distance standard deviation
            noise_levels['pca_eigenvalue_ratio'],  # Local PCA eigenvalue ratio
            noise_levels['density_variation']  # Density variation
        ])
        
        # Round bounding box features (3)
        round_box = compute_roundbox_of_pointcloud(point_cloud)
        features.extend([
            round_box['x'],
            round_box['y'],
            round_box['z']
        ])
    else:
        features.extend([0] * 9) # 3 basic features + 3 noise features + 3 bounding box features

    return np.array(features, dtype=np.float64)


def create_sample_pkl(point_cloud_file: str, 
                     persistence_diagram_file: str,
                     labels_array: np.ndarray,
                     sample_id: int,
                     output_file: str,
                     metadata: dict = None):
    """
    Create .pkl sample file that meets program requirements
    
    Args:
        point_cloud_file: Point cloud txt file path
        persistence_diagram_file: Persistence diagram txt file path
        labels_array: Label array
        sample_id: 样本ID
        output_file: Output pkl file path
        metadata: Optional metadata dictionary
    """
    
    # 1. load point cloud and persistence diagram
    point_cloud = load_point_cloud_from_txt(point_cloud_file)
    pd_points = load_persistence_diagram_from_txt(persistence_diagram_file)
    
    # 2. verify labels array
    if len(labels_array) != len(pd_points):
        raise ValueError(f"Label array length ({len(labels_array)}) does not match number of persistence diagram points ({len(pd_points)})")
    
    # 3. Ensure labels are binary (0/1)
    significance_labels = np.array(labels_array, dtype=np.int32)
    if not np.all(np.isin(significance_labels, [0, 1])):
        raise ValueError("Label array should only contain 0 and 1")
    
    # 3. Generate auxiliary features
    aux_features = generate_auxiliary_features(pd_points, point_cloud)
    
    # 4. Create metadata
    if metadata is None:
        metadata = {}
    
    default_metadata = {
        'cloud_type': 'custom',
        'label_method': 'manual',
        'n_pd_points': len(pd_points),
        'n_pc_points': len(point_cloud),
        'n_significant': np.sum(significance_labels),
        'significance_ratio': np.mean(significance_labels)
    }
    default_metadata.update(metadata)
    
    # 5. Create sample dictionary
    sample = {
        'sample_id': sample_id,
        'pd_points': pd_points,
        'point_cloud': point_cloud,
        'significance_labels': significance_labels,
        'aux_features': aux_features,
        'metadata': default_metadata
    }
    
    # 6. Save file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(sample, f)

    
    return sample

def main():
    parser = argparse.ArgumentParser(description='Convert txt format data to pkl format for training')
    
    parser.add_argument('--point_cloud_file', required=True,
                       help='Point cloud txt file path')
    parser.add_argument('--persistence_diagram_file', required=True,
                       help='Persistence diagram txt file path')
    parser.add_argument('--labels_file', required=True,
                       help='Label file path (.npy or .txt)')
    parser.add_argument('--output_file', required=True,
                       help='Output pkl file path')
    parser.add_argument('--sample_id', type=int, default=1,
                       help='Sample ID')
    parser.add_argument('--cloud_type', default='custom',
                       help='Point cloud type description')
    parser.add_argument('--label_method', default='manual',
                       help='Label method description')
    
    args = parser.parse_args()
    
    try:
        # Load labels
        if args.labels_file.endswith('.npy'):
            labels_array = np.load(args.labels_file)
        else:
            labels_array = np.loadtxt(args.labels_file, dtype=int)
        
        # Create metadata
        metadata = {
            'cloud_type': args.cloud_type,
            'label_method': args.label_method
        }
        
        # Create sample
        create_sample_pkl(
            point_cloud_file=args.point_cloud_file,
            persistence_diagram_file=args.persistence_diagram_file,
            labels_array=labels_array,
            sample_id=args.sample_id,
            output_file=args.output_file,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())