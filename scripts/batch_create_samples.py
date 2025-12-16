"""
Batch create custom samples script
Used to batch process multiple txt files and create training datasets
"""
import shutil
import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json

np.random.seed(42)
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from create_custom_sample import create_sample_pkl

def find_matching_files(data_dir: str):
    """
    Find matching point cloud, persistence diagram, and label files in directory
    
    Expected file naming format:
    - Point cloud: sample_001_pointcloud.txt
    - Persistence diagram: sample_001_persistence.txt  
    - Labels: sample_001_labels.txt
    
    Returns:
        list: List of matched file groups
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"The data directory does not exist: {data_dir}")
    
    # Find all point cloud files
    pointcloud_files = list(data_path.glob("*_pointcloud.txt"))
    
    matched_groups = []
    
    for pc_file in pointcloud_files:
        # Extract base name (e.g: sample_001)
        base_name = pc_file.stem.replace('_pointcloud', '')
        
        # Find corresponding persistence diagram file
        pd_file = data_path / f"{base_name}_persistence.txt"
        if not pd_file.exists():
            print(f"Warning: No persistence diagram file found for {base_name}")
            continue
        
        # Find corresponding label file
        labels_npy = data_path / f"{base_name}_labels.npy"
        labels_txt = data_path / f"{base_name}_labels.txt"
        
        labels_file = None
        if labels_npy.exists():
            labels_file = labels_npy
        elif labels_txt.exists():
            labels_file = labels_txt
        else:
            print(f"Warning: No label file found for {base_name}")
            continue
        
        matched_groups.append({
            'base_name': base_name,
            'pointcloud_file': str(pc_file),
            'persistence_file': str(pd_file),
            'labels_file': str(labels_file)
        })
    
    return matched_groups

def create_dataset_from_directory(data_dir: str, 
                                output_dir: str, 
                                split: str = 'train',
                                start_id: int = 1):
    """
    Batch create dataset from directory
    
    Args:
        data_dir: Directory containing txt files
        output_dir: Output directory
        split: Dataset split name (train/val/test)
        start_id: Starting sample ID
    """
    
    # Find matching files
    print(f"Scanning directory: {data_dir}")
    matched_groups = find_matching_files(data_dir)
    
    if not matched_groups:
        print(f"No matching file groups found")
        return
    print(f"Found {len(matched_groups)} matching file groups")
    
    # Create output directory
    split_dir = os.path.join(output_dir, split)
    shutil.rmtree(split_dir, ignore_errors=True)
    os.makedirs(split_dir)
    
    # Batch process
    created_samples = []
    
    for i, group in enumerate(tqdm(matched_groups, desc="Creating samples")):
        try:
            sample_id = start_id + i
            output_file = os.path.join(split_dir, f"sample_{sample_id:06d}.pkl")
            
            # Load labels
            if group['labels_file'].endswith('.npy'):
                labels_array = np.load(group['labels_file'])
            else:
                labels_array = np.loadtxt(group['labels_file'], dtype=int)
            
            # Create sample
            sample = create_sample_pkl(
                point_cloud_file=group['pointcloud_file'],
                persistence_diagram_file=group['persistence_file'],
                labels_array=labels_array,
                sample_id=sample_id,
                output_file=output_file,
                metadata={
                    'source_files': group,
                    'cloud_type': 'custom',
                    'label_method': 'manual'
                }
            )
            
            created_samples.append({
                'sample_id': sample_id,
                'base_name': group['base_name'],
                'output_file': output_file,
                'n_pd_points': int(len(sample['pd_points'])),
                'n_pc_points': int(len(sample['point_cloud'])),
                'n_significant': int(np.sum(sample['significance_labels']))
            })
            
        except Exception as e:
            print(f"Error processing {group['base_name']}: {e}")
            continue
    
    # Save processing report    
    report_file = os.path.join(output_dir, f"{split}_creation_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'split': split,
            'total_samples': len(created_samples),
            'source_directory': data_dir,
            'output_directory': split_dir,
            'samples': created_samples
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully created {len(created_samples)} samples")
    print(f"Output directory: {split_dir}")
    print(f"Processing report: {report_file}")
    
    if created_samples:
        total_pd_points = sum(s['n_pd_points'] for s in created_samples)
        total_pc_points = sum(s['n_pc_points'] for s in created_samples)
        total_significant = sum(s['n_significant'] for s in created_samples)
        
        print(f"\nDataset statistics:")
        print(f"  - Total samples: {len(created_samples)}")
        print(f"  - Total PD points: {total_pd_points}")
        print(f"  - Total point cloud points: {total_pc_points}")
        print(f"  - Total significant points: {total_significant}")
        print(f"  - Average significance ratio: {total_significant/total_pd_points:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Batch create custom dataset')
    
    parser.add_argument('--data_dir', required=True,
                       help='Input directory containing txt files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory')
    parser.add_argument('--split', default='train',
                       help='Dataset split name')
    parser.add_argument('--start_id', type=int, default=1,
                       help='Starting sample ID')
    
    args = parser.parse_args()
    
    try:
        create_dataset_from_directory(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            split=args.split,
            start_id=args.start_id
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())