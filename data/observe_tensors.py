import torch
import pickle
import numpy as np
import pandas as pd

# Define paths
gene_exp_path = '/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl'
methylation_path = '/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl'
cancer_tags_path = '/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl'

def inspect_tensor(name, tensor):
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Type: {type(tensor)}")
    print(f"Shape: {tensor.shape} (Samples: {tensor.shape[0]}, Features: {tensor.shape[1] if len(tensor.shape) > 1 else 1})")
    print(f"Dtype: {tensor.dtype}")
    
    # Check for NaNs and Infs
    # Note: Using .float() in case it's not already, though it should be float32
    print(f"Has NaNs: {torch.isnan(tensor).any().item()}")
    print(f"Has Infs: {torch.isinf(tensor).any().item()}")
    
    # Statistics
    if tensor.numel() > 0:
        print(f"Min: {tensor.min().item():.4f}")
        print(f"Max: {tensor.max().item():.4f}")
        print(f"Mean: {tensor.mean().item():.4f}")
        print(f"Std Dev: {tensor.std().item():.4f}")
    
    # Peek at first few rows/cols
    rows = min(5, tensor.shape[0])
    cols = min(5, tensor.shape[1])
    print(f"Peek (First {rows} samples, {cols} features):")
    print(tensor[:rows, :cols])

def main():
    try:
        # Load data
        print("Loading datasets...")
        with open(gene_exp_path, 'rb') as f:
            gene_exp = pickle.load(f)
        with open(methylation_path, 'rb') as f:
            methylation = pickle.load(f)
        with open(cancer_tags_path, 'rb') as f:
            cancer_tags = pickle.load(f)
        
        # Inspections
        inspect_tensor("Gene Expression Tensor", gene_exp)
        inspect_tensor("Methylation Tensor", methylation)
        
        # Specilized inspection for one-hot cancer tags
        print(f"\n{'='*20} Cancer Tags Tensor {'='*20}")
        print(f"Shape: {cancer_tags.shape}")
        print(f"Dtype: {cancer_tags.dtype}")
        
        # Reverse one-hot to see class distribution (if possible)
        # Note: We don't have the original labels here, but we can see counts
        class_indices = torch.argmax(cancer_tags, dim=1)
        unique_classes, counts = torch.unique(class_indices, return_counts=True)
        print(f"Number of classes: {len(unique_classes)}")
        print("Class distributions (indices):")
        for cls, count in zip(unique_classes, counts):
            print(f"  Class index {cls.item()}: {count.item()} samples")

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
