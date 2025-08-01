#!/usr/bin/env python3
"""
Script to compute Representational Dissimilarity Matrices (RDMs) from attention maps.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

import rsatoolbox


def load_attention_maps(file_path):
    """Load attention maps from pickle file."""
    try:
        with open(file_path, 'rb') as file:
            attention_maps = pickle.load(file)
        print(f"Successfully loaded attention maps from {file_path}")
        return attention_maps
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading attention maps: {e}")
        sys.exit(1)

def compute_rdm_fast(data):
    """
    Compute representational dissimilarity matrix using squared Euclidean distance.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
    
    Returns:
        numpy array of shape (n_pairs,) containing upper triangular RDM values
        where n_pairs = n_samples * (n_samples - 1) / 2
    """
    n_samples = data.shape[0]
    
    # Compute squared norms for each sample
    squared_norms = np.sum(data**2, axis=1)
    
    # Compute dot product matrix
    dot_products = np.dot(data, data.T)
    
    # Compute squared Euclidean distance matrix using:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
    distances_squared = (squared_norms[:, np.newaxis] + 
                        squared_norms[np.newaxis, :] - 
                        2 * dot_products)
    
    # Extract upper triangular part (excluding diagonal)
    upper_tri_indices = np.triu_indices(n_samples, k=1)
    rdm_flat = distances_squared[upper_tri_indices]
    
    return rdm_flat

import torch
    
def compute_rdm_fast_torch(data, device='cuda'):
    """
    GPU-accelerated RDM computation using PyTorch.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        device: 'cuda' or 'cpu'
    
    Returns:
        numpy array of shape (n_pairs,) containing upper triangular RDM values
    """
    # Transfer data to GPU
    data_tensor = torch.from_numpy(data).float().to(device)
    n_samples = data_tensor.shape[0]
    
    # Compute squared norms for each sample
    squared_norms = torch.sum(data_tensor**2, dim=1)
    
    # Compute dot product matrix
    dot_products = torch.mm(data_tensor, data_tensor.T)
    
    # Compute squared Euclidean distance matrix
    distances_squared = (squared_norms.unsqueeze(1) + 
                        squared_norms.unsqueeze(0) - 
                        2 * dot_products)
    
    # Extract upper triangular part (excluding diagonal)
    upper_tri_indices = torch.triu_indices(n_samples, n_samples, offset=1)
    rdm_flat = distances_squared[upper_tri_indices[0], upper_tri_indices[1]]
    
    # Transfer result back to CPU
    return rdm_flat.cpu().numpy()

def compute_rdms(attention_maps):
    """Compute RDMs for all layers and heads."""
    all_rdms = {}
    
    print("Computing RDMs...")
    # for layer in tqdm(attention_maps, desc="Processing layers"):
    #     all_rdms[layer] = {}
    #     for head in attention_maps[layer]:
    #         data = rsatoolbox.data.Dataset(attention_maps[layer][head])
    #         rdms = rsatoolbox.rdm.calc_rdm(data)
    #         all_rdms[layer][head] = rdms.dissimilarities.reshape(-1)

    all_rdms = {}
    for layer in tqdm(attention_maps, desc="Processing layers"):
        all_rdms[layer] = {}
        for head in attention_maps[layer]:
            # Get attention data for this head
            head_data = attention_maps[layer][head]
            
            # Ensure data is 2D (samples x features)
            assert head_data.ndim == 2
            
            # Compute RDM using fast custom function
            # rdm_flat = compute_rdm_fast(head_data)
            # rdm_flat = compute_rdm_einsum(head_data)
            rdm_flat = compute_rdm_fast_torch(head_data)
            if np.isnan(rdm_flat.sum()):
                print(f"Warning! nan for layer{layer}-head{head}")
            all_rdms[layer][head] = rdm_flat    
    
    return all_rdms

def save_rdms(all_rdms, file_path):
    """Save RDMs to pickle file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(all_rdms, f)
        print(f"RDMs saved to {file_path}")
    except Exception as e:
        print(f"Error saving RDMs: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Representational Dissimilarity Matrices (RDMs) from attention maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Clean model name (used in file paths)"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples used"
    )
    
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=150,
        help="Sequence length"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="./results/attention_maps",
        help="Directory containing attention maps"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./results/attention_maps",
        help="Directory to save RDMs"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="pile",
        help="Dataset name"
    )
    
    args = parser.parse_args()
    
    # Construct file paths
    input_filename = f"attn_maps_{args.model_name}_{args.dataset}-{args.num_samples}-{args.seq_len}.pkl"
    input_path = Path(args.input_dir) / input_filename
    
    output_filename = f"rdms_{args.model_name}_{args.dataset}-{args.num_samples}-{args.seq_len}.pkl"
    output_path = Path(args.output_dir) / output_filename
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
    # Load attention maps
    attention_maps = load_attention_maps(input_path)
    
    # Compute RDMs
    all_rdms = compute_rdms(attention_maps)
    
    # Save RDMs
    save_rdms(all_rdms, output_path)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()