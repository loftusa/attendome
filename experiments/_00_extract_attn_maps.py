#!/usr/bin/env python3
"""
Extract attention maps from transformer models using the Common Pile dataset.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
import torch
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

def parse_args():
    parser = argparse.ArgumentParser(description="Extract attention maps from transformer models")
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        choices=[
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B", 
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen3-Embedding-8B",
            "allenai/OLMo-2-1124-7B",
            "EleutherAI/pythia-6.9b",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "EleutherAI/gpt-j-6b",
            "EleutherAI/gpt-neox-20b",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf"
        ],
        help="Model name to load"
    )
    parser.add_argument(
        "--model_name_clean", 
        type=str, 
        required=True,
        help="Model name to save"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to process from the dataset"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="common-pile/comma_v0.1_training_dataset",
        help="Dataset name to load from HuggingFace"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=150,
        help="Maximum sequence length"
    )
    
    # Attention extraction arguments
    parser.add_argument(
        "--store_all", 
        action="store_true", 
        default=True,
        help="Store attention maps for all heads (default: True)"
    )
    
    parser.add_argument(
        "--store_specific_heads",
        action="store_true",
        help="Store only specific heads defined in store_list"
    )
    
    parser.add_argument(
        "--head_list",
        type=str,
        default="",
        help="Comma-separated list of layer,head pairs (e.g., '13,13;14,12;2,31')"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results/attention_maps",
        help="Output directory for attention maps"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on (cuda/cpu)"
    )

    # compute RDM directly for batch size = 2 
    parser.add_argument(
        "--compute_rdm",
        action="store_true", 
        default=False,
    )
    
    return parser.parse_args()

def parse_head_list(head_list_str):
    """Parse head list string into list of tuples."""
    if not head_list_str:
        return []
    
    head_pairs = []
    for pair in head_list_str.split(';'):
        layer, head = map(int, pair.split(','))
        head_pairs.append((layer, head))
    return head_pairs

def load_dataset_samples(dataset_name, num_samples):
    """Load dataset samples."""
    from datasets import load_dataset
    
    dataset = load_dataset(
        dataset_name, 
        streaming=True,
    )
    
    # Convert first N examples to a list
    subset_data = []
    for i, example in enumerate(dataset['train']):
        if i >= num_samples:
            break
        subset_data.append(example['text'].lstrip())
    
    return subset_data

def setup_model_and_tokenizer(model_name, device):
    """Load model and tokenizer."""
    from attendome.dataset import ModelLoader 
    
    loader = ModelLoader(device)
    model, tokenizer = loader.load_model(model_name)
    
    # Configure model for attention extraction
    model.eval()
    model.config._attn_implementation = "eager"
    
    # Set up tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    
    return model, tokenizer

def extract_attention_maps(model, tokenizer, subset_data, args):
    """Extract attention maps from the model."""
    attention_maps = {}
    attention_distances = {}
    
    if args.compute_rdm:
        assert args.batch_size == 2, f"unsupported batch size: {args.batch_size}"
    
    # Parse head list if using specific heads
    store_list = []
    if args.store_specific_heads:
        store_list = parse_head_list(args.head_list)
        store_all = False
    else:
        store_all = args.store_all
    
    with torch.no_grad():
        for i in tqdm(range(0, args.num_samples, args.batch_size), desc="Extracting attention maps"):
            begin_index = i
            end_index = min(i + args.batch_size, args.num_samples)
            current_batch_size = end_index - begin_index
            
            # Get batch of strings
            batch_strings = subset_data[begin_index:end_index]
            
            # Tokenize with padding and truncation
            batch_tokens = tokenizer(
                batch_strings,
                padding=True,
                truncation=True,
                max_length=args.seq_len,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            input_data = {
                "input_ids": batch_tokens["input_ids"].to(args.device),
                "attention_mask": batch_tokens["attention_mask"].to(args.device)
            }
            
            result = model(**input_data, output_attentions=True)
            
            if store_all:
                # Store attention maps for each layer and head
                for layer in range(model.config.num_hidden_layers):
                    layer_values = result.attentions[layer]  # Shape: (batch_size, num_heads, seq_len, seq_len)
                    
                    if layer not in attention_maps:
                        attention_maps[layer] = {}
                        attention_distances[layer] = {}
                    for head in range(layer_values.shape[1]):  # num_heads
                        if head not in attention_maps[layer]:
                            attention_maps[layer][head] = []
                            attention_distances[layer][head] = []
                        
                        # Store attention map for this head across all samples in batch
                        head_attention = layer_values[:, head, :, :].cpu()  # (batch_size, seq_len, seq_len)
                        if torch.isnan(head_attention.sum()):
                            print(f"Warning!! Layer{layer}-head{head} contains NaN")

                        if args.compute_rdm:
                            map1 = head_attention[0]  # (seq_len, seq_len)
                            map2 = head_attention[1]  # (seq_len, seq_len)
                            
                            # Compute squared Euclidean distance between the two maps
                            squared_distance = torch.sum((map1 - map2) ** 2).item()
                            
                            # Store the distance
                            attention_distances[layer][head].append(squared_distance)
                        else:
                            attention_maps[layer][head].append(head_attention)
            else:
                # Store selected heads only
                for (layer, head) in store_list:
                    if layer not in attention_maps:
                        attention_maps[layer] = {}
                    if head not in attention_maps[layer]:
                        attention_maps[layer][head] = []
                    head_attention = result.attentions[layer][:, head, :, :].cpu()  # (batch_size, seq_len, seq_len)
                    attention_maps[layer][head].append(head_attention)

    if args.compute_rdm:
        return attention_distances
    else:
        return attention_maps

def save_attention_maps(attention_maps, model_name, args):
    """Save attention maps to file."""
    # Concatenate all batches for each head
    print("Concatenating attention maps...")
    for layer in attention_maps:
        for head in attention_maps[layer]:
            if args.compute_rdm:
                attention_maps[layer][head] = np.array(attention_maps[layer][head])
            else:# concatenating attention maps
                attention_maps[layer][head] = torch.cat(attention_maps[layer][head], dim=0)
                n_sample = attention_maps[layer][head].shape[0]
                attention_maps[layer][head] = attention_maps[layer][head].reshape((n_sample, -1)).numpy()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    if args.compute_rdm:
        file_path = output_dir / f"rdms_{args.model_name_clean}_pile-{args.num_samples}-{args.seq_len}.pkl"
    else:
        file_path = output_dir / f"attn_maps_{args.model_name_clean}_pile-{args.num_samples}-{args.seq_len}.pkl"
    
    with open(file_path, 'wb') as f:
        pickle.dump(attention_maps, f)
    
    print(f"Attention maps saved to {file_path}")
    return file_path

def main():
    args = parse_args()
    
    print(f"Extracting attention maps from {args.model_name}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Device: {args.device}")
    
    # Load dataset
    print("Loading dataset...")
    subset_data = load_dataset_samples(args.dataset_name, args.num_samples)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
    
    # Extract attention maps
    print("Extracting attention maps...")
    attention_maps = extract_attention_maps(model, tokenizer, subset_data, args)
    
    # Save results
    print("Saving attention maps...")
    output_file = save_attention_maps(attention_maps, args.model_name, args)
    
    print(f"Extraction complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()