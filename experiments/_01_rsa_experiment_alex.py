# %%
"""RSA experiment for analyzing attention head representations - Interactive Notebook Version"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from attendome.dataset.data_loader import ModelLoader
from attendome.dataset.attention_head_classifier import InductionHeadClassifier
from attendome.dataset.rsa_analysis import run_rsa_experiment
from attendome.dataset.utils import save_results, generate_output_filename

# %%
# Configuration parameters
models = ["gpt2", "distilgpt2"]
num_sequences = 20  # Increased for better RSA analysis
seq_len = 16        # Reasonable size for testing
max_heads_per_model = 4
distance_metric = "euclidean"
flatten_method = "upper_triangle"  # Fixed to work with causal attention
clustering_method = "kmeans"       # Options: "kmeans", "hierarchical"
n_clusters = 2
device = None  # Auto-detect
output_dir = "results/rsa_analysis"
verbose = True

print(f"Configuration:")
print(f"  Models: {models}")
print(f"  Sequences: {num_sequences} of length {seq_len}")
print(f"  Max heads per model: {max_heads_per_model}")
print(f"  Distance metric: {distance_metric}")
print(f"  Flatten method: {flatten_method}")

# %%
# Initialize components
print("Initializing components...")
model_loader = ModelLoader(device=device)
classifier = InductionHeadClassifier(device=device)
print(f"Device: {classifier.device}")

# %%
# Step 1: Load models and compute induction scores
all_attention_maps = {}
all_classified_heads = {}
model_configs = {}

# %%
# Load first model and analyze
model_name = models[0]
print(f"\nAnalyzing model: {model_name}")

# Load model
model, tokenizer = model_loader.load_model(model_name)
print(f"  Model loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")

# %%
# Compute induction scores
print("  Computing induction scores...")
results = classifier.analyze_model(
    model=model,
    tokenizer=tokenizer,
    model_name=model_name,
    num_of_samples=50,  # Small for testing
    seq_len=seq_len,
    extract_attention_maps=False
)

print(f"  Found {len(results.classified_heads['high_induction'])} high induction heads")
print(f"  Found {len(results.classified_heads['medium_induction'])} medium induction heads")
print(f"  Found {len(results.classified_heads['low_induction'])} low induction heads")

# %%
# Show top induction heads
print("\nTop induction heads:")
for i, head_info in enumerate(results.classified_heads["high_induction"][:5]):
    print(f"  {i+1}. Layer {head_info['layer']}, Head {head_info['head']}: {head_info['score']:.4f}")

# %%
# Store classifications and config
all_classified_heads[model_name] = results.classified_heads
model_configs[model_name] = results.model_configuration

# Select heads to analyze (balanced mix: half induction, half non-induction)
selected_heads_info = []
head_type_mapping = {}  # Track which heads are induction vs non-induction

# Calculate balanced split
half_heads = max_heads_per_model // 2

# Add induction heads (high + medium if needed)
induction_heads = results.classified_heads["high_induction"].copy()
if len(induction_heads) < half_heads:
    # Add medium induction heads if we don't have enough high ones
    induction_heads.extend(results.classified_heads["medium_induction"])

# Take first half_heads induction heads
for head_info in induction_heads[:half_heads]:
    layer_idx, head_idx = head_info["layer"], head_info["head"]
    selected_heads_info.append((layer_idx, head_idx, "induction"))
    head_type_mapping[f"{model_name}_layer_{layer_idx}_head_{head_idx}"] = "induction"

# Add non-induction heads (low induction)
non_induction_heads = results.classified_heads["low_induction"].copy()
np.random.shuffle(non_induction_heads)
for head_info in non_induction_heads[:half_heads]:
    layer_idx, head_idx = head_info["layer"], head_info["head"]
    selected_heads_info.append((layer_idx, head_idx, "non_induction"))
    head_type_mapping[f"{model_name}_layer_{layer_idx}_head_{head_idx}"] = "non_induction"

print(f"\nSelected {len(selected_heads_info)} heads for RSA analysis:")
for layer_idx, head_idx, label in selected_heads_info:
    print(f"  Layer {layer_idx}, Head {head_idx} ({label})")

# %%
# Extract attention maps for selected heads
print("\nExtracting attention maps...")

for layer_idx, head_idx, label in selected_heads_info:
    attention_maps = classifier.extract_induction_attention_maps(
        model=model,
        tokenizer=tokenizer,
        num_sequences=num_sequences,
        seq_len=seq_len,
        layer_indices=[layer_idx],
        head_indices=[head_idx],
        batch_size=8
    )
    
    # Add model prefix to attention map keys
    for key, value in attention_maps.items():
        all_attention_maps[f"{model_name}_{key}"] = value
        print(f"  Extracted {key}: shape {value.shape}")

# Clear model from memory
del model, tokenizer
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %%
# Load second model and analyze
if len(models) > 1:
    model_name = models[1]
    print(f"\nAnalyzing model: {model_name}")
    
    # Load model
    model, tokenizer = model_loader.load_model(model_name)
    print(f"  Model loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    
    # Compute induction scores
    print("  Computing induction scores...")
    results = classifier.analyze_model(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        num_of_samples=50,
        seq_len=seq_len,
        extract_attention_maps=False
    )
    
    print(f"  Found {len(results.classified_heads['high_induction'])} high induction heads")
    
    # Store classifications and config
    all_classified_heads[model_name] = results.classified_heads
    model_configs[model_name] = results.model_configuration
    
    # Select heads (balanced mix: half induction, half non-induction)
    selected_heads_info = []
    
    # Calculate balanced split
    half_heads = max_heads_per_model // 2
    
    # Add induction heads (high + medium if needed)
    induction_heads = results.classified_heads["high_induction"].copy()
    if len(induction_heads) < half_heads:
        # Add medium induction heads if we don't have enough high ones
        induction_heads.extend(results.classified_heads["medium_induction"])
    
    # Take first half_heads induction heads
    for head_info in induction_heads[:half_heads]:
        layer_idx, head_idx = head_info["layer"], head_info["head"]
        selected_heads_info.append((layer_idx, head_idx, "induction"))
        head_type_mapping[f"{model_name}_layer_{layer_idx}_head_{head_idx}"] = "induction"
    
    # Add non-induction heads (low induction)
    non_induction_heads = results.classified_heads["low_induction"].copy()
    np.random.shuffle(non_induction_heads)
    for head_info in non_induction_heads[:half_heads]:
        layer_idx, head_idx = head_info["layer"], head_info["head"]
        selected_heads_info.append((layer_idx, head_idx, "non_induction"))
        head_type_mapping[f"{model_name}_layer_{layer_idx}_head_{head_idx}"] = "non_induction"
    
    print(f"\nSelected {len(selected_heads_info)} heads from {model_name}")
    
    # Extract attention maps
    print("  Extracting attention maps...")
    for layer_idx, head_idx, label in selected_heads_info:
        attention_maps = classifier.extract_induction_attention_maps(
            model=model,
            tokenizer=tokenizer,
            num_sequences=num_sequences,
            seq_len=seq_len,
            layer_indices=[layer_idx],
            head_indices=[head_idx],
            batch_size=8
        )
        
        for key, value in attention_maps.items():
            all_attention_maps[f"{model_name}_{key}"] = value
            print(f"    Extracted {key}: shape {value.shape}")
    
    # Clear model from memory
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %%
# Check what we extracted
print(f"\nTotal attention maps extracted: {len(all_attention_maps)}")
print("Attention map summary:")
for key, tensor in list(all_attention_maps.items())[:8]:  # Show first 8
    print(f"  {key}: shape {tensor.shape}")
    # Check if attention maps contain actual values (not all zeros)
    tensor_np = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else np.array(tensor)
    print(f"    Mean: {np.mean(tensor_np):.6f}, Max: {np.max(tensor_np):.6f}, Min: {np.min(tensor_np):.6f}")
    if np.all(tensor_np == 0):
        print(f"    WARNING: {key} contains all zeros!")

# %%
# Visualize attention maps in a grid
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nCreating attention maps visualization...")
    
    # Select which sequence to visualize (default: first sequence)
    sequence_idx = 0
    
    # Calculate grid dimensions
    n_maps = len(all_attention_maps)
    n_cols = int(np.ceil(np.sqrt(n_maps)))
    n_rows = int(np.ceil(n_maps / n_cols))
    
    print(f"Creating {n_rows}x{n_cols} grid for {n_maps} attention maps")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    fig.suptitle(f'Attention Maps Grid (Sequence {sequence_idx})', fontsize=16, y=0.98)
    
    # Handle single subplot case
    if n_maps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    # Find global min/max for consistent color scaling
    all_values = []
    for tensor in all_attention_maps.values():
        attention_matrix = tensor[sequence_idx].cpu().numpy() if hasattr(tensor, 'cpu') else np.array(tensor[sequence_idx])
        all_values.extend(attention_matrix.flatten())
    
    vmin, vmax = np.min(all_values), np.max(all_values)
    print(f"Attention value range: [{vmin:.4f}, {vmax:.4f}]")
    
    # Plot each attention map
    for idx, (head_key, tensor) in enumerate(all_attention_maps.items()):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        
        # Convert tensor to numpy
        attention_matrix = tensor[sequence_idx].cpu().numpy() if hasattr(tensor, 'cpu') else np.array(tensor[sequence_idx])
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            ax=ax,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=False,
            xticklabels=False,
            yticklabels=False
        )
        
        # Parse head information for title
        title_parts = head_key.split('_')
        if len(title_parts) >= 4:
            model_name = title_parts[0]
            layer_idx = title_parts[2]
            head_idx = title_parts[4]
            
            # Determine head type from our mapping
            head_type_label = head_type_mapping.get(head_key, "unknown")
            
            if head_type_label == "induction":
                head_type = "INDUCTION"
                type_color = "red"
            elif head_type_label == "non_induction":
                head_type = "NON-INDUCTION"
                type_color = "blue"
            else:
                head_type = "UNKNOWN"
                type_color = "gray"
            
            title = f"{model_name} L{layer_idx}H{head_idx}\n{head_type}"
        else:
            title = head_key[:20] + "..." if len(head_key) > 20 else head_key
            type_color = "gray"
        
        ax.set_title(title, fontsize=9, pad=3, color=type_color, weight='bold')
        
        # Add colored border based on head type
        for spine in ax.spines.values():
            spine.set_color(type_color)
            spine.set_linewidth(3)
    
    # Hide unused subplots
    for idx in range(n_maps, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add single colorbar
    if n_maps > 0:
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap="Blues")
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=axes_flat[:n_maps], orientation='horizontal', 
                           pad=0.15, shrink=0.8, aspect=40)
        cbar.set_label('Attention Weight', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.3)
    plt.show()
    
    # Also create a comparison plot of induction vs non-induction heads
    induction_heads = {k: v for k, v in all_attention_maps.items() if head_type_mapping.get(k) == "induction"}
    non_induction_heads = {k: v for k, v in all_attention_maps.items() if head_type_mapping.get(k) == "non_induction"}
    
    if induction_heads and non_induction_heads:
        print(f"\nComparing induction vs non-induction heads...")
        
        # Take first example of each type
        induction_key = list(induction_heads.keys())[0]
        non_induction_key = list(non_induction_heads.keys())[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Induction vs Non-Induction Head Comparison', fontsize=14)
        
        # Induction head
        induction_matrix = induction_heads[induction_key][sequence_idx]
        induction_matrix = induction_matrix.cpu().numpy() if hasattr(induction_matrix, 'cpu') else np.array(induction_matrix)
        
        sns.heatmap(induction_matrix, ax=ax1, cmap="Reds", square=True, 
                   cbar_kws={'label': 'Attention'}, xticklabels=False, yticklabels=False)
        ax1.set_title(f"INDUCTION HEAD\n{induction_key}", fontsize=11, color='red', weight='bold')
        
        # Non-induction head  
        non_induction_matrix = non_induction_heads[non_induction_key][sequence_idx]
        non_induction_matrix = non_induction_matrix.cpu().numpy() if hasattr(non_induction_matrix, 'cpu') else np.array(non_induction_matrix)
        
        sns.heatmap(non_induction_matrix, ax=ax2, cmap="Blues", square=True,
                   cbar_kws={'label': 'Attention'}, xticklabels=False, yticklabels=False)
        ax2.set_title(f"NOT INDUCTION HEAD\n{non_induction_key}", fontsize=11, color='blue', weight='bold')
        
        plt.tight_layout()
        plt.show()

except ImportError:
    print("Matplotlib/seaborn not available - skipping attention map visualization")
except Exception as e:
    print(f"Error creating attention map visualization: {e}")

# %%
# Create classified heads ONLY for the extracted attention maps
# This ensures the RSA analysis only considers heads we actually have data for
extracted_head_keys = set(all_attention_maps.keys())
print(f"Extracted attention maps for: {len(extracted_head_keys)} heads")

combined_classified_heads = {
    "high_induction": [],
    "medium_induction": [],
    "low_induction": []
}

# Only include heads that we have attention maps for
for model_name, classified in all_classified_heads.items():
    for category, heads in classified.items():
        for head_info in heads:
            # Create the key that would be used in attention maps
            attention_map_key = f"{model_name}_layer_{head_info['layer']}_head_{head_info['head']}"
            
            # Only include if we have an attention map for this head
            if attention_map_key in extracted_head_keys:
                global_head_info = head_info.copy()
                combined_classified_heads[category].append(global_head_info)

print(f"Filtered classifications (only heads with attention maps):")
for category, heads in combined_classified_heads.items():
    print(f"  {category}: {len(heads)} heads")

# %%
# Step 2: Run RSA Analysis
print(f"\nRunning RSA analysis...")
print(f"Distance metric: {distance_metric}")
print(f"Flatten method: {flatten_method}")
print(f"Clustering method: {clustering_method}")

experiment_results = run_rsa_experiment(
    attention_maps=all_attention_maps,
    classified_heads=combined_classified_heads,
    distance_metric=distance_metric,
    flatten_method=flatten_method,
    clustering_method=clustering_method,
    n_clusters=n_clusters,
    target_seq_len=seq_len * 2  # Use consistent sequence length
)

# %%
# Step 3: Analyze Results
print("\n" + "="*50)
print("RSA EXPERIMENT RESULTS")
print("="*50)

metrics = experiment_results["evaluation_metrics"]
print(f"Total heads analyzed: {len(experiment_results['head_labels'])}")
print(f"True induction heads: {np.sum(experiment_results['true_labels'])}")

if "adjusted_rand_score" in metrics:
    print(f"Adjusted Rand Score: {metrics['adjusted_rand_score']:.4f}")
    print(f"  (1.0 = perfect clustering, 0.0 = random)")

if "silhouette_score" in metrics:
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  (1.0 = well separated clusters, -1.0 = poor clustering)")

# %%
# Show detailed cluster assignments
print(f"\nDetailed cluster assignments:")
head_labels = experiment_results["head_labels"]
cluster_labels = experiment_results["clustering_results"]["cluster_labels"]
true_labels = experiment_results["true_labels"]

for i, (head, cluster, is_induction) in enumerate(zip(head_labels, cluster_labels, true_labels)):
    status = "INDUCTION" if is_induction else "other"
    print(f"  {head}: cluster {cluster} ({status})")

# %%
# Examine the representation matrix
representation_matrix = experiment_results["representation_matrix"]
print(f"\nRepresentation matrix shape: {representation_matrix.shape}")
print(f"Each row represents one attention head")
print(f"Each column represents a distance feature")

# Show some statistics
print(f"Matrix statistics:")
print(f"  Mean: {np.mean(representation_matrix):.4f}")
print(f"  Std: {np.std(representation_matrix):.4f}")
print(f"  Min: {np.min(representation_matrix):.4f}")
print(f"  Max: {np.max(representation_matrix):.4f}")

# %%
# Simplified RSA Results Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Separate heads by type for clearer visualization
induction_indices = [i for i, label in enumerate(true_labels) if label == 1]
non_induction_indices = [i for i, label in enumerate(true_labels) if label == 0]

print(f"\nHead breakdown:")
print(f"  Induction heads: {len(induction_indices)} (indices: {induction_indices})")
print(f"  Non-induction heads: {len(non_induction_indices)} (indices: {non_induction_indices})")

# Create a focused comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Representation matrix with clear labels
colors = ['red' if label == 1 else 'blue' for label in true_labels]
y_labels = []
for i, (head, is_induction) in enumerate(zip(head_labels, true_labels)):
    head_type = "[IND]" if is_induction else "[NON]"
    y_labels.append(f"{head_type} {head}")

im = ax1.imshow(representation_matrix, cmap='viridis', aspect='auto')
ax1.set_title('RSA Representations: Induction vs Non-Induction Heads')
ax1.set_xlabel('Distance Features')
ax1.set_ylabel('Attention Heads')
ax1.set_yticks(range(len(head_labels)))
ax1.set_yticklabels(y_labels, fontsize=8)

# Color y-axis labels
for i, (tick, color) in enumerate(zip(ax1.get_yticklabels(), colors)):
    tick.set_color(color)
    tick.set_weight('bold')

plt.colorbar(im, ax=ax1, label='Distance')

# 2. Clustering results with clear separation
cluster_colors = ['red' if label == 1 else 'blue' for label in true_labels]
cluster_shapes = ['o' if cluster == 0 else 's' for cluster in cluster_labels]

# Plot induction heads
induction_x = [i for i in induction_indices]
induction_y = [0.1] * len(induction_indices)
induction_clusters = [cluster_labels[i] for i in induction_indices]
induction_shapes = ['o' if c == 0 else 's' for c in induction_clusters]

for x, shape in zip(induction_x, induction_shapes):
    ax2.scatter(x, 0.1, c='red', marker=shape, s=150, alpha=0.8, edgecolors='darkred', linewidth=2)

# Plot non-induction heads
non_induction_x = [i for i in non_induction_indices]
non_induction_y = [-0.1] * len(non_induction_indices)
non_induction_clusters = [cluster_labels[i] for i in non_induction_indices]
non_induction_shapes = ['o' if c == 0 else 's' for c in non_induction_clusters]

for x, shape in zip(non_induction_x, non_induction_shapes):
    ax2.scatter(x, -0.1, c='blue', marker=shape, s=150, alpha=0.8, edgecolors='darkblue', linewidth=2)

ax2.set_title('Clustering Results: Induction vs Non-Induction')
ax2.set_xlabel('Head Index')
ax2.set_ylabel('')
ax2.set_ylim(-0.3, 0.3)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

# Add text labels
ax2.text(-0.5, 0.1, 'INDUCTION\nHEADS', ha='right', va='center', 
         fontweight='bold', color='red', fontsize=10)
ax2.text(-0.5, -0.1, 'NON-INDUCTION\nHEADS', ha='right', va='center', 
         fontweight='bold', color='blue', fontsize=10)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Cluster 0'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Cluster 1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Induction Head'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Non-Induction Head')
]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

# Print clustering performance summary
print(f"\nClustering Performance Summary:")
print(f"  Total heads: {len(head_labels)}")
print(f"  Induction heads: {len(induction_indices)} | Non-induction heads: {len(non_induction_indices)}")
if len(induction_indices) > 0 and len(non_induction_indices) > 0:
    print(f"  Perfect separation would have Adjusted Rand Score = 1.0")
    print(f"  Actual Adjusted Rand Score: {metrics.get('adjusted_rand_score', 'N/A'):.4f}")

# %%
# Step 4: Save results (optional)
save_results_flag = False  # Set to True if you want to save

if save_results_flag:
    # Add metadata
    experiment_results["metadata"] = {
        "models": list(models),
        "model_configs": model_configs,
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "max_heads_per_model": max_heads_per_model,
        "total_heads_analyzed": len(all_attention_maps)
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = generate_output_filename(
        f"rsa_experiment_{distance_metric}_{flatten_method}",
        base_dir=str(output_path),
        extension="json"
    )
    
    # Remove non-serializable objects for saving
    save_results_copy = experiment_results.copy()
    if "clusterer" in save_results_copy["clustering_results"]:
        del save_results_copy["clustering_results"]["clusterer"]
    
    # Convert numpy arrays to lists for JSON serialization
    save_results_copy["representation_matrix"] = experiment_results["representation_matrix"].tolist()
    save_results_copy["true_labels"] = experiment_results["true_labels"].tolist()
    save_results_copy["clustering_results"]["cluster_labels"] = experiment_results["clustering_results"]["cluster_labels"].tolist()
    
    save_results(save_results_copy, filename, format="json")
    print(f"\nResults saved to: {filename}")

# %%
# t-SNE Visualization of All Attention Heads
print("\n" + "="*50)
print("t-SNE VISUALIZATION OF ATTENTION HEADS")
print("="*50)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Prepare data for t-SNE
print("Running t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(head_labels)-1))
tsne_results = tsne.fit_transform(representation_matrix)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 10))

# Color points by induction vs non-induction
colors = ['red' if label == 1 else 'blue' for label in true_labels]
labels = ['Induction' if label == 1 else 'Non-induction' for label in true_labels]

# Plot points
scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                    c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=1)

# Add text labels for each point
for i, (x, y) in enumerate(tsne_results):
    # Create clean label from head info
    head_label = head_labels[i]
    head_type = "IND" if true_labels[i] == 1 else "NON"
    cluster = cluster_labels[i]
    
    # Position text slightly offset from point
    ax.annotate(f'{head_label}\n{head_type}(C{cluster})', 
                xy=(x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Customize plot
ax.set_title('t-SNE Visualization of Attention Head Representations', fontsize=16, fontweight='bold')
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.grid(True, alpha=0.3)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, label='Induction Heads'),
    Patch(facecolor='blue', alpha=0.7, label='Non-Induction Heads')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Add statistics text
stats_text = f"""
Total Heads: {len(head_labels)}
Induction: {np.sum(true_labels)} heads
Non-induction: {len(true_labels) - np.sum(true_labels)} heads
Clusters: {len(np.unique(cluster_labels))}
"""
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Print analysis of t-SNE results
print(f"\nt-SNE Analysis:")
print(f"  Total heads plotted: {len(head_labels)}")
print(f"  Induction heads: {np.sum(true_labels)}")
print(f"  Non-induction heads: {len(true_labels) - np.sum(true_labels)}")

# Calculate distances between induction and non-induction heads in t-SNE space
induction_points = tsne_results[true_labels == 1]
non_induction_points = tsne_results[true_labels == 0]

if len(induction_points) > 0 and len(non_induction_points) > 0:
    # Calculate average distance between groups
    from scipy.spatial.distance import cdist
    cross_distances = cdist(induction_points, non_induction_points, metric='euclidean')
    avg_cross_distance = np.mean(cross_distances)
    
    print(f"  Average distance between induction and non-induction heads: {avg_cross_distance:.2f}")
    
    # Calculate within-group distances
    if len(induction_points) > 1:
        induction_distances = pdist(induction_points, metric='euclidean')
        avg_induction_distance = np.mean(induction_distances)
        print(f"  Average distance within induction heads: {avg_induction_distance:.2f}")
    
    if len(non_induction_points) > 1:
        non_induction_distances = pdist(non_induction_points, metric='euclidean')
        avg_non_induction_distance = np.mean(non_induction_distances)
        print(f"  Average distance within non-induction heads: {avg_non_induction_distance:.2f}")

# %%
# Experiment with different parameters
print("\n" + "="*50)
print("PARAMETER EXPLORATION")
print("="*50)

# You can now easily modify the parameters above and re-run the analysis
# Try different:
# - distance_metric: "euclidean", "cosine", "manhattan"
# - flatten_method: "full", "upper_triangle", "diagonal"
# - clustering_method: "kmeans", "hierarchical"
# - num_sequences: 5, 10, 20, 50
# - seq_len: 10, 20, 30, 50

print("To experiment with different parameters:")
print("1. Modify the configuration cell (2nd cell)")
print("2. Re-run from the RSA analysis cell onward")
print("3. Compare results between different configurations")

# %%