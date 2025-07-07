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
num_sequences = 10  # Small for testing
seq_len = 20        # Small for testing
max_heads_per_model = 4
distance_metric = "euclidean"
flatten_method = "upper_triangle"  # Options: "full", "upper_triangle", "diagonal"
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
# Quick visualization of results using the new plot method
print("\nQuick visualization of analysis results:")
results.plot("overview")

# %%
# Show different plot types
print("\nDifferent plot types available:")
print("1. Overview plot (4 subplots)")
results.plot("overview", figsize=(15, 10))

# %%
print("2. Detailed scores heatmap")
results.plot("scores", annotate=True)

# %%
print("3. Heatmap with classification boundaries")
results.plot("heatmap")

# %%
print("4. Score distribution analysis")
results.plot("distribution")

# %%
# Show top induction heads
print("\nTop induction heads:")
for i, head_info in enumerate(results.classified_heads["high_induction"][:5]):
    print(f"  {i+1}. Layer {head_info['layer']}, Head {head_info['head']}: {head_info['score']:.4f}")

# %%
# Store classifications and config
all_classified_heads[model_name] = results.classified_heads
model_configs[model_name] = results.model_configuration

# Select heads to analyze (mix of high induction and random heads)
selected_heads_info = []

# Add high induction heads
high_induction = results.classified_heads["high_induction"]
for head_info in high_induction[:max_heads_per_model//2]:
    selected_heads_info.append((head_info["layer"], head_info["head"], "high"))

# Add some random low induction heads for comparison
low_induction = results.classified_heads["low_induction"]
np.random.shuffle(low_induction)
for head_info in low_induction[:max_heads_per_model//2]:
    selected_heads_info.append((head_info["layer"], head_info["head"], "low"))

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
    results2 = classifier.analyze_model(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        num_of_samples=50,
        seq_len=seq_len,
        extract_attention_maps=False
    )
    
    print(f"  Found {len(results2.classified_heads['high_induction'])} high induction heads")
    
    # Quick plot for second model
    print("  Visualization for second model:")
    results2.plot("overview")
    
    # Store classifications and config
    all_classified_heads[model_name] = results2.classified_heads
    model_configs[model_name] = results2.model_configuration
    
    # Select heads
    selected_heads_info = []
    high_induction = results2.classified_heads["high_induction"]
    for head_info in high_induction[:max_heads_per_model//2]:
        selected_heads_info.append((head_info["layer"], head_info["head"], "high"))
    
    low_induction = results2.classified_heads["low_induction"]
    np.random.shuffle(low_induction)
    for head_info in low_induction[:max_heads_per_model//2]:
        selected_heads_info.append((head_info["layer"], head_info["head"], "low"))
    
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

# %%
# Combine all classified heads for true label creation
combined_classified_heads = {
    "high_induction": [],
    "medium_induction": [],
    "low_induction": []
}

for model_name, classified in all_classified_heads.items():
    for category, heads in classified.items():
        for head_info in heads:
            # Add model prefix to create global head identifier
            global_head_info = head_info.copy()
            combined_classified_heads[category].append(global_head_info)

print(f"Combined classifications:")
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
# Visualize RSA results
import matplotlib.pyplot as plt
import seaborn as sns

# Plot representation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(representation_matrix, 
            yticklabels=head_labels,
            cmap='viridis',
            cbar_kws={'label': 'Distance'})
plt.title('RSA Head Representations (Distance Matrix Features)')
plt.xlabel('Distance Features')
plt.ylabel('Attention Heads')
plt.tight_layout()
plt.show()

# Plot cluster assignments
plt.figure(figsize=(10, 6))
colors = ['red' if label == 1 else 'blue' for label in true_labels]
markers = ['o' if cluster == 0 else 's' for cluster in cluster_labels]

for i, (head, color, marker) in enumerate(zip(head_labels, colors, markers)):
    plt.scatter(i, 0, c=color, marker=marker, s=100, alpha=0.7)

plt.title('Cluster Assignments vs True Labels')
plt.xlabel('Head Index')
plt.ylabel('')
plt.legend(['True: Other', 'True: Induction', 'Cluster 0', 'Cluster 1'])
plt.xticks(range(len(head_labels)), head_labels, rotation=45)
plt.tight_layout()
plt.show()

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
# Demonstration of different plot types for AnalysisResults
print("\nDemonstrating different plot types for AnalysisResults:")
print("Available plot types:")
print("  results.plot('overview')      # Complete overview with 4 subplots")
print("  results.plot('scores')        # Detailed heatmap")
print("  results.plot('heatmap')       # Heatmap with classification boundaries")  
print("  results.plot('distribution')  # Score distributions with statistics")

# You can uncomment these to try different plots:
# results.plot("overview")      
# results.plot("scores", annotate=True)
# results.plot("heatmap", threshold_high=0.8, threshold_medium=0.4)
# results.plot("distribution", bins=25)

# %%