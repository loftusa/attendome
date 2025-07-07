"""Representational Similarity Analysis (RSA) utilities for attention head analysis."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from dataclasses import dataclass


@dataclass
class RSAResults:
    """Results from RSA analysis."""
    distance_matrices: Dict[str, np.ndarray]
    vectorized_representations: Dict[str, np.ndarray]
    clustering_results: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, float]] = None


class AttentionMapRSA:
    """Representational Similarity Analysis for attention maps."""
    
    def __init__(self, distance_metric: str = "euclidean"):
        """Initialize RSA analyzer.
        
        Args:
            distance_metric: Distance metric for comparing attention maps
        """
        self.distance_metric = distance_metric
    
    def vectorize_attention_maps(
        self, 
        attention_maps: Dict[str, np.ndarray],
        flatten_method: str = "full",
        target_seq_len: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Vectorize attention maps for distance computation.
        
        Args:
            attention_maps: Dictionary of attention maps [num_sequences, seq_len, seq_len]
            flatten_method: How to flatten attention maps ("full", "upper_triangle", "diagonal")
            target_seq_len: Target sequence length for padding/truncation (None = use max)
        
        Returns:
            Dictionary of vectorized attention maps [num_sequences, feature_dim]
        """
        vectorized_maps = {}
        
        # First pass: determine target sequence length if not provided
        if target_seq_len is None:
            max_seq_len = 0
            for attention_tensor in attention_maps.values():
                if isinstance(attention_tensor, dict) and 'data' in attention_tensor:
                    shape = attention_tensor['shape']
                    seq_len = shape[1]
                else:
                    attention_array = np.array(attention_tensor)
                    seq_len = attention_array.shape[1]
                max_seq_len = max(max_seq_len, seq_len)
            target_seq_len = max_seq_len
        
        for head_key, attention_tensor in attention_maps.items():
            if isinstance(attention_tensor, dict) and 'data' in attention_tensor:
                # Handle serialized format
                attention_array = np.array(attention_tensor['data'])
                shape = attention_tensor['shape']
                attention_array = attention_array.reshape(shape)
            else:
                attention_array = np.array(attention_tensor)
            
            # Ensure we have the right shape [num_sequences, seq_len, seq_len]
            if len(attention_array.shape) != 3:
                raise ValueError(f"Expected 3D attention array, got shape {attention_array.shape}")
            
            num_sequences, seq_len, _ = attention_array.shape
            
            # Pad or truncate to target sequence length
            if seq_len != target_seq_len:
                if seq_len < target_seq_len:
                    # Pad with zeros
                    pad_size = target_seq_len - seq_len
                    attention_padded = np.zeros((num_sequences, target_seq_len, target_seq_len))
                    attention_padded[:, :seq_len, :seq_len] = attention_array
                    attention_array = attention_padded
                else:
                    # Truncate
                    attention_array = attention_array[:, :target_seq_len, :target_seq_len]
                
                seq_len = target_seq_len
            
            if flatten_method == "full":
                # Flatten the entire attention matrix
                vectorized = attention_array.reshape(num_sequences, -1)
            
            elif flatten_method == "upper_triangle":
                # Only upper triangle (without diagonal) to avoid redundancy
                vectorized = []
                for seq_idx in range(num_sequences):
                    upper_tri = attention_array[seq_idx][np.triu_indices(seq_len, k=1)]
                    vectorized.append(upper_tri)
                vectorized = np.array(vectorized)
            
            elif flatten_method == "diagonal":
                # Only diagonal elements
                vectorized = []
                for seq_idx in range(num_sequences):
                    diagonal = np.diag(attention_array[seq_idx])
                    vectorized.append(diagonal)
                vectorized = np.array(vectorized)
            
            else:
                raise ValueError(f"Unknown flatten_method: {flatten_method}")
            
            vectorized_maps[head_key] = vectorized
        
        return vectorized_maps
    
    def compute_distance_matrices(
        self,
        vectorized_maps: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute pairwise distance matrices for vectorized attention maps.
        
        Args:
            vectorized_maps: Dictionary of vectorized attention maps
        
        Returns:
            Dictionary of distance matrices [num_sequences, num_sequences]
        """
        distance_matrices = {}
        
        for head_key, vectors in vectorized_maps.items():
            # Compute pairwise distances
            distances = pdist(vectors, metric=self.distance_metric)
            # Convert to square matrix
            distance_matrix = squareform(distances)
            distance_matrices[head_key] = distance_matrix
        
        return distance_matrices
    
    def vectorize_distance_matrices(
        self,
        distance_matrices: Dict[str, np.ndarray],
        use_upper_triangle: bool = True
    ) -> Dict[str, np.ndarray]:
        """Vectorize distance matrices to create head representations.
        
        Args:
            distance_matrices: Dictionary of distance matrices
            use_upper_triangle: Whether to use only upper triangle of symmetric matrix
        
        Returns:
            Dictionary of vectorized distance matrices
        """
        vectorized_matrices = {}
        
        for head_key, dist_matrix in distance_matrices.items():
            if use_upper_triangle:
                # Extract upper triangle (without diagonal)
                upper_tri_indices = np.triu_indices(dist_matrix.shape[0], k=1)
                vectorized = dist_matrix[upper_tri_indices]
            else:
                # Flatten entire matrix
                vectorized = dist_matrix.flatten()
            
            vectorized_matrices[head_key] = vectorized
        
        return vectorized_matrices
    
    def create_head_representation_matrix(
        self,
        vectorized_distance_matrices: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """Create a matrix where each row is a head's vectorized distance matrix.
        
        Args:
            vectorized_distance_matrices: Dictionary of vectorized distance matrices
        
        Returns:
            Tuple of (representation_matrix, head_labels)
        """
        head_labels = sorted(vectorized_distance_matrices.keys())
        representations = [vectorized_distance_matrices[head] for head in head_labels]
        
        # Debug: Check shapes before stacking
        print(f"Debug: Vectorized distance matrix shapes:")
        for i, (label, rep) in enumerate(zip(head_labels, representations)):
            print(f"  {label}: {rep.shape}")
            if i >= 5:  # Only show first 5
                break
        
        representation_matrix = np.vstack(representations)
        
        return representation_matrix, head_labels
    
    def analyze_attention_maps(
        self,
        attention_maps: Dict[str, np.ndarray],
        flatten_method: str = "full",
        use_upper_triangle: bool = True,
        target_seq_len: Optional[int] = None
    ) -> RSAResults:
        """Complete RSA analysis pipeline.
        
        Args:
            attention_maps: Dictionary of attention maps
            flatten_method: How to flatten individual attention maps
            use_upper_triangle: Whether to use upper triangle of distance matrices
            target_seq_len: Target sequence length for normalization
        
        Returns:
            RSAResults object with all analysis outputs
        """
        # Step 1: Vectorize attention maps
        vectorized_maps = self.vectorize_attention_maps(attention_maps, flatten_method, target_seq_len)
        
        # Step 2: Compute distance matrices
        distance_matrices = self.compute_distance_matrices(vectorized_maps)
        
        # Step 3: Vectorize distance matrices for head representations
        vectorized_representations = self.vectorize_distance_matrices(
            distance_matrices, use_upper_triangle
        )
        
        return RSAResults(
            distance_matrices=distance_matrices,
            vectorized_representations=vectorized_representations
        )


class AttentionHeadClusterer:
    """Clustering and evaluation for attention head representations."""
    
    def __init__(self):
        """Initialize clusterer."""
        pass
    
    def cluster_heads(
        self,
        representation_matrix: np.ndarray,
        head_labels: List[str],
        method: str = "kmeans",
        n_clusters: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """Cluster attention heads based on their representations.
        
        Args:
            representation_matrix: Matrix where each row is a head representation
            head_labels: Labels for each head
            method: Clustering method ("kmeans", "hierarchical")
            n_clusters: Number of clusters
            **kwargs: Additional arguments for clustering algorithm
        
        Returns:
            Dictionary with clustering results
        """
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, **kwargs)
            cluster_labels = clusterer.fit_predict(representation_matrix)
            
            results = {
                "method": "kmeans",
                "cluster_labels": cluster_labels,
                "head_labels": head_labels,
                "n_clusters": n_clusters,
                "clusterer": clusterer
            }
            
        elif method == "hierarchical":
            linkage_matrix = linkage(representation_matrix, method=kwargs.get("linkage_method", "ward"))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
            
            results = {
                "method": "hierarchical",
                "cluster_labels": cluster_labels,
                "head_labels": head_labels,
                "n_clusters": n_clusters,
                "linkage_matrix": linkage_matrix
            }
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return results
    
    def evaluate_clustering(
        self,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray,
        representation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate clustering performance.
        
        Args:
            cluster_labels: Predicted cluster labels
            true_labels: True labels (e.g., induction vs non-induction)
            representation_matrix: Head representations
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Adjusted Rand Index (measures agreement with true labels)
        if true_labels is not None:
            metrics["adjusted_rand_score"] = adjusted_rand_score(true_labels, cluster_labels)
        
        # Silhouette score (internal clustering quality)
        if len(np.unique(cluster_labels)) > 1:
            metrics["silhouette_score"] = silhouette_score(representation_matrix, cluster_labels)
        
        return metrics
    
    def create_true_labels_from_classified_heads(
        self,
        head_labels: List[str],
        classified_heads: Dict[str, List[Dict[str, Any]]]
    ) -> np.ndarray:
        """Create true labels array from classified heads.
        
        Args:
            head_labels: List of head identifiers (e.g., "layer_0_head_1")
            classified_heads: Dictionary with high/medium/low induction heads
        
        Returns:
            Array of true labels (1 for induction heads, 0 for others)
        """
        true_labels = np.zeros(len(head_labels))
        
        # Create set of induction head identifiers
        induction_heads = set()
        for head_info in classified_heads.get("high_induction", []):
            head_id = f"layer_{head_info['layer']}_head_{head_info['head']}"
            induction_heads.add(head_id)
        
        # Could also include medium induction heads
        for head_info in classified_heads.get("medium_induction", []):
            head_id = f"layer_{head_info['layer']}_head_{head_info['head']}"
            induction_heads.add(head_id)
        
        # Set labels
        for i, head_label in enumerate(head_labels):
            if head_label in induction_heads:
                true_labels[i] = 1
        
        return true_labels


def run_rsa_experiment(
    attention_maps: Dict[str, np.ndarray],
    classified_heads: Dict[str, List[Dict[str, Any]]],
    distance_metric: str = "euclidean",
    flatten_method: str = "full",
    clustering_method: str = "kmeans",
    n_clusters: int = 2,
    target_seq_len: Optional[int] = None
) -> Dict[str, Any]:
    """Run complete RSA experiment on attention maps.
    
    Args:
        attention_maps: Dictionary of attention maps from models
        classified_heads: Classified heads with induction labels
        distance_metric: Distance metric for RSA
        flatten_method: How to flatten attention maps
        clustering_method: Clustering algorithm
        n_clusters: Number of clusters
    
    Returns:
        Dictionary with complete experiment results
    """
    # Initialize RSA analyzer
    rsa_analyzer = AttentionMapRSA(distance_metric=distance_metric)
    
    # Run RSA analysis
    rsa_results = rsa_analyzer.analyze_attention_maps(
        attention_maps, 
        flatten_method=flatten_method,
        target_seq_len=target_seq_len
    )
    
    # Create head representation matrix
    representation_matrix, head_labels = rsa_analyzer.create_head_representation_matrix(
        rsa_results.vectorized_representations
    )
    
    # Initialize clusterer
    clusterer = AttentionHeadClusterer()
    
    # Run clustering
    clustering_results = clusterer.cluster_heads(
        representation_matrix=representation_matrix,
        head_labels=head_labels,
        method=clustering_method,
        n_clusters=n_clusters
    )
    
    # Create true labels
    true_labels = clusterer.create_true_labels_from_classified_heads(
        head_labels, classified_heads
    )
    
    # Evaluate clustering
    evaluation_metrics = clusterer.evaluate_clustering(
        cluster_labels=clustering_results["cluster_labels"],
        true_labels=true_labels,
        representation_matrix=representation_matrix
    )
    
    return {
        "rsa_results": rsa_results,
        "representation_matrix": representation_matrix,
        "head_labels": head_labels,
        "clustering_results": clustering_results,
        "true_labels": true_labels,
        "evaluation_metrics": evaluation_metrics,
        "experiment_config": {
            "distance_metric": distance_metric,
            "flatten_method": flatten_method,
            "clustering_method": clustering_method,
            "n_clusters": n_clusters
        }
    }