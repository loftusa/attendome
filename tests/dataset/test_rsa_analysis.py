"""Tests for RSA analysis functionality."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from attendome.dataset.rsa_analysis import (
    AttentionMapRSA,
    AttentionHeadClusterer,
    RSAResults,
    run_rsa_experiment
)


class TestAttentionMapRSA:
    """Test cases for AttentionMapRSA."""
    
    @pytest.fixture
    def rsa_analyzer(self):
        """Create an RSA analyzer instance for testing."""
        return AttentionMapRSA(distance_metric="euclidean")
    
    @pytest.fixture
    def sample_attention_maps(self):
        """Create sample attention maps for testing."""
        np.random.seed(42)  # For reproducible tests
        return {
            "head_1": np.random.rand(5, 10, 10),  # 5 sequences, 10x10 attention
            "head_2": np.random.rand(5, 10, 10),
            "head_3": np.random.rand(5, 8, 8),    # Different sequence length
        }
    
    @pytest.fixture
    def sample_serialized_attention_maps(self):
        """Create sample serialized attention maps (like from JSON)."""
        np.random.seed(42)
        data = np.random.rand(5, 10, 10)
        return {
            "head_1": {
                "data": data.tolist(),
                "shape": [5, 10, 10]
            }
        }
    
    def test_init_default_metric(self):
        """Test RSA analyzer initialization with default metric."""
        analyzer = AttentionMapRSA()
        assert analyzer.distance_metric == "euclidean"
    
    def test_init_custom_metric(self):
        """Test RSA analyzer initialization with custom metric."""
        analyzer = AttentionMapRSA(distance_metric="cosine")
        assert analyzer.distance_metric == "cosine"
    
    def test_vectorize_attention_maps_full(self, rsa_analyzer, sample_attention_maps):
        """Test full vectorization of attention maps."""
        vectorized = rsa_analyzer.vectorize_attention_maps(
            sample_attention_maps, 
            flatten_method="full"
        )
        
        assert len(vectorized) == 3
        assert "head_1" in vectorized
        assert "head_2" in vectorized
        assert "head_3" in vectorized
        
        # Check shapes - should be normalized to max sequence length (10)
        assert vectorized["head_1"].shape == (5, 100)  # 10*10
        assert vectorized["head_2"].shape == (5, 100)
        assert vectorized["head_3"].shape == (5, 100)  # Padded from 8x8 to 10x10
    
    def test_vectorize_attention_maps_upper_triangle(self, rsa_analyzer, sample_attention_maps):
        """Test upper triangle vectorization of attention maps."""
        vectorized = rsa_analyzer.vectorize_attention_maps(
            sample_attention_maps,
            flatten_method="upper_triangle"
        )
        
        # Upper triangle of 10x10 matrix (excluding diagonal) = 45 elements
        assert vectorized["head_1"].shape == (5, 45)
        assert vectorized["head_2"].shape == (5, 45)
        assert vectorized["head_3"].shape == (5, 45)
    
    def test_vectorize_attention_maps_diagonal(self, rsa_analyzer, sample_attention_maps):
        """Test diagonal vectorization of attention maps."""
        vectorized = rsa_analyzer.vectorize_attention_maps(
            sample_attention_maps,
            flatten_method="diagonal"
        )
        
        # Diagonal of 10x10 matrix = 10 elements
        assert vectorized["head_1"].shape == (5, 10)
        assert vectorized["head_2"].shape == (5, 10)
        assert vectorized["head_3"].shape == (5, 10)
    
    def test_vectorize_attention_maps_invalid_method(self, rsa_analyzer, sample_attention_maps):
        """Test vectorization with invalid flatten method."""
        with pytest.raises(ValueError, match="Unknown flatten_method"):
            rsa_analyzer.vectorize_attention_maps(
                sample_attention_maps,
                flatten_method="invalid_method"
            )
    
    def test_vectorize_attention_maps_custom_target_length(self, rsa_analyzer):
        """Test vectorization with custom target sequence length."""
        attention_maps = {
            "head_1": np.random.rand(3, 5, 5),  # 5x5 attention
        }
        
        vectorized = rsa_analyzer.vectorize_attention_maps(
            attention_maps,
            flatten_method="full",
            target_seq_len=8  # Force to 8x8
        )
        
        assert vectorized["head_1"].shape == (3, 64)  # 8*8
    
    def test_vectorize_attention_maps_serialized_format(self, rsa_analyzer, sample_serialized_attention_maps):
        """Test vectorization with serialized attention maps."""
        vectorized = rsa_analyzer.vectorize_attention_maps(
            sample_serialized_attention_maps,
            flatten_method="full"
        )
        
        assert len(vectorized) == 1
        assert vectorized["head_1"].shape == (5, 100)  # 10*10
    
    def test_vectorize_attention_maps_wrong_shape(self, rsa_analyzer):
        """Test vectorization with wrong input shape."""
        attention_maps = {
            "head_1": np.random.rand(10, 10),  # 2D instead of 3D
        }
        
        with pytest.raises(ValueError, match="Expected 3D attention array"):
            rsa_analyzer.vectorize_attention_maps(attention_maps)
    
    def test_compute_distance_matrices(self, rsa_analyzer):
        """Test computation of distance matrices."""
        vectorized_maps = {
            "head_1": np.random.rand(5, 20),
            "head_2": np.random.rand(5, 20),
        }
        
        distance_matrices = rsa_analyzer.compute_distance_matrices(vectorized_maps)
        
        assert len(distance_matrices) == 2
        assert distance_matrices["head_1"].shape == (5, 5)
        assert distance_matrices["head_2"].shape == (5, 5)
        
        # Check symmetry
        assert np.allclose(distance_matrices["head_1"], distance_matrices["head_1"].T)
        
        # Check diagonal is zero (distance from point to itself)
        assert np.allclose(np.diag(distance_matrices["head_1"]), 0)
    
    def test_vectorize_distance_matrices_upper_triangle(self, rsa_analyzer):
        """Test vectorization of distance matrices using upper triangle."""
        distance_matrices = {
            "head_1": np.random.rand(5, 5),
            "head_2": np.random.rand(4, 4),
        }
        
        vectorized = rsa_analyzer.vectorize_distance_matrices(
            distance_matrices, 
            use_upper_triangle=True
        )
        
        # Upper triangle of 5x5 matrix (excluding diagonal) = 10 elements
        assert vectorized["head_1"].shape == (10,)
        # Upper triangle of 4x4 matrix (excluding diagonal) = 6 elements  
        assert vectorized["head_2"].shape == (6,)
    
    def test_vectorize_distance_matrices_full(self, rsa_analyzer):
        """Test vectorization of distance matrices using full matrix."""
        distance_matrices = {
            "head_1": np.random.rand(5, 5),
        }
        
        vectorized = rsa_analyzer.vectorize_distance_matrices(
            distance_matrices,
            use_upper_triangle=False
        )
        
        assert vectorized["head_1"].shape == (25,)  # 5*5
    
    def test_create_head_representation_matrix(self, rsa_analyzer):
        """Test creation of head representation matrix."""
        vectorized_distance_matrices = {
            "head_b": np.random.rand(10),
            "head_a": np.random.rand(10),
            "head_c": np.random.rand(10),
        }
        
        representation_matrix, head_labels = rsa_analyzer.create_head_representation_matrix(
            vectorized_distance_matrices
        )
        
        assert representation_matrix.shape == (3, 10)
        assert len(head_labels) == 3
        assert head_labels == ["head_a", "head_b", "head_c"]  # Should be sorted
    
    def test_analyze_attention_maps_full_pipeline(self, rsa_analyzer, sample_attention_maps):
        """Test complete RSA analysis pipeline."""
        results = rsa_analyzer.analyze_attention_maps(
            sample_attention_maps,
            flatten_method="upper_triangle",
            use_upper_triangle=True
        )
        
        assert isinstance(results, RSAResults)
        assert len(results.distance_matrices) == 3
        assert len(results.vectorized_representations) == 3
        
        # Check that all heads have same vectorized representation length
        rep_lengths = [rep.shape[0] for rep in results.vectorized_representations.values()]
        assert len(set(rep_lengths)) == 1  # All same length


class TestAttentionHeadClusterer:
    """Test cases for AttentionHeadClusterer."""
    
    @pytest.fixture
    def clusterer(self):
        """Create a clusterer instance for testing."""
        return AttentionHeadClusterer()
    
    @pytest.fixture
    def sample_representation_matrix(self):
        """Create sample representation matrix for testing."""
        np.random.seed(42)
        return np.random.rand(6, 20)  # 6 heads, 20 features each
    
    @pytest.fixture
    def sample_head_labels(self):
        """Create sample head labels."""
        return ["head_1", "head_2", "head_3", "head_4", "head_5", "head_6"]
    
    @pytest.fixture
    def sample_classified_heads(self):
        """Create sample classified heads for testing."""
        return {
            "high_induction": [
                {"layer": 0, "head": 1, "score": 0.8},
                {"layer": 1, "head": 0, "score": 0.9}
            ],
            "medium_induction": [
                {"layer": 0, "head": 2, "score": 0.6}
            ],
            "low_induction": [
                {"layer": 1, "head": 1, "score": 0.2},
                {"layer": 1, "head": 2, "score": 0.1}
            ]
        }
    
    def test_cluster_heads_kmeans(self, clusterer, sample_representation_matrix, sample_head_labels):
        """Test K-means clustering of attention heads."""
        results = clusterer.cluster_heads(
            sample_representation_matrix,
            sample_head_labels,
            method="kmeans",
            n_clusters=2
        )
        
        assert results["method"] == "kmeans"
        assert len(results["cluster_labels"]) == 6
        assert results["head_labels"] == sample_head_labels
        assert results["n_clusters"] == 2
        assert "clusterer" in results
        
        # Check cluster labels are in valid range
        assert all(0 <= label <= 1 for label in results["cluster_labels"])
    
    def test_cluster_heads_hierarchical(self, clusterer, sample_representation_matrix, sample_head_labels):
        """Test hierarchical clustering of attention heads."""
        results = clusterer.cluster_heads(
            sample_representation_matrix,
            sample_head_labels,
            method="hierarchical",
            n_clusters=3
        )
        
        assert results["method"] == "hierarchical"
        assert len(results["cluster_labels"]) == 6
        assert results["n_clusters"] == 3
        assert "linkage_matrix" in results
        
        # Check cluster labels are in valid range
        assert all(1 <= label <= 3 for label in results["cluster_labels"])  # hierarchical uses 1-indexed
    
    def test_cluster_heads_invalid_method(self, clusterer, sample_representation_matrix, sample_head_labels):
        """Test clustering with invalid method."""
        with pytest.raises(ValueError, match="Unknown clustering method"):
            clusterer.cluster_heads(
                sample_representation_matrix,
                sample_head_labels,
                method="invalid_method"
            )
    
    def test_evaluate_clustering_with_true_labels(self, clusterer, sample_representation_matrix):
        """Test clustering evaluation with true labels."""
        cluster_labels = np.array([0, 0, 1, 1, 0, 1])
        true_labels = np.array([1, 1, 0, 0, 1, 0])
        
        metrics = clusterer.evaluate_clustering(
            cluster_labels, true_labels, sample_representation_matrix
        )
        
        assert "adjusted_rand_score" in metrics
        assert "silhouette_score" in metrics
        assert -1 <= metrics["adjusted_rand_score"] <= 1
        assert -1 <= metrics["silhouette_score"] <= 1
    
    def test_evaluate_clustering_without_true_labels(self, clusterer, sample_representation_matrix):
        """Test clustering evaluation without true labels."""
        cluster_labels = np.array([0, 0, 1, 1, 0, 1])
        
        metrics = clusterer.evaluate_clustering(
            cluster_labels, None, sample_representation_matrix
        )
        
        assert "adjusted_rand_score" not in metrics
        assert "silhouette_score" in metrics
    
    def test_evaluate_clustering_single_cluster(self, clusterer, sample_representation_matrix):
        """Test clustering evaluation with single cluster."""
        cluster_labels = np.array([0, 0, 0, 0, 0, 0])  # All same cluster
        true_labels = np.array([1, 1, 0, 0, 1, 0])
        
        metrics = clusterer.evaluate_clustering(
            cluster_labels, true_labels, sample_representation_matrix
        )
        
        assert "adjusted_rand_score" in metrics
        assert "silhouette_score" not in metrics  # Cannot compute for single cluster
    
    def test_create_true_labels_from_classified_heads(self, clusterer, sample_classified_heads):
        """Test creation of true labels from classified heads."""
        head_labels = [
            "layer_0_head_1",  # high induction
            "layer_1_head_0",  # high induction  
            "layer_0_head_2",  # medium induction
            "layer_1_head_1",  # low induction
            "layer_1_head_2",  # low induction
            "layer_2_head_0"   # not in classified heads
        ]
        
        true_labels = clusterer.create_true_labels_from_classified_heads(
            head_labels, sample_classified_heads
        )
        
        expected = np.array([1, 1, 1, 0, 0, 0])  # First 3 are induction (high + medium)
        assert np.array_equal(true_labels, expected)
    
    def test_create_true_labels_empty_classified_heads(self, clusterer):
        """Test creation of true labels with empty classified heads."""
        head_labels = ["layer_0_head_1", "layer_1_head_0"]
        classified_heads = {
            "high_induction": [],
            "medium_induction": [],
            "low_induction": []
        }
        
        true_labels = clusterer.create_true_labels_from_classified_heads(
            head_labels, classified_heads
        )
        
        expected = np.array([0, 0])  # All non-induction
        assert np.array_equal(true_labels, expected)


class TestRunRSAExperiment:
    """Test cases for the complete RSA experiment function."""
    
    @pytest.fixture
    def sample_attention_maps(self):
        """Create sample attention maps for testing."""
        np.random.seed(42)
        return {
            "model1_layer_0_head_1": np.random.rand(4, 8, 8),
            "model1_layer_1_head_0": np.random.rand(4, 8, 8),
            "model2_layer_0_head_1": np.random.rand(4, 8, 8),
        }
    
    @pytest.fixture
    def sample_classified_heads(self):
        """Create sample classified heads."""
        return {
            "high_induction": [
                {"layer": 0, "head": 1, "score": 0.8},
                {"layer": 1, "head": 0, "score": 0.9}
            ],
            "medium_induction": [],
            "low_induction": [
                {"layer": 0, "head": 1, "score": 0.2}  # Different model, same layer/head
            ]
        }
    
    def test_run_rsa_experiment_complete(self, sample_attention_maps, sample_classified_heads):
        """Test complete RSA experiment pipeline."""
        results = run_rsa_experiment(
            attention_maps=sample_attention_maps,
            classified_heads=sample_classified_heads,
            distance_metric="euclidean",
            flatten_method="upper_triangle",
            clustering_method="kmeans",
            n_clusters=2
        )
        
        # Check all expected keys are present
        expected_keys = [
            "rsa_results", "representation_matrix", "head_labels",
            "clustering_results", "true_labels", "evaluation_metrics",
            "experiment_config"
        ]
        for key in expected_keys:
            assert key in results
        
        # Check shapes and types
        assert isinstance(results["rsa_results"], RSAResults)
        assert results["representation_matrix"].shape[0] == 3  # 3 heads
        assert len(results["head_labels"]) == 3
        assert len(results["true_labels"]) == 3
        assert len(results["clustering_results"]["cluster_labels"]) == 3
        
        # Check experiment config
        config = results["experiment_config"]
        assert config["distance_metric"] == "euclidean"
        assert config["flatten_method"] == "upper_triangle"
        assert config["clustering_method"] == "kmeans"
        assert config["n_clusters"] == 2
    
    def test_run_rsa_experiment_hierarchical_clustering(self, sample_attention_maps, sample_classified_heads):
        """Test RSA experiment with hierarchical clustering."""
        results = run_rsa_experiment(
            attention_maps=sample_attention_maps,
            classified_heads=sample_classified_heads,
            clustering_method="hierarchical",
            n_clusters=2
        )
        
        assert results["clustering_results"]["method"] == "hierarchical"
        assert "linkage_matrix" in results["clustering_results"]
    
    def test_run_rsa_experiment_different_metrics(self, sample_attention_maps, sample_classified_heads):
        """Test RSA experiment with different distance metrics and flatten methods."""
        # Test cosine distance
        results_cosine = run_rsa_experiment(
            attention_maps=sample_attention_maps,
            classified_heads=sample_classified_heads,
            distance_metric="cosine"
        )
        assert results_cosine["experiment_config"]["distance_metric"] == "cosine"
        
        # Test diagonal flatten method
        results_diagonal = run_rsa_experiment(
            attention_maps=sample_attention_maps,
            classified_heads=sample_classified_heads,
            flatten_method="diagonal"
        )
        assert results_diagonal["experiment_config"]["flatten_method"] == "diagonal"
    
    @patch('attendome.dataset.rsa_analysis.AttentionMapRSA')
    @patch('attendome.dataset.rsa_analysis.AttentionHeadClusterer')
    def test_run_rsa_experiment_mock_components(self, mock_clusterer_class, mock_rsa_class,
                                               sample_attention_maps, sample_classified_heads):
        """Test RSA experiment with mocked components to verify call flow."""
        # Mock RSA analyzer
        mock_rsa = Mock()
        mock_rsa.analyze_attention_maps.return_value = RSAResults(
            distance_matrices={}, vectorized_representations={}
        )
        mock_rsa.create_head_representation_matrix.return_value = (
            np.random.rand(3, 10), ["head1", "head2", "head3"]
        )
        mock_rsa_class.return_value = mock_rsa
        
        # Mock clusterer
        mock_clusterer = Mock()
        mock_clusterer.cluster_heads.return_value = {
            "cluster_labels": np.array([0, 1, 0]),
            "method": "kmeans"
        }
        mock_clusterer.create_true_labels_from_classified_heads.return_value = np.array([1, 0, 1])
        mock_clusterer.evaluate_clustering.return_value = {"adjusted_rand_score": 0.5}
        mock_clusterer_class.return_value = mock_clusterer
        
        # Run experiment
        results = run_rsa_experiment(
            attention_maps=sample_attention_maps,
            classified_heads=sample_classified_heads
        )
        
        # Verify components were called
        mock_rsa_class.assert_called_once_with(distance_metric="euclidean")
        mock_rsa.analyze_attention_maps.assert_called_once()
        mock_clusterer.cluster_heads.assert_called_once()
        mock_clusterer.evaluate_clustering.assert_called_once()


class TestRSAResultsDataClass:
    """Test cases for RSAResults dataclass."""
    
    def test_rsa_results_creation(self):
        """Test creation of RSAResults object."""
        distance_matrices = {"head1": np.random.rand(5, 5)}
        vectorized_representations = {"head1": np.random.rand(10)}
        
        results = RSAResults(
            distance_matrices=distance_matrices,
            vectorized_representations=vectorized_representations
        )
        
        assert results.distance_matrices == distance_matrices
        assert results.vectorized_representations == vectorized_representations
        assert results.clustering_results is None
        assert results.evaluation_metrics is None
    
    def test_rsa_results_with_optional_fields(self):
        """Test RSAResults with optional fields."""
        clustering_results = {"method": "kmeans"}
        evaluation_metrics = {"adjusted_rand_score": 0.8}
        
        results = RSAResults(
            distance_matrices={},
            vectorized_representations={},
            clustering_results=clustering_results,
            evaluation_metrics=evaluation_metrics
        )
        
        assert results.clustering_results == clustering_results
        assert results.evaluation_metrics == evaluation_metrics


# Integration tests with realistic data
class TestRSAIntegration:
    """Integration tests with more realistic scenarios."""
    
    def test_rsa_with_different_sequence_lengths(self):
        """Test RSA analysis with attention maps of different sequence lengths."""
        attention_maps = {
            "head_1": np.random.rand(3, 10, 10),
            "head_2": np.random.rand(3, 15, 15),  # Different length
            "head_3": np.random.rand(3, 8, 8),    # Another different length
        }
        
        classified_heads = {
            "high_induction": [{"layer": 0, "head": 1, "score": 0.8}],
            "medium_induction": [],
            "low_induction": [
                {"layer": 0, "head": 2, "score": 0.2},
                {"layer": 0, "head": 3, "score": 0.1}
            ]
        }
        
        # Should not raise an error due to different sequence lengths
        results = run_rsa_experiment(
            attention_maps=attention_maps,
            classified_heads=classified_heads,
            target_seq_len=12  # Force normalization to 12x12
        )
        
        # All heads should have same representation size after normalization
        rep_matrix = results["representation_matrix"]
        assert rep_matrix.shape[0] == 3  # 3 heads
        # All rows should have same length (same feature dimensionality)
        assert len(set([rep_matrix[i].shape[0] for i in range(rep_matrix.shape[0])])) == 1
    
    def test_rsa_with_minimal_data(self):
        """Test RSA analysis with minimal valid data."""
        attention_maps = {
            "head_1": np.random.rand(3, 3, 3),  # 3 sequences (minimum for silhouette with 2 clusters)
            "head_2": np.random.rand(3, 3, 3),
            "head_3": np.random.rand(3, 3, 3),
        }
        
        classified_heads = {
            "high_induction": [{"layer": 0, "head": 1, "score": 0.8}],
            "medium_induction": [],
            "low_induction": [
                {"layer": 0, "head": 2, "score": 0.2},
                {"layer": 0, "head": 3, "score": 0.1}
            ]
        }
        
        results = run_rsa_experiment(
            attention_maps=attention_maps,
            classified_heads=classified_heads,
            n_clusters=2
        )
        
        assert results["representation_matrix"].shape[0] == 3
        assert len(results["head_labels"]) == 3
        assert len(results["true_labels"]) == 3


if __name__ == "__main__":
    pytest.main([__file__])