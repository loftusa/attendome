"""Tests for InductionHeadClassifier."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from attendome.dataset.attention_head_classifier import (
    InductionHeadClassifier,
    AnalysisResults,
    ModelConfig
)


class TestInductionHeadClassifier:
    """Test cases for InductionHeadClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for testing."""
        return InductionHeadClassifier(device="cpu")
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock transformer model."""
        model = Mock()
        model.config.num_hidden_layers = 2
        model.config.num_attention_heads = 4
        model.config._attn_implementation = None
        model.config.hidden_size = 64
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.vocab_size = 1000
        return tokenizer
    
    def test_init_default_device(self):
        """Test classifier initialization with default device."""
        classifier = InductionHeadClassifier()
        assert classifier.device in ["cpu", "cuda"]
    
    def test_init_custom_device(self):
        """Test classifier initialization with custom device."""
        classifier = InductionHeadClassifier(device="cpu")
        assert classifier.device == "cpu"
    
    def test_compute_induction_score_shape(self, classifier, mock_model, mock_tokenizer):
        """Test that compute_induction_score returns correct shape."""
        # Mock attention outputs
        attention_tensor = torch.randn(1, 4, 100, 100)  # batch, heads, seq, seq
        mock_model.return_value = Mock(attentions=[attention_tensor, attention_tensor])
        mock_model.eval = Mock()
        
        with patch('torch.no_grad'):
            scores = classifier.compute_induction_score(
                mock_model, 
                mock_tokenizer,
                num_of_samples=32,
                seq_len=50,
                batch_size=16
            )
        
        assert len(scores) == 2  # num_layers
        assert len(scores[0]) == 4  # num_heads
        assert len(scores[1]) == 4  # num_heads
        assert all(isinstance(score, float) for layer_scores in scores for score in layer_scores)
    
    def test_compute_induction_score_parameters(self, classifier, mock_model, mock_tokenizer):
        """Test compute_induction_score with various parameters."""
        attention_tensor = torch.randn(1, 4, 60, 60)
        mock_model.return_value = Mock(attentions=[attention_tensor])
        mock_model.eval = Mock()
        mock_model.config.num_hidden_layers = 1  # Make it consistent with one attention tensor
        
        with patch('torch.no_grad'):
            scores = classifier.compute_induction_score(
                mock_model,
                mock_tokenizer, 
                num_of_samples=10,
                seq_len=30,
                batch_size=5
            )
        
        # Verify model was called with correct inputs
        assert mock_model.call_count > 0
        
        # Verify score structure
        assert len(scores) == 1  # Changed to match mock with 1 layer
        assert all(isinstance(score, float) for score in scores[0])
    
    def test_classify_heads_thresholds(self, classifier):
        """Test head classification with different thresholds."""
        scores = [
            [0.8, 0.3, 0.1, 0.05],  # Layer 0
            [0.6, 0.4, 0.15, 0.02]  # Layer 1 
        ]
        
        classified = classifier.classify_heads(
            scores, 
            high_threshold=0.5,
            medium_threshold=0.2
        )
        
        # Check high induction heads (0.8, 0.6 >= 0.5)
        assert len(classified["high_induction"]) == 2
        high_scores = [h["score"] for h in classified["high_induction"]]
        assert all(score >= 0.5 for score in high_scores)
        
        # Check medium induction heads (0.3, 0.4 >= 0.2 and < 0.5)
        assert len(classified["medium_induction"]) == 2
        medium_scores = [h["score"] for h in classified["medium_induction"]]
        assert all(0.2 <= score < 0.5 for score in medium_scores)
        
        # Check low induction heads (0.1, 0.05, 0.15, 0.02 < 0.2)
        assert len(classified["low_induction"]) == 4
        low_scores = [h["score"] for h in classified["low_induction"]]
        assert all(score < 0.2 for score in low_scores)
    
    def test_classify_heads_structure(self, classifier):
        """Test that classified heads have correct structure."""
        scores = [[0.7, 0.3], [0.1, 0.9]]
        
        classified = classifier.classify_heads(scores)
        
        for category in ["high_induction", "medium_induction", "low_induction"]:
            for head in classified[category]:
                assert "layer" in head
                assert "head" in head
                assert "score" in head
                assert isinstance(head["layer"], int)
                assert isinstance(head["head"], int)
                assert isinstance(head["score"], float)
    
    def test_analyze_model_integration(self, classifier, mock_model, mock_tokenizer):
        """Test complete model analysis workflow."""
        attention_tensor = torch.randn(1, 4, 100, 100)
        mock_model.return_value = Mock(attentions=[attention_tensor, attention_tensor])
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        with patch('torch.no_grad'):
            results = classifier.analyze_model(
                mock_model,
                mock_tokenizer,
                "test-model",
                num_of_samples=16
            )
        
        # Check results structure
        assert isinstance(results, AnalysisResults)
        assert results.model_name == "test-model"
        assert isinstance(results.model_configuration, dict)
        assert isinstance(results.induction_scores, list)
        assert isinstance(results.classified_heads, dict)
        
        # Check model config
        config = results.model_configuration
        assert config["num_layers"] == 2
        assert config["num_heads"] == 4
        assert config["hidden_size"] == 64
        
        # Check that model was moved to device
        mock_model.to.assert_called_with("cpu")
    
    def test_save_random_repetitive_sequence(self, classifier, mock_model, mock_tokenizer):
        """Test saving of random repetitive sequence."""
        attention_tensor = torch.randn(1, 4, 100, 100)
        mock_model.return_value = Mock(attentions=[attention_tensor])
        mock_model.eval = Mock()
        mock_model.config.num_hidden_layers = 1
        
        with patch('torch.no_grad'):
            classifier.compute_induction_score(
                mock_model,
                mock_tokenizer,
                num_of_samples=8,
                save_random_repetitive_sequence=True
            )
        
        # Check that sequence was saved
        assert hasattr(classifier, 'random_repetitive_sequence_')
        assert classifier.random_repetitive_sequence_.shape == (8, 100)  # doubled seq_len
    
    @pytest.mark.parametrize("high_thresh,med_thresh", [
        (0.8, 0.4),
        (0.6, 0.3), 
        (0.5, 0.1)
    ])
    def test_different_thresholds(self, classifier, high_thresh, med_thresh):
        """Test classification with different threshold combinations."""
        scores = [[0.9, 0.5, 0.2, 0.05]]
        
        classified = classifier.classify_heads(
            scores,
            high_threshold=high_thresh,
            medium_threshold=med_thresh
        )
        
        # Verify threshold logic
        for head in classified["high_induction"]:
            assert head["score"] >= high_thresh
        
        for head in classified["medium_induction"]:
            assert med_thresh <= head["score"] < high_thresh
            
        for head in classified["low_induction"]:
            assert head["score"] < med_thresh
    
    def test_empty_scores(self, classifier):
        """Test handling of empty score lists."""
        scores = []
        
        classified = classifier.classify_heads(scores)
        
        assert len(classified["high_induction"]) == 0
        assert len(classified["medium_induction"]) == 0  
        assert len(classified["low_induction"]) == 0
    
    def test_attention_implementation_restoration(self, classifier, mock_model, mock_tokenizer):
        """Test that attention implementation is properly restored."""
        original_impl = "flash_attention_2"
        mock_model.config._attn_implementation = original_impl
        
        attention_tensor = torch.randn(1, 4, 100, 100)
        mock_model.return_value = Mock(attentions=[attention_tensor])
        mock_model.eval = Mock()
        mock_model.config.num_hidden_layers = 1
        
        with patch('torch.no_grad'):
            classifier.compute_induction_score(mock_model, mock_tokenizer, num_of_samples=4)
        
        # Check that original implementation was restored
        assert mock_model.config._attn_implementation == original_impl
    
    def test_extract_attention_maps_basic(self, classifier, mock_model, mock_tokenizer):
        """Test basic attention map extraction."""
        # Mock sequences
        sequences = torch.randint(0, 1000, (3, 10))  # 3 sequences, length 10
        
        # Mock attention outputs
        attention_tensor = torch.randn(3, 4, 10, 10)  # batch, heads, seq, seq
        mock_model.return_value = Mock(attentions=[attention_tensor])
        mock_model.eval = Mock()
        mock_model.config.num_hidden_layers = 1
        mock_model.config.num_attention_heads = 4
        
        with patch('torch.no_grad'):
            attention_maps = classifier.extract_attention_maps(
                mock_model, mock_tokenizer, sequences
            )
        
        # Should extract all heads from all layers by default
        assert len(attention_maps) == 4  # 1 layer * 4 heads
        
        for head_idx in range(4):
            key = f"layer_0_head_{head_idx}"
            assert key in attention_maps
            assert attention_maps[key].shape == (3, 10, 10)  # 3 sequences, 10x10 attention
    
    def test_extract_attention_maps_specific_layers_heads(self, classifier, mock_model, mock_tokenizer):
        """Test attention map extraction for specific layers and heads."""
        sequences = torch.randint(0, 1000, (2, 8))
        
        # Mock 2 layers
        attention_tensor1 = torch.randn(2, 4, 8, 8)
        attention_tensor2 = torch.randn(2, 4, 8, 8)
        mock_model.return_value = Mock(attentions=[attention_tensor1, attention_tensor2])
        mock_model.config.num_hidden_layers = 2
        mock_model.config.num_attention_heads = 4
        
        with patch('torch.no_grad'):
            attention_maps = classifier.extract_attention_maps(
                mock_model, mock_tokenizer, sequences,
                layer_indices=[0, 1],  # Specific layers
                head_indices=[1, 3]    # Specific heads
            )
        
        # Should only extract specified layer/head combinations
        expected_keys = ["layer_0_head_1", "layer_0_head_3", "layer_1_head_1", "layer_1_head_3"]
        assert len(attention_maps) == 4
        for key in expected_keys:
            assert key in attention_maps
            assert attention_maps[key].shape == (2, 8, 8)
    
    def test_extract_attention_maps_batching(self, classifier, mock_model, mock_tokenizer):
        """Test attention map extraction with batching."""
        sequences = torch.randint(0, 1000, (5, 6))  # 5 sequences
        
        attention_tensor = torch.randn(2, 2, 6, 6)  # Batch size 2, 2 heads
        mock_model.side_effect = [
            Mock(attentions=[attention_tensor]),  # First batch
            Mock(attentions=[attention_tensor]),  # Second batch  
            Mock(attentions=[torch.randn(1, 2, 6, 6)])  # Last batch (size 1)
        ]
        mock_model.config.num_attention_heads = 2
        
        with patch('torch.no_grad'):
            attention_maps = classifier.extract_attention_maps(
                mock_model, mock_tokenizer, sequences,
                layer_indices=[0], head_indices=[0, 1],
                batch_size=2
            )
        
        # Should concatenate batches correctly
        assert len(attention_maps) == 2
        assert attention_maps["layer_0_head_0"].shape == (5, 6, 6)
        assert attention_maps["layer_0_head_1"].shape == (5, 6, 6)
    
    def test_extract_induction_attention_maps(self, classifier, mock_model, mock_tokenizer):
        """Test extraction of induction-specific attention maps."""
        attention_tensor = torch.randn(3, 4, 20, 20)  # 3 sequences, 4 heads, 20x20 (2*10)
        mock_model.return_value = Mock(attentions=[attention_tensor])
        mock_model.config.num_hidden_layers = 1
        mock_model.config.num_attention_heads = 4
        
        with patch('torch.no_grad'):
            attention_maps = classifier.extract_induction_attention_maps(
                mock_model, mock_tokenizer,
                num_sequences=3, seq_len=10,  # Will create 2*10=20 length sequences
                layer_indices=[0], head_indices=[1, 2]
            )
        
        # Should extract specified heads
        assert len(attention_maps) == 2
        assert "layer_0_head_1" in attention_maps
        assert "layer_0_head_2" in attention_maps
        
        for key in attention_maps:
            assert attention_maps[key].shape == (3, 20, 20)
    
    def test_analyze_model_with_attention_maps(self, classifier):
        """Test analyze_model with attention map extraction enabled."""
        # Create a proper mock model
        mock_model = Mock()
        mock_model.config.num_hidden_layers = 2
        mock_model.config.num_attention_heads = 3
        mock_model.config.hidden_size = 64
        mock_model.to.return_value = mock_model  # Mock the .to(device) call
        
        mock_tokenizer = Mock()
        
        # Mock both score computation and attention extraction
        with patch.object(classifier, 'compute_induction_score') as mock_scores, \
             patch.object(classifier, 'extract_induction_attention_maps') as mock_extract:
            
            mock_scores.return_value = [[0.8, 0.3, 0.1], [0.6, 0.4, 0.15]]
            mock_extract.return_value = {
                "layer_0_head_0": torch.randn(5, 10, 10),
                "layer_1_head_1": torch.randn(5, 10, 10)
            }
            
            results = classifier.analyze_model(
                mock_model, mock_tokenizer, "test_model",
                extract_attention_maps=True,
                attention_map_config={
                    "num_sequences": 5,
                    "seq_len": 10,
                    "layer_indices": [0, 1],
                    "head_indices": [0, 1]
                }
            )
        
        # Check that attention maps were extracted and serialized
        assert results.attention_maps is not None
        assert "layer_0_head_0" in results.attention_maps
        assert "layer_1_head_1" in results.attention_maps
        
        # Check serialization format
        for key, attention_data in results.attention_maps.items():
            assert "data" in attention_data
            assert "shape" in attention_data
            assert attention_data["shape"] == [5, 10, 10]
    
    def test_analyze_model_without_attention_maps(self, classifier):
        """Test analyze_model without attention map extraction."""
        # Create a proper mock model
        mock_model = Mock()
        mock_model.config.num_hidden_layers = 1
        mock_model.config.num_attention_heads = 2
        mock_model.config.hidden_size = 32
        mock_model.to.return_value = mock_model  # Mock the .to(device) call
        
        mock_tokenizer = Mock()
        
        with patch.object(classifier, 'compute_induction_score') as mock_scores:
            mock_scores.return_value = [[0.7, 0.2]]
            
            results = classifier.analyze_model(
                mock_model, mock_tokenizer, "test_model",
                extract_attention_maps=False
            )
        
        # Check that no attention maps were extracted
        assert results.attention_maps is None


class TestAnalysisResults:
    """Test cases for AnalysisResults Pydantic model."""
    
    def test_analysis_results_creation(self):
        """Test creation of AnalysisResults object."""
        results = AnalysisResults(
            model_name="gpt2",
            model_configuration={"num_layers": 12, "num_heads": 12, "hidden_size": 768},
            induction_scores=[[0.8, 0.3], [0.6, 0.4]],
            classified_heads={
                "high_induction": [{"layer": 0, "head": 0, "score": 0.8}],
                "medium_induction": [],
                "low_induction": [{"layer": 0, "head": 1, "score": 0.3}]
            },
            analysis_params={"num_of_samples": 100, "seq_len": 50}
        )
        
        assert results.model_name == "gpt2"
        assert results.model_configuration["num_layers"] == 12
        assert len(results.induction_scores) == 2
        assert len(results.classified_heads["high_induction"]) == 1
        assert results.attention_maps is None  # Optional field
    
    def test_analysis_results_with_attention_maps(self):
        """Test AnalysisResults with attention maps."""
        attention_maps = {
            "layer_0_head_0": {
                "data": [[[0.1, 0.2], [0.3, 0.4]]],
                "shape": [1, 2, 2]
            }
        }
        
        results = AnalysisResults(
            model_name="test",
            model_configuration={"num_layers": 1, "num_heads": 1, "hidden_size": 64},
            induction_scores=[[0.5]],
            classified_heads={"high_induction": [], "medium_induction": [], "low_induction": []},
            analysis_params={},
            attention_maps=attention_maps
        )
        
        assert results.attention_maps is not None
        assert "layer_0_head_0" in results.attention_maps
        assert results.attention_maps["layer_0_head_0"]["shape"] == [1, 2, 2]
    
    def test_analysis_results_plot_method_import_error(self):
        """Test plot method handles import errors gracefully."""
        results = AnalysisResults(
            model_name="test",
            model_configuration={"num_layers": 1, "num_heads": 1, "hidden_size": 64},
            induction_scores=[[0.5]],
            classified_heads={"high_induction": [], "medium_induction": [], "low_induction": []},
            analysis_params={}
        )
        
        # Mock import failure
        with patch('builtins.__import__', side_effect=ImportError("No matplotlib")):
            # Should not raise an exception
            results.plot("overview")
    
    def test_analysis_results_plot_unknown_type(self):
        """Test plot method with unknown plot type."""
        results = AnalysisResults(
            model_name="test",
            model_configuration={"num_layers": 1, "num_heads": 1, "hidden_size": 64},
            induction_scores=[[0.5]],
            classified_heads={"high_induction": [], "medium_induction": [], "low_induction": []},
            analysis_params={}
        )
        
        # Should handle unknown plot type gracefully
        with patch('builtins.print') as mock_print:
            results.plot("unknown_type")
            mock_print.assert_called()  # Should print error message