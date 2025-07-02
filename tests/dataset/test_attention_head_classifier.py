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