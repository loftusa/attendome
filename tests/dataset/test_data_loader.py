"""Tests for ModelLoader."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from attendome.dataset.data_loader import ModelLoader, ModelInfo


class TestModelLoader:
    """Test cases for ModelLoader."""
    
    @pytest.fixture
    def loader(self):
        """Create a ModelLoader instance for testing."""
        return ModelLoader(device="cpu", torch_dtype=torch.float32)
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.to = Mock(return_value=model)
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        return tokenizer
    
    def test_init_default_device_cuda_available(self):
        """Test initialization when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            loader = ModelLoader()
            assert loader.device == "cuda"
            assert loader.torch_dtype == torch.float16
    
    def test_init_default_device_cuda_unavailable(self):
        """Test initialization when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            loader = ModelLoader()
            assert loader.device == "cpu"
            assert loader.torch_dtype == torch.float32
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        loader = ModelLoader(device="cpu", torch_dtype=torch.float16)
        assert loader.device == "cpu"
        assert loader.torch_dtype == torch.float16
        assert loader._loaded_models == {}
    
    @patch('attendome.dataset.data_loader.AutoTokenizer')
    @patch('attendome.dataset.data_loader.AutoModel')
    def test_load_model_success(self, mock_auto_model, mock_auto_tokenizer, loader, mock_model, mock_tokenizer):
        """Test successful model loading."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        model, tokenizer = loader.load_model("gpt2")
        
        # Verify tokenizer loading
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "gpt2", 
            trust_remote_code=False
        )
        
        # Verify model loading
        mock_auto_model.from_pretrained.assert_called_once_with(
            "gpt2",
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=False
        )
        
        # Verify padding token was set
        assert mock_tokenizer.pad_token == "</s>"
        
        # Verify model was moved to CPU
        mock_model.to.assert_called_once_with("cpu")
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
    
    @patch('attendome.dataset.data_loader.AutoTokenizer')
    @patch('attendome.dataset.data_loader.AutoModel')
    def test_load_model_with_caching(self, mock_auto_model, mock_auto_tokenizer, loader, mock_model, mock_tokenizer):
        """Test model loading with caching enabled."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Load model with caching
        model1, tokenizer1 = loader.load_model("gpt2", cache_model=True)
        
        # Load same model again - should return cached version
        model2, tokenizer2 = loader.load_model("gpt2", cache_model=True)
        
        # Verify model was only loaded once
        assert mock_auto_model.from_pretrained.call_count == 1
        assert mock_auto_tokenizer.from_pretrained.call_count == 1
        
        # Verify same instances returned
        assert model1 is model2
        assert tokenizer1 is tokenizer2
        
        # Verify model is cached
        assert "gpt2" in loader._loaded_models
    
    @patch('attendome.dataset.data_loader.AutoTokenizer')
    @patch('attendome.dataset.data_loader.AutoModel')
    def test_load_model_trust_remote_code(self, mock_auto_model, mock_auto_tokenizer, loader):
        """Test model loading with trust_remote_code=True."""
        mock_auto_tokenizer.from_pretrained.return_value = Mock(pad_token="<pad>", eos_token="</s>")
        mock_auto_model.from_pretrained.return_value = Mock()
        
        loader.load_model("custom-model", trust_remote_code=True)
        
        # Verify trust_remote_code was passed
        mock_auto_tokenizer.from_pretrained.assert_called_with(
            "custom-model",
            trust_remote_code=True
        )
        mock_auto_model.from_pretrained.assert_called_with(
            "custom-model",
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
    
    def test_get_supported_models(self, loader):
        """Test getting list of supported models."""
        models = loader.get_supported_models()
        
        # Check that it returns a list of strings
        assert isinstance(models, list)
        assert all(isinstance(model, str) for model in models)
        
        # Check some expected models are present
        expected_models = ["gpt2", "gpt2-medium", "distilgpt2"]
        for model in expected_models:
            assert model in models
    
    @patch('attendome.dataset.data_loader.ModelLoader.load_model')
    def test_batch_load_models_success(self, mock_load_model, loader):
        """Test successful batch loading of models."""
        # Mock successful loading
        mock_load_model.side_effect = [
            (Mock(), Mock()),  # gpt2
            (Mock(), Mock()),  # gpt2-medium
        ]
        
        model_names = ["gpt2", "gpt2-medium"]
        loaded_models = loader.batch_load_models(model_names)
        
        assert len(loaded_models) == 2
        assert "gpt2" in loaded_models
        assert "gpt2-medium" in loaded_models
        
        # Verify load_model was called for each model
        assert mock_load_model.call_count == 2
    
    @patch('attendome.dataset.data_loader.ModelLoader.load_model')
    def test_batch_load_models_with_failure(self, mock_load_model, loader, capsys):
        """Test batch loading with some failures."""
        # Mock one success, one failure
        mock_load_model.side_effect = [
            (Mock(), Mock()),  # gpt2 success
            Exception("Model not found"),  # gpt2-medium failure
        ]
        
        model_names = ["gpt2", "gpt2-medium"] 
        loaded_models = loader.batch_load_models(model_names)
        
        # Only successful model should be in results
        assert len(loaded_models) == 1
        assert "gpt2" in loaded_models
        assert "gpt2-medium" not in loaded_models
        
        # Check error was printed
        captured = capsys.readouterr()
        assert "Failed to load gpt2-medium" in captured.out
    
    @patch('attendome.dataset.data_loader.ModelLoader.load_model')
    def test_batch_load_models_with_progress_callback(self, mock_load_model, loader):
        """Test batch loading with progress callback."""
        mock_load_model.return_value = (Mock(), Mock())
        
        progress_messages = []
        def progress_callback(message):
            progress_messages.append(message)
        
        model_names = ["gpt2", "gpt2-medium"]
        loader.batch_load_models(model_names, progress_callback=progress_callback)
        
        assert len(progress_messages) == 2
        assert "Loading gpt2 (1/2)" in progress_messages[0]
        assert "Loading gpt2-medium (2/2)" in progress_messages[1]
    
    @patch('torch.cuda.empty_cache')
    @patch('attendome.dataset.data_loader.gc.collect')
    def test_clear_cache_all(self, mock_gc_collect, mock_cuda_empty_cache, loader):
        """Test clearing all cached models."""
        # Add some mock cached models
        loader._loaded_models = {
            "gpt2": (Mock(), Mock()),
            "gpt2-medium": (Mock(), Mock())
        }
        
        with patch('torch.cuda.is_available', return_value=True):
            loader.clear_cache()
        
        assert len(loader._loaded_models) == 0
        mock_gc_collect.assert_called_once()
        mock_cuda_empty_cache.assert_called_once()
    
    def test_clear_cache_specific_model(self, loader):
        """Test clearing specific cached model."""
        # Add some mock cached models
        loader._loaded_models = {
            "gpt2": (Mock(), Mock()),
            "gpt2-medium": (Mock(), Mock())
        }
        
        loader.clear_cache("gpt2")
        
        assert "gpt2" not in loader._loaded_models
        assert "gpt2-medium" in loader._loaded_models
    
    def test_clear_cache_nonexistent_model(self, loader):
        """Test clearing cache for nonexistent model."""
        loader._loaded_models = {"gpt2": (Mock(), Mock())}
        
        # Should not raise error
        loader.clear_cache("nonexistent-model")
        
        assert "gpt2" in loader._loaded_models
    
    @patch('transformers.AutoConfig')
    def test_get_model_info_success(self, mock_auto_config, loader):
        """Test successful model info retrieval."""
        mock_config = Mock()
        mock_config.model_type = "gpt2"
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.hidden_size = 768
        mock_config.vocab_size = 50257
        
        mock_auto_config.from_pretrained.return_value = mock_config
        
        info = loader.get_model_info("gpt2")
        
        assert isinstance(info, ModelInfo)
        assert info.model_name == "gpt2"
        assert info.model_type == "gpt2"
        assert info.num_layers == 12
        assert info.num_heads == 12
        assert info.hidden_size == 768
        assert info.vocab_size == 50257
        mock_auto_config.from_pretrained.assert_called_once_with("gpt2")
    
    @patch('transformers.AutoConfig')
    def test_get_model_info_failure(self, mock_auto_config, loader):
        """Test model info retrieval failure."""
        mock_auto_config.from_pretrained.side_effect = Exception("Model not found")
        
        info = loader.get_model_info("nonexistent-model")
        
        assert isinstance(info, ModelInfo)
        assert info.model_name == "nonexistent-model"
        assert info.error is not None
        assert "Model not found" in info.error
    
    @patch('transformers.AutoConfig')
    def test_get_model_info_missing_attributes(self, mock_auto_config, loader):
        """Test model info with missing config attributes."""
        # Create a config object that raises AttributeError for missing attributes
        mock_config = Mock()
        
        def getattr_side_effect(obj, name, default='unknown'):
            if name == 'model_type':
                return 'custom'
            return default
        
        mock_auto_config.from_pretrained.return_value = mock_config
        
        # Patch getattr calls within the function
        with patch('attendome.dataset.data_loader.getattr', side_effect=getattr_side_effect):
            info = loader.get_model_info("custom-model")
        
        assert isinstance(info, ModelInfo)
        assert info.model_name == "custom-model"
        assert info.model_type == "custom"
        assert info.num_layers == "unknown"
        assert info.num_heads == "unknown"
        assert info.hidden_size == "unknown"
        assert info.vocab_size == "unknown"
    
    def test_device_handling_cuda(self):
        """Test device handling for CUDA device."""
        loader = ModelLoader(device="cuda")
        assert loader.device == "cuda"
        
        with patch('attendome.dataset.data_loader.AutoModel') as mock_auto_model:
            with patch('attendome.dataset.data_loader.AutoTokenizer') as mock_auto_tokenizer:
                mock_tokenizer = Mock(pad_token=None, eos_token="</s>")
                mock_model = Mock()
                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                mock_auto_model.from_pretrained.return_value = mock_model
                
                loader.load_model("gpt2")
                
                # For CUDA, device_map should be set to the device
                mock_auto_model.from_pretrained.assert_called_once_with(
                    "gpt2",
                    torch_dtype=loader.torch_dtype,
                    device_map="cuda",
                    trust_remote_code=False
                )
                
                # Model.to should not be called for CUDA (handled by device_map)
                mock_model.to.assert_not_called()