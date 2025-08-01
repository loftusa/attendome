"""Model loading utilities for attention head analysis."""

from typing import List, Tuple, Optional, Dict, Any
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import gc
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Basic information about a transformer model."""
    model_name: str
    model_type: str = "unknown"
    num_layers: Any = "unknown"
    num_heads: Any = "unknown"
    hidden_size: Any = "unknown"
    vocab_size: Any = "unknown"
    error: Optional[str] = None


class ModelLoader:
    """Utility class for loading and managing transformer models."""
    
    def __init__(self, device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None):
        """Initialize the model loader.
        
        Args:
            device: Device to load models on. If None, uses CUDA if available.
            torch_dtype: Data type for model weights. If None, uses float16 on CUDA, float32 on CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self._loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
    
    def load_model(
        self, 
        model_name: str,
        cache_model: bool = False,
        trust_remote_code: bool = False,
        **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a transformer model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            cache_model: Whether to cache the loaded model for reuse
            trust_remote_code: Whether to trust remote code execution
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if cache_model and model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        # Set up tokenizer padding
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype = torch.float32,
            # torch_dtype=self.torch_dtype,
            # device_map=self.device if self.device != "cpu" else None,
            device_map="cuda",
            trust_remote_code=trust_remote_code,
            **model_kwargs
        )

        # Configure model for attention extraction
        model.eval()
        model.config._attn_implementation = "eager"
        
        if self.device == "cpu":
            model = model.to(self.device)
        
        if cache_model:
            self._loaded_models[model_name] = (model, tokenizer)

        # import torch.nn as nn
        # model = nn.DataParallel(model)
        # model.layers = model.layers[:3]
        
        return model, tokenizer
    
    def get_supported_models(self) -> List[str]:
        """Get a list of commonly used models for induction head analysis.
        
        Returns:
            List of model names suitable for analysis
        """
        return [
            "gpt2",
            "gpt2-medium", 
            "gpt2-large",
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "distilgpt2",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-j-6B",
        ]
    
    def batch_load_models(
        self, 
        model_names: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]]:
        """Load multiple models with memory management.
        
        Args:
            model_names: List of model names to load
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping model names to (model, tokenizer) tuples
        """
        loaded_models = {}
        
        for i, model_name in enumerate(model_names):
            try:
                if progress_callback:
                    progress_callback(f"Loading {model_name} ({i+1}/{len(model_names)})")
                
                model, tokenizer = self.load_model(model_name)
                loaded_models[model_name] = (model, tokenizer)
                
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)}")
                continue
        
        return loaded_models
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cached models to free memory.
        
        Args:
            model_name: Specific model to clear. If None, clears all cached models.
        """
        if model_name:
            if model_name in self._loaded_models:
                del self._loaded_models[model_name]
        else:
            self._loaded_models.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self, model_name: str) -> "ModelInfo":
        """Get basic information about a model without loading it.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            ModelInfo object with model configuration information
        """
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            return ModelInfo(
                model_name=model_name,
                model_type=getattr(config, 'model_type', 'unknown'),
                num_layers=getattr(config, 'num_hidden_layers', 'unknown'),
                num_heads=getattr(config, 'num_attention_heads', 'unknown'),
                hidden_size=getattr(config, 'hidden_size', 'unknown'),
                vocab_size=getattr(config, 'vocab_size', 'unknown')
            )
        except Exception as e:
            return ModelInfo(
                model_name=model_name,
                error=str(e)
            )