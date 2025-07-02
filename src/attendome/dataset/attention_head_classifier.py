"""Attention head classification for identifying induction heads and other patterns."""

from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration information for a transformer model."""
    num_layers: int
    num_heads: int
    hidden_size: int


class AnalysisResults(BaseModel):
    """Complete analysis results for a transformer model."""
    model_name: str
    model_configuration: Dict[str, int]  # Simple dict instead of ModelConfig
    induction_scores: List[List[float]]
    classified_heads: Dict[str, List[Dict[str, Any]]]
    analysis_params: Dict[str, Any]


class InductionHeadClassifier:
    """Classifier for identifying induction heads in transformer models."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the classifier.
        
        Args:
            device: Device to run computations on. If None, uses CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_induction_score(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_of_samples: int = 2048,
        seq_len: int = 50,
        batch_size: int = 16,
        save_random_repetitive_sequence: bool = False,
    ) -> List[List[float]]:
        """Compute induction scores for all attention heads in a model.
        
        Args:
            model: The transformer model to analyze
            tokenizer: Tokenizer for the model
            num_of_samples: Number of random sequences to generate
            seq_len: Length of each sequence
            batch_size: Batch size for processing
            
        Returns:
            List of lists containing induction scores for each layer and head
        """
        # Store original attention implementation
        original_attn_impl = getattr(model.config, '_attn_implementation', None)
        model.config._attn_implementation = "eager"
        
        # Initialize score tensor
        induction_scores = torch.zeros(
            model.config.num_hidden_layers, 
            model.config.num_attention_heads
        ).to(self.device)
        
        # Generate random repetitive sequences
        vocab_size = tokenizer.vocab_size
        random_sequence = torch.randint(1, vocab_size, (num_of_samples, seq_len))
        random_repetitive_sequence = torch.cat([random_sequence, random_sequence], dim=1)
        if save_random_repetitive_sequence:
            self.random_repetitive_sequence_ = random_repetitive_sequence
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, num_of_samples, batch_size), desc="Computing induction scores"):
                begin_index = i
                end_index = min(i + batch_size, num_of_samples)
                batch = random_repetitive_sequence[begin_index:end_index, :]
                input_data = {"input_ids": batch.to(self.device)}
                
                result = model(**input_data, output_attentions=True)
                
                for layer in range(model.config.num_hidden_layers):
                    layer_values = result.attentions[layer]
                    curr_ind_scores = (
                        layer_values.diagonal(offset=-seq_len + 1, dim1=-2, dim2=-1)[..., 1:]
                        .mean(dim=-1)
                        .sum(dim=0)
                    )
                    induction_scores[layer] += curr_ind_scores
        
        # Normalize scores
        induction_scores /= num_of_samples
        
        # Convert to list structure
        induction_score_list = []
        for layer_idx in range(model.config.num_hidden_layers):
            layer_scores = induction_scores[layer_idx].cpu().tolist()
            induction_score_list.append(layer_scores)
        
        # Restore original attention implementation
        if original_attn_impl is not None:
            model.config._attn_implementation = original_attn_impl
        
        return induction_score_list
    
    def classify_heads(
        self, 
        induction_scores: List[List[float]], 
        high_threshold: float = 0.7,
        medium_threshold: float = .35
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify attention heads based on induction scores.
        
        Args:
            induction_scores: Nested list of scores from compute_induction_score
            high_threshold: Threshold for high induction heads
            medium_threshold: Threshold for medium induction heads
            
        Returns:
            Dictionary with classified heads by category
        """
        classified_heads = {
            "high_induction": [],
            "medium_induction": [], 
            "low_induction": []
        }
        
        for layer_idx, layer_scores in enumerate(induction_scores):
            for head_idx, score in enumerate(layer_scores):
                head_info = {
                    "layer": layer_idx,
                    "head": head_idx, 
                    "score": score
                }
                
                if score >= high_threshold:
                    classified_heads["high_induction"].append(head_info)
                elif score >= medium_threshold:
                    classified_heads["medium_induction"].append(head_info)
                else:
                    classified_heads["low_induction"].append(head_info)
        
        return classified_heads
    
    def analyze_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        **kwargs
    ) -> AnalysisResults:
        """Complete analysis of a model's attention heads.
        
        Args:
            model: The transformer model to analyze
            tokenizer: Tokenizer for the model
            model_name: Name identifier for the model
            **kwargs: Additional arguments for compute_induction_score
            
        Returns:
            Complete analysis results including scores and classifications
        """
        # Move model to device
        model = model.to(self.device)
        
        # Compute induction scores
        induction_scores = self.compute_induction_score(model, tokenizer, **kwargs)
        
        # Extract thresholds from kwargs if provided
        high_threshold = kwargs.get('high_threshold', 0.7)
        medium_threshold = kwargs.get('medium_threshold', 0.35)
        
        # Classify heads
        classified_heads = self.classify_heads(induction_scores, high_threshold, medium_threshold)
        
        # Create model config dict
        model_configuration = {
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "hidden_size": model.config.hidden_size
        }
        
        # Compile results
        results = AnalysisResults(
            model_name=model_name,
            model_configuration=model_configuration,
            induction_scores=induction_scores,
            classified_heads=classified_heads,
            analysis_params=kwargs
        )
        
        return results