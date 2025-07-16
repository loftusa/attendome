"""Attention head classification for identifying induction heads and other patterns."""

from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
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
    # concept_induction_scores: List[List[float]]
    classified_heads: Dict[str, List[Dict[str, Any]]]
    analysis_params: Dict[str, Any]
    # classification_thresholds: Dict[str, float]
    attention_maps: Optional[Dict[str, Any]] = None  # Optional attention maps for RSA
    
    def plot(self, plot_type: str = "overview", **kwargs):
        """Plot analysis results for quick visualization.
        
        Args:
            plot_type: Type of plot ("overview", "scores", "heatmap", "distribution")
            **kwargs: Additional plotting arguments
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
        except ImportError:
            print("Matplotlib and seaborn are required for plotting.")
            print("Install with: pip install matplotlib seaborn")
            return
        
        if plot_type == "overview":
            self._plot_overview(**kwargs)
        elif plot_type == "scores":
            self._plot_scores(**kwargs)
        elif plot_type == "heatmap":
            self._plot_heatmap(**kwargs)
        elif plot_type == "distribution":
            self._plot_distribution(**kwargs)
        else:
            print(f"Unknown plot type: {plot_type}")
            print("Available types: 'overview', 'scores', 'heatmap', 'distribution'")
    
    def _plot_overview(self, figsize=(15, 10), **kwargs):
        """Create an overview plot with multiple subplots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Induction Head Analysis: {self.model_name}', fontsize=16)
        
        # 1. Score heatmap
        scores_array = np.array(self.induction_scores)
        im1 = axes[0, 0].imshow(scores_array, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Induction Scores by Layer and Head')
        axes[0, 0].set_xlabel('Head Index')
        axes[0, 0].set_ylabel('Layer Index')
        plt.colorbar(im1, ax=axes[0, 0], label='Induction Score')
        
        # 2. Classification summary
        categories = ['High', 'Medium', 'Low']
        counts = [
            len(self.classified_heads['high_induction']),
            len(self.classified_heads['medium_induction']),
            len(self.classified_heads['low_induction'])
        ]
        colors = ['red', 'orange', 'blue']
        axes[0, 1].bar(categories, counts, color=colors, alpha=0.7)
        axes[0, 1].set_title('Head Classification Counts')
        axes[0, 1].set_ylabel('Number of Heads')
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[0, 1].text(i, count + 0.1, str(count), ha='center')
        
        # 3. Score distribution
        all_scores = [score for layer_scores in self.induction_scores for score in layer_scores]
        axes[1, 0].hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Distribution of Induction Scores')
        axes[1, 0].set_xlabel('Induction Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_scores):.3f}')
        axes[1, 0].legend()
        
        # 4. Top heads visualization
        all_heads = []
        for category, heads in self.classified_heads.items():
            for head in heads:
                all_heads.append((head['score'], head['layer'], head['head'], category))
        
        # Sort by score and take top 10
        all_heads.sort(reverse=True)
        top_heads = all_heads[:10]
        
        if top_heads:
            scores = [h[0] for h in top_heads]
            labels = [f"L{h[1]}H{h[2]}" for h in top_heads]
            categories_top = [h[3] for h in top_heads]
            
            # Color by category
            color_map = {'high_induction': 'red', 'medium_induction': 'orange', 'low_induction': 'blue'}
            bar_colors = [color_map.get(cat, 'gray') for cat in categories_top]
            
            bars = axes[1, 1].barh(range(len(top_heads)), scores, color=bar_colors, alpha=0.7)
            axes[1, 1].set_yticks(range(len(top_heads)))
            axes[1, 1].set_yticklabels(labels)
            axes[1, 1].set_title('Top 10 Induction Heads')
            axes[1, 1].set_xlabel('Induction Score')
            axes[1, 1].invert_yaxis()  # Highest scores at top
        
        plt.tight_layout()
        plt.show()
    
    def _plot_scores(self, figsize=(12, 8), **kwargs):
        """Plot detailed induction scores heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        scores_array = np.array(self.induction_scores)
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(scores_array, 
                    annot=kwargs.get('annotate', False),
                    fmt='.3f',
                    cmap=kwargs.get('cmap', 'viridis'),
                    cbar_kws={'label': 'Induction Score'})
        
        plt.title(f'Induction Scores Heatmap: {self.model_name}')
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        
        # Add model info
        plt.figtext(0.02, 0.02, 
                   f"Model: {self.model_name} | "
                   f"Layers: {self.model_configuration['num_layers']} | "
                   f"Heads: {self.model_configuration['num_heads']}",
                   fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_heatmap(self, threshold_high=0.7, threshold_medium=0.35, figsize=(12, 8), **kwargs):
        """Plot heatmap with classification boundaries."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        scores_array = np.array(self.induction_scores)
        
        # Create custom colormap with threshold boundaries
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(scores_array, cmap='RdYlBu_r', aspect='auto')
        
        # Add classification overlays
        for layer_idx, layer_scores in enumerate(self.induction_scores):
            for head_idx, score in enumerate(layer_scores):
                if score >= threshold_high:
                    # High induction - red border
                    rect = plt.Rectangle((head_idx-0.4, layer_idx-0.4), 0.8, 0.8, 
                                       fill=False, edgecolor='red', linewidth=3)
                    ax.add_patch(rect)
                elif score >= threshold_medium:
                    # Medium induction - orange border
                    rect = plt.Rectangle((head_idx-0.4, layer_idx-0.4), 0.8, 0.8, 
                                       fill=False, edgecolor='orange', linewidth=2)
                    ax.add_patch(rect)
        
        # Customize plot
        ax.set_title(f'Induction Scores with Classification: {self.model_name}')
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Induction Score')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='red', linewidth=3, label=f'High (≥{threshold_high})'),
            Patch(facecolor='white', edgecolor='orange', linewidth=2, label=f'Medium (≥{threshold_medium})'),
            Patch(facecolor='white', edgecolor='gray', label=f'Low (<{threshold_medium})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_distribution(self, bins=30, figsize=(12, 6), **kwargs):
        """Plot distribution of induction scores with statistics."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        all_scores = [score for layer_scores in self.induction_scores for score in layer_scores]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(all_scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'Score Distribution: {self.model_name}')
        ax1.set_xlabel('Induction Score')
        ax1.set_ylabel('Frequency')
        
        # Add statistics lines
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        std_score = np.std(all_scores)
        
        ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        ax1.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
        ax1.axvline(mean_score + std_score, color='orange', linestyle=':', alpha=0.7, label=f'±1 STD: {std_score:.3f}')
        ax1.axvline(mean_score - std_score, color='orange', linestyle=':', alpha=0.7)
        ax1.legend()
        
        # Box plot by layer
        layer_scores = []
        layer_labels = []
        for i, layer_scores_list in enumerate(self.induction_scores):
            if layer_scores_list:  # Only add non-empty layers
                layer_scores.append(layer_scores_list)
                layer_labels.append(f'L{i}')
        
        if layer_scores:
            bp = ax2.boxplot(layer_scores, labels=layer_labels, patch_artist=True)
            ax2.set_title('Score Distribution by Layer')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Induction Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nSummary Statistics for {self.model_name}:")
        print(f"  Total heads: {len(all_scores)}")
        print(f"  Mean score: {mean_score:.4f}")
        print(f"  Median score: {median_score:.4f}")
        print(f"  Std deviation: {std_score:.4f}")
        print(f"  Min score: {np.min(all_scores):.4f}")
        print(f"  Max score: {np.max(all_scores):.4f}")
        print(f"  High induction heads: {len(self.classified_heads['high_induction'])}")
        print(f"  Medium induction heads: {len(self.classified_heads['medium_induction'])}")
        print(f"  Low induction heads: {len(self.classified_heads['low_induction'])}")


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
    
    def extract_attention_maps(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sequences: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        batch_size: int = 16
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps from specific layers/heads for RSA analysis.
        
        Args:
            model: The transformer model to analyze
            tokenizer: Tokenizer for the model
            sequences: Input sequences [num_sequences, seq_len]
            layer_indices: Specific layers to extract (None = all layers)
            head_indices: Specific heads to extract (None = all heads)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with attention maps keyed by 'layer_X_head_Y'
        """
        # Store original attention implementation
        original_attn_impl = getattr(model.config, '_attn_implementation', None)
        model.config._attn_implementation = "eager"
        
        # Default to all layers/heads if not specified
        if layer_indices is None:
            layer_indices = list(range(model.config.num_hidden_layers))
        if head_indices is None:
            head_indices = list(range(model.config.num_attention_heads))
        
        attention_maps = {}
        num_sequences = sequences.shape[0]
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, num_sequences, batch_size), desc="Extracting attention maps"):
                end_idx = min(i + batch_size, num_sequences)
                batch = sequences[i:end_idx].to(self.device)
                
                result = model(input_ids=batch, output_attentions=True)
                
                # Extract attention maps for specified layers/heads
                for layer_idx in layer_indices:
                    if layer_idx < len(result.attentions):
                        layer_attention = result.attentions[layer_idx]  # [batch, heads, seq, seq]
                        
                        for head_idx in head_indices:
                            if head_idx < layer_attention.shape[1]:
                                key = f"layer_{layer_idx}_head_{head_idx}"
                                head_attention = layer_attention[:, head_idx, :, :]  # [batch, seq, seq]
                                
                                if key not in attention_maps:
                                    attention_maps[key] = []
                                
                                attention_maps[key].append(head_attention.cpu())
        
        # Concatenate batches
        for key in attention_maps:
            attention_maps[key] = torch.cat(attention_maps[key], dim=0)
        
        # Restore original attention implementation
        if original_attn_impl is not None:
            model.config._attn_implementation = original_attn_impl
        
        return attention_maps
    
    def extract_induction_attention_maps(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_sequences: int = 100,
        seq_len: int = 50,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        batch_size: int = 16
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps specifically for induction head analysis.
        
        Args:
            model: The transformer model to analyze
            tokenizer: Tokenizer for the model
            num_sequences: Number of random repetitive sequences to generate
            seq_len: Length of each sequence part (total will be 2*seq_len)
            layer_indices: Specific layers to extract (None = all layers)
            head_indices: Specific heads to extract (None = all heads)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with attention maps keyed by 'layer_X_head_Y'
        """
        # Generate random repetitive sequences
        vocab_size = tokenizer.vocab_size
        random_sequence = torch.randint(1, vocab_size, (num_sequences, seq_len))
        random_repetitive_sequence = torch.cat([random_sequence, random_sequence], dim=1)
        
        return self.extract_attention_maps(
            model=model,
            tokenizer=tokenizer,
            sequences=random_repetitive_sequence,
            layer_indices=layer_indices,
            head_indices=head_indices,
            batch_size=batch_size
        )
    
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
        extract_attention_maps: bool = False,
        attention_map_config: Optional[Dict] = None,
        **kwargs
    ) -> AnalysisResults:
        """Complete analysis of a model's attention heads.
        
        Args:
            model: The transformer model to analyze
            tokenizer: Tokenizer for the model
            model_name: Name identifier for the model
            extract_attention_maps: Whether to extract attention maps for RSA analysis
            attention_map_config: Configuration for attention map extraction
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
        
        # Extract attention maps if requested
        attention_maps = None
        if extract_attention_maps:
            attention_map_config = attention_map_config or {}
            attention_maps = self.extract_induction_attention_maps(
                model=model,
                tokenizer=tokenizer,
                num_sequences=attention_map_config.get('num_sequences', 100),
                seq_len=attention_map_config.get('seq_len', kwargs.get('seq_len', 50)),
                layer_indices=attention_map_config.get('layer_indices'),
                head_indices=attention_map_config.get('head_indices'),
                batch_size=attention_map_config.get('batch_size', 16)
            )
            
            # Convert tensors to lists for serialization
            attention_maps_serializable = {}
            for key, tensor in attention_maps.items():
                attention_maps_serializable[key] = {
                    'data': tensor.numpy().tolist(),
                    'shape': list(tensor.shape)
                }
            attention_maps = attention_maps_serializable
        
        # Compile results
        results = AnalysisResults(
            model_name=model_name,
            model_configuration=model_configuration,
            induction_scores=induction_scores,
            classified_heads=classified_heads,
            analysis_params=kwargs,
            attention_maps=attention_maps
        )
        
        return results