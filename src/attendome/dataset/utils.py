"""Utility functions for dataset management and analysis."""

import json
import pickle
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np
from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass


def save_results(
    results: Dict[str, Any], 
    filepath: str, 
    format: str = "json",
    create_dirs: bool = True
) -> None:
    """Save analysis results to file.
    
    Args:
        results: Dictionary containing analysis results
        filepath: Path to save the file
        format: Format to save in ("json" or "pickle")
        create_dirs: Whether to create parent directories if they don't exist
    """
    if create_dirs:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == "json":
        # with open(filepath, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, indent=2, ensure_ascii=False)
        
        def convert_to_dict(obj):
            """Convert Pydantic models to dictionaries recursively."""
            if hasattr(obj, 'model_dump'):  # Pydantic model
                return obj.model_dump()
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        serializable_results = convert_to_dict(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
    elif format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.")


def load_results(filepath: str, format: str = "json") -> Dict[str, Any]:
    """Load analysis results from file.
    
    Args:
        filepath: Path to the results file
        format: Format of the file ("json" or "pickle")
        
    Returns:
        Dictionary containing loaded results
    """
    if format == "json":
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.")


class DatasetMetadata(BaseModel):
    """Metadata for a dataset of model analyses."""
    created_at: str = ""
    description: str = ""
    num_models: int = 0
    models: List[str] = []
    total_heads: int = 0
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()
        super().__init__(**data)
    
    model_config = ConfigDict(extra="allow")


def create_dataset_metadata(
    model_results: List[Dict[str, Any]],
    description: str = "",
    **kwargs
) -> DatasetMetadata:
    """Create metadata for a dataset of model analyses.
    
    Args:
        model_results: List of analysis results for different models
        description: Description of the dataset
        **kwargs: Additional metadata fields
        
    Returns:
        DatasetMetadata object containing dataset metadata
    """
    return DatasetMetadata(
        description=description,
        num_models=len(model_results),
        models=[result.model_name for result in model_results],
        total_heads=sum(
            # result.get("model_config", result.get("model_configuration", {})).get("num_layers", 0) * 
            # result.get("model_config", result.get("model_configuration", {})).get("num_heads", 0)
            result.model_configuration.get("num_layers", 0) * 
            result.model_configuration.get("num_heads", 0)
            for result in model_results
        ),
        **kwargs
    )


@dataclass
class ScoreDistribution:
    """Statistical analysis of induction scores."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    total_heads: int


def analyze_score_distribution(
    induction_scores: List[List[float]]
) -> ScoreDistribution:
    """Analyze the distribution of induction scores.
    
    Args:
        induction_scores: Nested list of induction scores
        
    Returns:
        ScoreDistribution object with distribution statistics
    """
    # Flatten scores
    all_scores = [score for layer_scores in induction_scores for score in layer_scores]
    all_scores = np.array(all_scores)
    
    return ScoreDistribution(
        mean=float(np.mean(all_scores)),
        std=float(np.std(all_scores)),
        min=float(np.min(all_scores)),
        max=float(np.max(all_scores)),
        median=float(np.median(all_scores)),
        q25=float(np.percentile(all_scores, 25)),
        q75=float(np.percentile(all_scores, 75)),
        total_heads=len(all_scores)
    )


def get_top_induction_heads(
    classified_heads: Dict[str, List[Dict[str, Any]]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Get the top-k induction heads by score.
    
    Args:
        classified_heads: Dictionary of classified heads from InductionHeadClassifier
        top_k: Number of top heads to return
        
    Returns:
        List of top induction heads sorted by score
    """
    all_heads = []
    for category in ["high_induction", "medium_induction", "low_induction"]:
        all_heads.extend(classified_heads.get(category, []))
    
    # Sort by score and return top k
    sorted_heads = sorted(all_heads, key=lambda x: x["score"], reverse=True)
    return sorted_heads[:top_k]


def create_summary_report(results: Union[Dict[str, Any], "AnalysisResults"]) -> str:
    """Create a human-readable summary report of analysis results.
    
    Args:
        results: Analysis results from InductionHeadClassifier (dict or AnalysisResults)
        
    Returns:
        Formatted summary string
    """
    # Handle both dict and Pydantic model formats
    if hasattr(results, "model_name"):
        # Pydantic model
        model_name = results.model_name
        config = results.model_configuration
        classified = results.classified_heads
        induction_scores = results.induction_scores
    else:
        # Dictionary format
        model_name = results.model_name
        config = results.get("model_config", results.get("model_configuration", {}))
        classified = results.classified_heads
        induction_scores = results.induction_scores
    
    total_heads = config["num_layers"] * config["num_heads"]
    high_count = len(classified["high_induction"])
    medium_count = len(classified["medium_induction"])
    low_count = len(classified["low_induction"])
    
    # Get score statistics
    score_stats = analyze_score_distribution(induction_scores)
    
    # Create report
    report = f"""
Induction Head Analysis Report
=============================

Model: {model_name}
Architecture: {config["num_layers"]} layers, {config["num_heads"]} heads per layer
Total attention heads: {total_heads}

Classification Results:
- High induction heads: {high_count} ({high_count/total_heads*100:.1f}%)
- Medium induction heads: {medium_count} ({medium_count/total_heads*100:.1f}%)
- Low induction heads: {low_count} ({low_count/total_heads*100:.1f}%)

Score Statistics:
- Mean: {score_stats.mean:.4f}
- Median: {score_stats.median:.4f}
- Standard deviation: {score_stats.std:.4f}
- Range: [{score_stats.min:.4f}, {score_stats.max:.4f}]

Top 5 Induction Heads:
"""
    
    top_heads = get_top_induction_heads(classified, top_k=5)
    for i, head in enumerate(top_heads, 1):
        report += f"{i}. Layer {head['layer']}, Head {head['head']}: {head['score']:.4f}\n"
    
    return report.strip()


def generate_output_filename(
    model_name: str, 
    base_dir: str = "results/induction_heads",
    extension: str = "json",
    include_timestamp: bool = True
) -> str:
    """Generate a standardized output filename for results.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory for results
        extension: File extension
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Full path for the output file
    """
    # Clean model name for filename
    clean_name = model_name.replace("/", "_").replace("\\", "_")
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{clean_name}_{timestamp}.{extension}"
    else:
        filename = f"{clean_name}.{extension}"
    
    return os.path.join(base_dir, filename)