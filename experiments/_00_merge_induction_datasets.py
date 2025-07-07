#!/usr/bin/env python3
"""
Script to merge two induction dataset JSON files into one.
Combines metadata and model results while handling duplicates appropriately.
"""

import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_dataset(filepath: Path) -> Dict[str, Any]:
    """Load and validate an induction dataset JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Basic validation
    assert 'metadata' in data, f"Missing metadata in {filepath}"
    assert 'model_results' in data, f"Missing model_results in {filepath}"
    
    return data


def merge_metadata(meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata from two datasets."""
    # Combine model lists, removing duplicates while preserving order
    models = list(meta1.get('models', []))
    for model in meta2.get('models', []):
        if model not in models:
            models.append(model)
    
    # Sum total heads and model counts
    total_heads = meta1.get('total_heads', 0) + meta2.get('total_heads', 0)
    num_models = len(models)
    
    # Merge analysis params (use first dataset's params, note differences)
    analysis_params = meta1.get('analysis_params', {}).copy()
    params2 = meta2.get('analysis_params', {})
    
    # Check for parameter differences
    param_diffs = []
    for key, value in params2.items():
        if key in analysis_params and analysis_params[key] != value:
            param_diffs.append(f"{key}: {analysis_params[key]} vs {value}")
    
    merged_meta = {
        "created_at": datetime.now().isoformat(),
        "description": f"Merged induction heads dataset from {num_models} transformer models",
        "num_models": num_models,
        "models": models,
        "total_heads": total_heads,
        "analysis_params": analysis_params,
        "source_files": [str(meta1.get('source_file', 'dataset1')), 
                        str(meta2.get('source_file', 'dataset2'))]
    }
    
    if param_diffs:
        merged_meta["parameter_differences"] = param_diffs
    
    return merged_meta


def merge_model_results(results1: List[Dict[str, Any]], 
                       results2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge model results, handling duplicates by model name."""
    # Create lookup by model name
    results_by_model = {}
    
    # Add results from first dataset
    for result in results1:
        model_name = result.get('model_name')
        if model_name:
            results_by_model[model_name] = result
    
    # Add results from second dataset, handling duplicates
    for result in results2:
        model_name = result.get('model_name')
        if model_name:
            if model_name in results_by_model:
                # Model exists in both datasets - combine head data
                existing = results_by_model[model_name]
                existing_heads = existing.get('heads', [])
                new_heads = result.get('heads', [])
                
                # Merge heads, avoiding duplicates by (layer, head) pairs
                head_lookup = {(h.get('layer'), h.get('head')): h for h in existing_heads}
                
                for head in new_heads:
                    key = (head.get('layer'), head.get('head'))
                    if key not in head_lookup:
                        existing_heads.append(head)
                
                # Update counts
                existing['num_heads'] = len(existing_heads)
                existing['heads'] = existing_heads
            else:
                results_by_model[model_name] = result
    
    return list(results_by_model.values())


@click.command()
@click.argument('dataset1_path', type=click.Path(exists=True, path_type=Path))
@click.argument('dataset2_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output file path (default: merged_induction_dataset.json)')
@click.option('--force', '-f', is_flag=True, 
              help='Overwrite output file if it exists')
def main(dataset1_path: Path, dataset2_path: Path, output: Path, force: bool):
    """
    Merge two induction dataset JSON files into one.
    
    DATASET1_PATH: Path to first induction dataset JSON file
    DATASET2_PATH: Path to second induction dataset JSON file
    """
    # Set default output path
    if output is None:
        output = Path('merged_induction_dataset.json')
    
    # Check if output exists
    if output.exists() and not force:
        click.echo(f"Output file {output} already exists. Use --force to overwrite.")
        return
    
    click.echo(f"Loading dataset 1: {dataset1_path}")
    data1 = load_dataset(dataset1_path)
    
    click.echo(f"Loading dataset 2: {dataset2_path}")
    data2 = load_dataset(dataset2_path)
    
    # Add source file info to metadata
    data1['metadata']['source_file'] = str(dataset1_path)
    data2['metadata']['source_file'] = str(dataset2_path)
    
    click.echo("Merging metadata...")
    merged_metadata = merge_metadata(data1['metadata'], data2['metadata'])
    
    click.echo("Merging model results...")
    merged_results = merge_model_results(data1['model_results'], data2['model_results'])
    
    # Create merged dataset
    merged_dataset = {
        'metadata': merged_metadata,
        'model_results': merged_results
    }
    
    click.echo(f"Writing merged dataset to: {output}")
    with open(output, 'w') as f:
        json.dump(merged_dataset, f, indent=2)
    
    # Print summary
    click.echo("\nMerge Summary:")
    click.echo(f"  Models: {merged_metadata['num_models']}")
    click.echo(f"  Total heads: {merged_metadata['total_heads']}")
    click.echo(f"  Model list: {', '.join(merged_metadata['models'])}")
    
    if 'parameter_differences' in merged_metadata:
        click.echo(f"  Parameter differences detected: {merged_metadata['parameter_differences']}")


if __name__ == "__main__":
    main()