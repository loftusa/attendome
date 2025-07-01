#!/usr/bin/env python3
"""Build induction heads dataset by analyzing multiple transformer models."""

import argparse
import sys
import os
from pathlib import Path

# Add src to path so we can import attendome
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attendome.dataset import InductionHeadClassifier, ModelLoader, save_results, create_dataset_metadata, create_summary_report, generate_output_filename


def main():
    parser = argparse.ArgumentParser(description="Build induction heads dataset")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-3-12b-it",
        ],
        help="List of model names to analyze",
    )
    parser.add_argument(
        "--output-dir", 
        default="results/induction_heads",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=2048,
        help="Number of samples for induction score computation"
    )
    parser.add_argument(
        "--seq-len", 
        type=int, 
        default=50,
        help="Sequence length for analysis"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device", 
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--high-threshold", 
        type=float, 
        default=0.7,
        help="Threshold for high induction heads"
    )
    parser.add_argument(
        "--medium-threshold", 
        type=float, 
        default=0.5,
        help="Threshold for medium induction heads"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "pickle"], 
        default="json",
        help="Output format for results"
    )
    parser.add_argument(
        "--save-individual", 
        action="store_true",
        help="Save individual model results in addition to combined dataset"
    )
    parser.add_argument(
        "--print-reports", 
        action="store_true",
        help="Print summary reports to console"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print(f"Initializing classifier on device: {args.device or 'auto-detect'}")
    classifier = InductionHeadClassifier(device=args.device)
    loader = ModelLoader(device=classifier.device)
    
    print(f"Will analyze {len(args.models)} models: {', '.join(args.models)}")
    
    # Store all results
    all_results = []
    
    # Analyze each model
    for i, model_name in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Analyzing {model_name}...")
        
        try:
            # Load model
            print(f"  Loading model and tokenizer...")
            model, tokenizer = loader.load_model(model_name)
            
            # Analyze model
            print(f"  Computing induction scores...")
            results = classifier.analyze_model(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                num_of_samples=args.num_samples,
                seq_len=args.seq_len,
                batch_size=args.batch_size
            )
            
            # Add classification thresholds to results
            results["classification_thresholds"] = {
                "high": args.high_threshold,
                "medium": args.medium_threshold
            }
            
            # Re-classify with custom thresholds if different from defaults
            if args.high_threshold != 0.5 or args.medium_threshold != 0.2:
                results["classified_heads"] = classifier.classify_heads(
                    results["induction_scores"],
                    high_threshold=args.high_threshold,
                    medium_threshold=args.medium_threshold
                )
            
            all_results.append(results)
            
            # Print summary if requested
            if args.print_reports:
                print("\n" + create_summary_report(results))
            
            # Save individual results if requested
            if args.save_individual:
                output_path = generate_output_filename(
                    model_name, 
                    args.output_dir, 
                    args.format,
                    include_timestamp=False
                )
                save_results(results, output_path, format=args.format)
                print(f"  Saved individual results to: {output_path}")
            
            # Clear model from memory
            loader.clear_cache(model_name)
            
        except Exception as e:
            print(f"  ERROR analyzing {model_name}: {str(e)}")
            continue
    
    if not all_results:
        print("\nNo models were successfully analyzed!")
        return
    
    # Create combined dataset
    print(f"\nCreating combined dataset from {len(all_results)} models...")
    
    # Generate dataset metadata
    metadata = create_dataset_metadata(
        all_results,
        description=f"Induction heads dataset generated from {len(all_results)} transformer models",
        analysis_params={
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "high_threshold": args.high_threshold,
            "medium_threshold": args.medium_threshold
        }
    )
    
    # Combine results
    combined_dataset = {
        "metadata": metadata,
        "model_results": all_results
    }
    
    # Save combined dataset
    combined_path = os.path.join(args.output_dir, f"induction_dataset.{args.format}")
    save_results(combined_dataset, combined_path, format=args.format)
    
    print(f"\nDataset creation complete!")
    print(f"Combined dataset saved to: {combined_path}")
    print(f"Total models analyzed: {len(all_results)}")
    print(f"Total attention heads: {metadata['total_heads']}")
    
    # Print overall statistics
    total_high = sum(len(r["classified_heads"]["high_induction"]) for r in all_results)
    total_medium = sum(len(r["classified_heads"]["medium_induction"]) for r in all_results)
    total_low = sum(len(r["classified_heads"]["low_induction"]) for r in all_results)
    total = metadata["total_heads"]
    
    print(f"\nOverall Classification Results:")
    print(f"- High induction heads: {total_high} ({total_high/total*100:.1f}%)")
    print(f"- Medium induction heads: {total_medium} ({total_medium/total*100:.1f}%)")
    print(f"- Low induction heads: {total_low} ({total_low/total*100:.1f}%)")


if __name__ == "__main__":
    main()