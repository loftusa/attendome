# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring universal embedding spaces for attention heads across different transformer models. The core question: Can we create embeddings where attention heads (like induction heads) in one model correspond to similar heads in different models?

## Development Commands

- **Run Python code**: Always use `uv run` prefix for Python execution
- **Install dependencies**: `uv add <package>` to add new dependencies to pyproject.toml
- **Build induction dataset**: `uv run experiments/build_induction_dataset.py [options]`
- **up-to-date documentation**: always update the README.MD and CLAUDE.MD files to reflect the current state of the project

## Key Dataset Generation Commands

```bash
# Generate induction heads dataset with default models
uv run experiments/build_induction_dataset.py

# Generate with custom models and parameters
uv run experiments/build_induction_dataset.py \
  --models gpt2 gpt2-medium EleutherAI/gpt-neo-125M \
  --num-samples 2000 \
  --print-reports \
  --save-individual
```

## Project Architecture

The project now has a structured Python package for attention head analysis:
- `src/attendome/` - Main package directory
  - `dataset/` - Dataset generation and analysis modules
    - `attention_head_classifier.py` - InductionHeadClassifier for computing scores
    - `data_loader.py` - ModelLoader for handling multiple transformer models  
    - `utils.py` - Analysis utilities and data management functions
- `experiments/` - Research experiment scripts
  - `build_induction_dataset.py` - Pipeline for generating induction head datasets
- `results/induction_heads/` - Output directory for generated datasets

## Research Context

The project has two planned implementation phases:
1. **Phase 1**: Use existing sentence embedding transformer (e.g., Qwen3-Embedding-8B) to clone attention patterns from target transformers without learning new embeddings
2. **Phase 2**: Learn compact embeddings to predict attention head behavior across ~10 different language models

Key research goals:
- Build classification datasets for known attention head types (induction heads, copying heads)
- Train models to predict attention patterns from sentence embeddings
- Measure embedding quality through attention head classification tasks


