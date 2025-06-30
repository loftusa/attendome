# Attendome

A research project exploring universal embedding spaces for attention heads across different transformer models.

## Overview

This project investigates whether we can create universal embeddings where attention heads (like induction heads) in one model correspond to similar heads in different models. Key research questions:

- Can an induction head in GPT-XL correspond to an induction head in Qwen 25B?
- Which attention heads correspond across models, and which are unique?
- Can we identify attention heads that are "non-embeddable"?

## Implementation Plan

### Phase 1: Attention Pattern Cloning
Use existing sentence embedding transformers (e.g., Qwen3-Embedding-8B) to train models that duplicate attention patterns from target transformer heads without learning new embeddings. This phase will:
- Test feasibility of cloning attention heads across models
- Address technical challenges like different tokenization schemes
- Establish baseline performance metrics

### Phase 2: Learned Embeddings
Create compact learned embeddings to predict attention head behavior across ~10 different language models. This addresses the scalability issues of Phase 1 (learned KVs are too large) and enables deeper generalization insights.

## Getting Started

### Requirements
- Python 3.12+
- PyTorch, Transformers, NumPy, tqdm (managed via uv)

### Installation
```bash
uv add torch transformers numpy tqdm
```

### Building Induction Heads Dataset
Generate classification dataset for induction heads across multiple models:

```bash
# Basic usage with default models (gpt2, distilgpt2, DialoGPT-small)
uv run experiments/build_induction_dataset.py

# Advanced usage with custom models and parameters
uv run experiments/build_induction_dataset.py \
  --models gpt2 gpt2-medium EleutherAI/gpt-neo-125M \
  --num-samples 2000 \
  --seq-len 50 \
  --batch-size 16 \
  --high-threshold 0.5 \
  --medium-threshold 0.2 \
  --print-reports \
  --save-individual
```

Results are saved to `results/induction_heads/` in JSON format with:
- Individual model analyses (if `--save-individual`)
- Combined dataset with metadata
- Classification of heads as high/medium/low induction

## Current Implementation

### Dataset Module (`src/attendome/dataset/`)
- **InductionHeadClassifier**: Computes induction scores and classifies attention heads
- **ModelLoader**: Handles loading multiple transformer models with memory management
- **Utilities**: Analysis, reporting, and data management functions

### Induction Head Detection
The classifier uses repeated sequences to identify induction heads:

```python
classifier = InductionHeadClassifier()
results = classifier.analyze_model(model, tokenizer, "gpt2")
classified_heads = results["classified_heads"]
```

Key features:
- Configurable classification thresholds
- Batch processing for efficiency
- Memory management for large models
- Comprehensive analysis reports

## Current Tasks

1. ✅ **Build classification datasets** for induction heads
2. **Extend to other head types** (copying heads, retrieval heads)
3. **Train prediction models**: Use sentence embeddings from Qwen3-Embedding-8B to predict attention patterns via shallow MLPs

## Resources

### Key Papers
- [The Wisdom of a Crowd of Brains: A Universal Brain Encoder](https://arxiv.org/abs/2406.12179) - Starting methodology
- [Shared Global and Local Geometry of Language Model Embeddings](https://arxiv.org/abs/2503.21073) - Related geometry work
- [Representational Similarity Analysis](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full) - Neuroscience motivation

### Tools & Models
- **Sentence Embeddings**: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (current SOTA open-source)
- **Benchmarks**: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **Induction Head Code**: [dual-route-induction](https://github.com/sfeucht/dual-route-induction/blob/main/scripts/attention_scores.py)

## Evaluation Strategy

Measure embedding quality through attention head classification tasks - if our embeddings capture meaningful attention head properties, we should be able to classify heads by type (induction, copying, etc.) across different models.

Metrics:

- Induction-retrieval@k: given an induction head in model A, rank heads from all other models by embedding cosine; report recall.
- Binary classification: logistic reg. on embeddings $\rightarrow$ F1 for induction vs non-induction.
- Reconstruction: $\text{MSE}(\hat{A}, A)$ on held-out prompts.
- Outlier score: mean reconstruction error per head; high-error tails flagged “non-embeddable’’.