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
- PyTorch, Transformers, NumPy, Pydantic, tqdm (managed via uv)

### Installation
```bash
uv sync
```

### Building Induction Heads Dataset
Generate classification dataset for induction heads across multiple models:

```bash
# Basic usage with default models (Qwen3-4B, Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-3-12B-IT)
uv run experiments/build_induction_dataset.py

# Advanced usage with custom models and parameters
uv run experiments/build_induction_dataset.py \
  --models gpt2 gpt2-medium EleutherAI/gpt-neo-125M \
  --num-samples 2000 \
  --seq-len 50 \
  --batch-size 16 \
  --high-threshold 0.7 \
  --medium-threshold 0.35 \
  --print-reports \
  --save-individual

# Memory-efficient processing for large models
uv run experiments/build_induction_dataset.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --batch-size 8 \
  --device cuda \
  --format json
```

Results are saved to `results/induction_heads/` in JSON/pickle format with:
- Individual model analyses (if `--save-individual`)
- Combined dataset with comprehensive metadata
- Classification of heads as high/medium/low induction with configurable thresholds
- Statistical summaries and score distributions
- Top-k induction heads ranking by score

## Current Implementation

### Dataset Module (`src/attendome/dataset/`)
- **InductionHeadClassifier**: Computes induction scores and classifies attention heads
- **ModelLoader**: Handles loading multiple transformer models with memory management
- **Utilities**: Analysis, reporting, and data management functions

### Induction Head Detection
The classifier identifies induction heads using repeated token sequences and diagonal attention patterns:

```python
# Basic usage
classifier = InductionHeadClassifier(device="cuda")
results = classifier.analyze_model(model, tokenizer, "gpt2")
classified_heads = results.classified_heads

# Advanced configuration
results = classifier.analyze_model(
    model, tokenizer, "gpt2",
    num_of_samples=2048,    # Number of random sequences
    seq_len=50,             # Length of each sequence
    batch_size=16,          # Processing batch size
)

# Access results
print(f"Model: {results.model_name}")
print(f"Architecture: {results.model_configuration['num_layers']} layers")
print(f"High induction heads: {len(results.classified_heads['high_induction'])}")

# Individual head access
for head in results.classified_heads["high_induction"]:
    print(f"Layer {head['layer']}, Head {head['head']}: {head['score']:.3f}")

# Custom classification thresholds
classified = classifier.classify_heads(
    results.induction_scores,
    high_threshold=0.7,     # High induction heads
    medium_threshold=0.35   # Medium induction heads
)
```

Key features:
- **Attention Pattern Analysis**: Uses diagonal attention on repeated sequences to compute induction scores
- **Validated Results**: Structured output with automatic serialization for analysis results
- **Configurable Thresholds**: Customizable classification boundaries (default: 0.7/0.35)
- **Batch Processing**: Memory-efficient processing with configurable batch sizes
- **Statistical Analysis**: Score distributions, percentiles, and ranking metrics
- **Multi-format Output**: JSON and pickle support for results storage

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

Measure embedding quality through attention head classification and retrieval tasks across models. The evaluation framework includes both supervised classification and unsupervised similarity metrics.

### Classification Metrics
- **Precision/Recall**: For induction head detection within each model using configurable thresholds
- **Cross-model F1**: Binary classification of induction vs non-induction heads using learned embeddings
- **Multi-class Classification**: Extending to copying heads, retrieval heads, and other attention patterns

### Retrieval Metrics  
- **Induction-retrieval@k**: Given an induction head in model A, rank all heads from other models by embedding similarity; measure recall@k for finding corresponding induction heads
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks for true positive retrievals
- **Top-k Accuracy**: Percentage of queries where the correct head type appears in top-k results

### Reconstruction Metrics
- **Attention MSE**: $\text{MSE}(\hat{A}, A)$ between predicted and actual attention patterns on held-out sequences
- **Pattern Correlation**: Pearson correlation between predicted and ground-truth attention weights
- **Outlier Detection**: Mean reconstruction error per head to identify "non-embeddable" attention patterns

### Current Statistical Analysis
The implementation provides comprehensive analysis including:
- Score distribution statistics (mean, std, percentiles, min/max)
- Head classification counts and percentages across models
- Top-k ranking of induction heads by score
- Cross-model comparison and aggregation statistics

## Embedding Strategies for Attention Heads

Below are potential *recipes* for converting an attention head `h` into a fixed-length vector embedding `e_h ∈ ℝᵈ`.  Each recipe specifies:

* **Raw signal** – what data is extracted from the head.
* **Alignment** – how the signal is normalised across different transformer architectures.
* **Embedding** – the procedure that maps the signal to a `d`-dimensional vector.
* **Pros / Cons** – why the method might succeed or fail.

### 1&nbsp;·&nbsp;Frozen-encoder KV-cloning (Phase-1 baseline)
* **Raw signal** Fit head-specific key/value tensors `(Kₕ, Vₕ)` so that, for a *frozen* sentence-encoder `S` with hidden states `z_p`, the reconstructed attention
  $\hat A_h(p)=\operatorname{Softmax}(z_p K_h^\top/\sqrt{d_s}) V_h$ matches the true pattern `A_h(p)` over many prompts `p`.
* **Alignment** All heads are projected into `S`'s token space, eliminating tokenizer mismatch.
* **Embedding** Concatenate and flatten `[Kₕ, Vₕ]`, then compress via PCA/random projection to size `d`.
* **Pros** Directly captures the *causal* mechanism that generates attention.
* **Cons** High-dimensional before compression; requires an optimisation per head.
