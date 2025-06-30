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

## Current Tasks

1. **Identify well-understood attention heads** in open-source models (induction heads, copying heads)
2. **Build classification datasets** for known attention head types:
   - Start with binary classification ("is it an induction head?")
   - Use induction head detection methods to create train/test splits
3. **Train prediction models**: Use sentence embeddings from Qwen3-Embedding-8B to predict attention patterns via shallow MLPs

## Methods

### Induction Head Detection
Current method for computing induction scores using repeated sequences:

```python
def compute_induction_score(
    self,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_of_samples: int = 2000,
    seq_len: int = 50,
    batch_size: int = 16,
) -> torch.Tensor:
    # Creates random repetitive sequences and measures attention
    # to positions that would indicate induction behavior
    # Uses diagonal offset of -seq_len + 1 to check proper positions
```

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