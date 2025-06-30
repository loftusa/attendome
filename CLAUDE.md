# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring universal embedding spaces for attention heads across different transformer models. The core question: Can we create embeddings where attention heads (like induction heads) in one model correspond to similar heads in different models?

## Development Commands

- **Run Python code**: Always use `uv run` prefix for Python execution
- **Install dependencies**: `uv add <package>` to add new dependencies to pyproject.toml

## Project Architecture

The project is in early development with a minimal Python package structure:
- `src/attendome/` - Main package directory with basic module structure
- `experiments/` - Directory for research experiments

## Research Context

The project has two planned implementation phases:
1. **Phase 1**: Use existing sentence embedding transformer (e.g., Qwen3-Embedding-8B) to clone attention patterns from target transformers without learning new embeddings
2. **Phase 2**: Learn compact embeddings to predict attention head behavior across ~10 different language models

Key research goals:
- Build classification datasets for known attention head types (induction heads, copying heads)
- Train models to predict attention patterns from sentence embeddings
- Measure embedding quality through attention head classification tasks