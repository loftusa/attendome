# Test Suite for Attendome

This directory contains comprehensive pytest tests for the attendome package.

## Test Structure

- `tests/dataset/` - Tests for the core dataset modules
  - `test_attention_head_classifier.py` - Tests for InductionHeadClassifier
  - `test_data_loader.py` - Tests for ModelLoader
  - `test_utils.py` - Tests for utility functions

## What's Tested

### InductionHeadClassifier (`test_attention_head_classifier.py`)
- Initialization with different devices
- Computation of induction scores with various parameters
- Head classification with different thresholds
- Complete model analysis workflow
- Attention implementation restoration
- Random sequence saving functionality

### ModelLoader (`test_data_loader.py`)
- Model and tokenizer loading with various configurations
- Model caching functionality
- Batch loading with error handling
- Memory management and cache clearing
- Model configuration retrieval
- Device handling for CPU and CUDA

### Utility Functions (`test_utils.py`)
- Saving and loading results in JSON and pickle formats
- Dataset metadata creation
- Score distribution analysis
- Top induction heads extraction
- Summary report generation
- Output filename generation

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/dataset/test_attention_head_classifier.py

# Run with coverage (if coverage is installed)
uv run pytest tests/ --cov=src/attendome
```

## Test Configuration

Tests are configured via `pytest.ini` in the project root with:
- Test discovery patterns
- Markers for different test types (unit, integration, slow)
- Output formatting options

## Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.slow` - Tests that take longer to run