"""Tests for utility functions."""

import pytest
import json
import pickle
import os
import tempfile
import numpy as np
from datetime import datetime
from unittest.mock import patch, mock_open
from typing import Dict, Any, List

from attendome.dataset.utils import (
    save_results,
    load_results,
    create_dataset_metadata,
    analyze_score_distribution,
    get_top_induction_heads,
    create_summary_report,
    generate_output_filename,
    DatasetMetadata,
    ScoreDistribution
)


class TestSaveLoadResults:
    """Test saving and loading result functions."""
    
    def test_save_results_json(self):
        """Test saving results in JSON format."""
        results = {"test": "data", "numbers": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            save_results(results, filepath, format="json")
            
            # Verify file was created and contains correct data
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == results
        finally:
            os.unlink(filepath)
    
    def test_save_results_pickle(self):
        """Test saving results in pickle format."""
        results = {"test": "data", "array": np.array([1, 2, 3])}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            save_results(results, filepath, format="pickle")
            
            # Verify file was created and contains correct data
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            assert loaded_data["test"] == results["test"]
            np.testing.assert_array_equal(loaded_data["array"], results["array"])
        finally:
            os.unlink(filepath)
    
    def test_save_results_create_dirs(self):
        """Test saving results with directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "subdir", "test.json")
            results = {"test": "data"}
            
            save_results(results, filepath, create_dirs=True)
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == results
    
    def test_save_results_invalid_format(self):
        """Test saving with invalid format raises error."""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Unsupported format"):
                save_results({"test": "data"}, f.name, format="invalid")
    
    def test_load_results_json(self):
        """Test loading JSON results."""
        results = {"test": "data", "numbers": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(results, f)
            filepath = f.name
        
        try:
            loaded_data = load_results(filepath, format="json")
            assert loaded_data == results
        finally:
            os.unlink(filepath)
    
    def test_load_results_pickle(self):
        """Test loading pickle results."""
        results = {"test": "data", "array": np.array([1, 2, 3])}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(results, f)
            filepath = f.name
        
        try:
            loaded_data = load_results(filepath, format="pickle")
            assert loaded_data["test"] == results["test"]
            np.testing.assert_array_equal(loaded_data["array"], results["array"])
        finally:
            os.unlink(filepath)
    
    def test_load_results_invalid_format(self):
        """Test loading with invalid format raises error."""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Unsupported format"):
                load_results(f.name, format="invalid")


class TestCreateDatasetMetadata:
    """Test dataset metadata creation."""
    
    def test_create_dataset_metadata_basic(self):
        """Test basic metadata creation."""
        model_results = [
            {
                "model_name": "gpt2",
                "model_configuration": {"num_layers": 12, "num_heads": 12}
            },
            {
                "model_name": "gpt2-medium", 
                "model_configuration": {"num_layers": 24, "num_heads": 16}
            }
        ]
        
        metadata = create_dataset_metadata(
            model_results,
            description="Test dataset"
        )
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.description == "Test dataset"
        assert metadata.num_models == 2
        assert metadata.models == ["gpt2", "gpt2-medium"]
        assert metadata.total_heads == (12 * 12) + (24 * 16)
        assert metadata.created_at is not None
        
        # Verify timestamp format
        datetime.fromisoformat(metadata.created_at)
    
    def test_create_dataset_metadata_with_kwargs(self):
        """Test metadata creation with additional fields."""
        model_results = [
            {
                "model_name": "gpt2",
                "model_configuration": {"num_layers": 12, "num_heads": 12}
            }
        ]
        
        metadata = create_dataset_metadata(
            model_results,
            description="Test dataset",
            version="1.0",
            author="test"
        )
        
        # Test extra fields work through Pydantic's Config.extra = "allow"
        assert hasattr(metadata, "version") or "version" in metadata.__pydantic_extra__
        assert hasattr(metadata, "author") or "author" in metadata.__pydantic_extra__
    
    def test_create_dataset_metadata_empty(self):
        """Test metadata creation with empty model list."""
        metadata = create_dataset_metadata([])
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.num_models == 0
        assert metadata.models == []
        assert metadata.total_heads == 0


class TestAnalyzeScoreDistribution:
    """Test score distribution analysis."""
    
    def test_analyze_score_distribution_basic(self):
        """Test basic score distribution analysis."""
        scores = [
            [0.1, 0.2, 0.8],  # Layer 0
            [0.3, 0.9, 0.4]   # Layer 1
        ]
        
        stats = analyze_score_distribution(scores)
        
        all_scores = [0.1, 0.2, 0.8, 0.3, 0.9, 0.4]
        expected_mean = np.mean(all_scores)
        expected_std = np.std(all_scores)
        
        assert isinstance(stats, ScoreDistribution)
        assert abs(stats.mean - expected_mean) < 1e-6
        assert abs(stats.std - expected_std) < 1e-6
        assert stats.min == 0.1
        assert stats.max == 0.9
        assert stats.total_heads == 6
        
        # Check that all values are floats (not numpy types)
        for attr in ["mean", "std", "min", "max", "median", "q25", "q75"]:
            assert isinstance(getattr(stats, attr), float)
    
    def test_analyze_score_distribution_single_layer(self):
        """Test distribution analysis with single layer."""
        scores = [[0.5, 0.6, 0.7]]
        
        stats = analyze_score_distribution(scores)
        
        assert isinstance(stats, ScoreDistribution)
        assert stats.mean == 0.6
        assert stats.min == 0.5
        assert stats.max == 0.7
        assert stats.total_heads == 3
    
    def test_analyze_score_distribution_empty(self):
        """Test distribution analysis with empty scores."""
        scores = []
        
        # This should raise an error for empty arrays
        with pytest.raises(ValueError):
            analyze_score_distribution(scores)


class TestGetTopInductionHeads:
    """Test getting top induction heads."""
    
    def test_get_top_induction_heads_basic(self):
        """Test getting top induction heads."""
        # For backward compatibility, get_top_induction_heads still accepts dict format
        classified_heads = {
            "high_induction": [
                {"layer": 0, "head": 0, "score": 0.9},
                {"layer": 1, "head": 1, "score": 0.8}
            ],
            "medium_induction": [
                {"layer": 0, "head": 1, "score": 0.6}
            ],
            "low_induction": [
                {"layer": 1, "head": 0, "score": 0.1}
            ]
        }
        
        top_heads = get_top_induction_heads(classified_heads, top_k=3)
        
        assert len(top_heads) == 3
        assert top_heads[0]["score"] == 0.9  # Highest score first
        assert top_heads[1]["score"] == 0.8
        assert top_heads[2]["score"] == 0.6
        
        # Verify descending order
        for i in range(len(top_heads) - 1):
            assert top_heads[i]["score"] >= top_heads[i + 1]["score"]
    
    def test_get_top_induction_heads_fewer_than_k(self):
        """Test getting top heads when fewer heads than k exist."""
        # For backward compatibility, get_top_induction_heads still accepts dict format
        classified_heads = {
            "high_induction": [{"layer": 0, "head": 0, "score": 0.9}],
            "medium_induction": [],
            "low_induction": []
        }
        
        top_heads = get_top_induction_heads(classified_heads, top_k=5)
        
        assert len(top_heads) == 1
        assert top_heads[0]["score"] == 0.9
    
    def test_get_top_induction_heads_empty(self):
        """Test getting top heads with empty classification."""
        # For backward compatibility, get_top_induction_heads still accepts dict format
        classified_heads = {
            "high_induction": [],
            "medium_induction": [],
            "low_induction": []
        }
        
        top_heads = get_top_induction_heads(classified_heads, top_k=5)
        
        assert len(top_heads) == 0


class TestCreateSummaryReport:
    """Test summary report creation."""
    
    def test_create_summary_report_basic(self):
        """Test basic summary report creation."""
        results = {
            "model_name": "gpt2",
            "model_configuration": {
                "num_layers": 12,
                "num_heads": 12,
                "hidden_size": 768
            },
            "induction_scores": [
                [0.9, 0.1, 0.2, 0.05] * 3,  # 12 heads per layer
            ] * 12,  # 12 layers
            "classified_heads": {
                "high_induction": [
                    {"layer": 0, "head": 0, "score": 0.9}
                ],
                "medium_induction": [
                    {"layer": 0, "head": 2, "score": 0.2}
                ],
                "low_induction": [
                    {"layer": 0, "head": 1, "score": 0.1},
                    {"layer": 0, "head": 3, "score": 0.05}
                ] * 36  # Fill out the remaining heads
            }
        }
        
        report = create_summary_report(results)
        
        assert "gpt2" in report
        assert "12 layers, 12 heads per layer" in report
        assert "Total attention heads: 144" in report
        assert "High induction heads:" in report
        assert "Score Statistics:" in report
        assert "Top 5 Induction Heads:" in report
        
        # Check that percentages are calculated
        assert "%" in report
    
    def test_create_summary_report_structure(self):
        """Test that summary report has expected structure."""
        results = {
            "model_name": "test-model",
            "model_configuration": {"num_layers": 2, "num_heads": 4, "hidden_size": 64},
            "induction_scores": [[0.8, 0.3, 0.1, 0.05], [0.6, 0.4, 0.15, 0.02]],
            "classified_heads": {
                "high_induction": [{"layer": 0, "head": 0, "score": 0.8}],
                "medium_induction": [{"layer": 0, "head": 1, "score": 0.3}],
                "low_induction": [{"layer": 0, "head": 2, "score": 0.1}]
            }
        }
        
        report = create_summary_report(results)
        
        # Check for expected sections
        sections = [
            "Induction Head Analysis Report",
            "Model:",
            "Architecture:",
            "Classification Results:",
            "Score Statistics:",
            "Top 5 Induction Heads:"
        ]
        
        for section in sections:
            assert section in report


class TestGenerateOutputFilename:
    """Test output filename generation."""
    
    def test_generate_output_filename_basic(self):
        """Test basic filename generation."""
        filename = generate_output_filename("gpt2")
        
        assert filename.startswith("results/induction_heads/gpt2_")
        assert filename.endswith(".json")
        assert len(filename.split("_")) >= 2  # Should have timestamp
    
    def test_generate_output_filename_no_timestamp(self):
        """Test filename generation without timestamp."""
        filename = generate_output_filename("gpt2", include_timestamp=False)
        
        assert filename == "results/induction_heads/gpt2.json"
    
    def test_generate_output_filename_custom_params(self):
        """Test filename generation with custom parameters."""
        filename = generate_output_filename(
            "gpt2-medium",
            base_dir="custom/dir",
            extension="pkl",
            include_timestamp=False
        )
        
        assert filename == "custom/dir/gpt2-medium.pkl"
    
    def test_generate_output_filename_clean_model_name(self):
        """Test that model name is cleaned for filesystem."""
        filename = generate_output_filename(
            "EleutherAI/gpt-neo-125M",
            include_timestamp=False
        )
        
        assert filename == "results/induction_heads/EleutherAI_gpt-neo-125M.json"
        assert "/" not in os.path.basename(filename)
        assert "\\" not in os.path.basename(filename)
    
    @patch('attendome.dataset.utils.datetime')
    def test_generate_output_filename_timestamp_format(self, mock_datetime):
        """Test timestamp format in filename."""
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        
        filename = generate_output_filename("gpt2", include_timestamp=True)
        
        assert "20240101_120000" in filename
        mock_datetime.now.return_value.strftime.assert_called_with("%Y%m%d_%H%M%S")