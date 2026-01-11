"""Unit tests for footprint_cache module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from kicad_mcp.utils.footprint_cache import (
    _process_library,
    compute_library_hash,
    get_cache_dir,
    get_cache_path,
    get_lock_path,
    is_cache_valid,
    load_footprint_index,
    save_footprint_index,
    search_index,
)


class TestCachePaths:
    """Tests for cache path functions."""

    def test_get_cache_dir_creates_directory(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": temp_dir}),
        ):
            cache_dir = get_cache_dir()
            assert os.path.exists(cache_dir)
            assert "kicad-mcp" in cache_dir

    def test_get_cache_path_returns_json_path(self):
        path = get_cache_path()
        assert path.endswith("footprint_index.json")

    def test_get_lock_path_returns_lock_path(self):
        path = get_lock_path()
        assert path.endswith("footprint_index.lock")


class TestComputeLibraryHash:
    """Tests for compute_library_hash function."""

    def test_returns_empty_for_nonexistent_path(self):
        result = compute_library_hash("/nonexistent/path")
        assert result == ""

    def test_returns_consistent_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a .pretty directory with a .kicad_mod file
            pretty_dir = os.path.join(temp_dir, "Test.pretty")
            os.makedirs(pretty_dir)
            mod_file = os.path.join(pretty_dir, "Test.kicad_mod")
            with open(mod_file, "w") as f:
                f.write("test content")

            hash1 = compute_library_hash(temp_dir)
            hash2 = compute_library_hash(temp_dir)
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5 hex length

    def test_ignores_non_pretty_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-.pretty directory
            other_dir = os.path.join(temp_dir, "Other")
            os.makedirs(other_dir)

            # Hash should be the MD5 of empty string since no .pretty directories
            result = compute_library_hash(temp_dir)
            # MD5 of empty string
            assert result == "d41d8cd98f00b204e9800998ecf8427e"


class TestIsCacheValid:
    """Tests for is_cache_valid function."""

    def test_invalid_version(self):
        cache_data = {"version": 999, "library_path": "/path", "library_hash": "abc"}
        assert is_cache_valid(cache_data, "/path") is False

    def test_invalid_library_path(self):
        cache_data = {"version": 1, "library_path": "/old/path", "library_hash": "abc"}
        assert is_cache_valid(cache_data, "/new/path") is False

    def test_stale_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_data = {"version": 1, "library_path": temp_dir, "library_hash": "stale_hash"}
            assert is_cache_valid(cache_data, temp_dir) is False

    def test_valid_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a .pretty directory
            pretty_dir = os.path.join(temp_dir, "Test.pretty")
            os.makedirs(pretty_dir)
            mod_file = os.path.join(pretty_dir, "Test.kicad_mod")
            with open(mod_file, "w") as f:
                f.write("test content")

            current_hash = compute_library_hash(temp_dir)
            cache_data = {"version": 1, "library_path": temp_dir, "library_hash": current_hash}
            assert is_cache_valid(cache_data, temp_dir) is True


class TestLoadSaveFootprintIndex:
    """Tests for load and save functions."""

    def test_load_returns_none_if_not_exists(self):
        with patch("kicad_mcp.utils.footprint_cache.get_cache_path") as mock_path:
            mock_path.return_value = "/nonexistent/path/cache.json"
            result = load_footprint_index()
            assert result is None

    def test_save_and_load_roundtrip(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("kicad_mcp.utils.footprint_cache.get_cache_path") as mock_path,
            patch("kicad_mcp.utils.footprint_cache.get_lock_path") as mock_lock,
        ):
            cache_file = os.path.join(temp_dir, "test_cache.json")
            mock_path.return_value = cache_file
            mock_lock.return_value = os.path.join(temp_dir, "test.lock")

            test_data = {"version": 1, "footprints": [], "footprint_count": 0}
            assert save_footprint_index(test_data) is True

            loaded = load_footprint_index()
            assert loaded["version"] == 1
            assert loaded["footprint_count"] == 0


class TestSearchIndex:
    """Tests for search_index function."""

    @pytest.fixture
    def sample_index(self):
        return {
            "footprints": [
                {
                    "id": "Resistor_SMD:R_0805_2012Metric",
                    "name": "R_0805_2012Metric",
                    "library": "Resistor_SMD",
                    "description": "Resistor SMD 0805",
                    "keywords": "resistor 0805 SMD",
                },
                {
                    "id": "Capacitor_SMD:C_0805_2012Metric",
                    "name": "C_0805_2012Metric",
                    "library": "Capacitor_SMD",
                    "description": "Capacitor SMD 0805",
                    "keywords": "capacitor 0805 SMD",
                },
                {
                    "id": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                    "name": "SOIC-8_3.9x4.9mm_P1.27mm",
                    "library": "Package_SO",
                    "description": "SOIC 8 pin package",
                    "keywords": "SOIC SO IC",
                },
            ]
        }

    def test_basic_search(self, sample_index):
        result = search_index(sample_index, "Resistor")
        assert result["total_matches"] == 1
        assert result["results"][0]["footprint"] == "Resistor_SMD:R_0805_2012Metric"

    def test_library_filter(self, sample_index):
        result = search_index(sample_index, "0805", library="Resistor_SMD")
        assert result["total_matches"] == 1
        assert result["results"][0]["footprint"] == "Resistor_SMD:R_0805_2012Metric"

    def test_truncation(self, sample_index):
        result = search_index(sample_index, "0", limit=1)  # matches multiple
        assert result["truncated"] is True
        assert len(result["results"]) == 1

    def test_no_truncation_when_under_limit(self, sample_index):
        result = search_index(sample_index, "SOIC", limit=10)
        assert result["truncated"] is False

    def test_case_insensitive(self, sample_index):
        result = search_index(sample_index, "RESISTOR")
        assert result["total_matches"] == 1

    def test_no_matches(self, sample_index):
        result = search_index(sample_index, "nonexistent")
        assert result["total_matches"] == 0
        assert result["results"] == []

    def test_search_by_keywords(self, sample_index):
        result = search_index(sample_index, "SMD")
        assert result["total_matches"] == 2  # Both resistor and capacitor have SMD


class TestProcessLibrary:
    """Tests for _process_library function."""

    def test_returns_empty_on_exception(self):
        parse_func = MagicMock(side_effect=Exception("Parse error"))

        result = _process_library("TestLib", "/fake/path", parse_func)
        assert result == []

    def test_returns_footprints_on_success(self):
        parse_func = MagicMock(
            return_value=[{"name": "FP1", "description": "Desc1", "keywords": "kw1"}]
        )

        result = _process_library("TestLib", "/fake/path", parse_func)
        assert len(result) == 1
        assert result[0]["id"] == "TestLib:FP1"
        assert result[0]["library"] == "TestLib"

    def test_handles_empty_library(self):
        parse_func = MagicMock(return_value=[])

        result = _process_library("EmptyLib", "/fake/path", parse_func)
        assert result == []
