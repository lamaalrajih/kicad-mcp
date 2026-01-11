"""Unit tests for symbol_cache module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from kicad_mcp.utils.symbol_cache import (
    _process_library,
    compute_library_hash,
    get_cache_dir,
    get_cache_path,
    get_lock_path,
    is_cache_valid,
    load_symbol_index,
    save_symbol_index,
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
        assert path.endswith("symbol_index.json")

    def test_get_lock_path_returns_lock_path(self):
        path = get_lock_path()
        assert path.endswith("symbol_index.lock")


class TestComputeLibraryHash:
    """Tests for compute_library_hash function."""

    def test_returns_empty_for_nonexistent_path(self):
        result = compute_library_hash("/nonexistent/path")
        assert result == ""

    def test_returns_consistent_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lib_file = os.path.join(temp_dir, "Test.kicad_sym")
            with open(lib_file, "w") as f:
                f.write("test content")

            hash1 = compute_library_hash(temp_dir)
            hash2 = compute_library_hash(temp_dir)
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5 hex length


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


class TestLoadSaveSymbolIndex:
    """Tests for load and save functions."""

    def test_load_returns_none_if_not_exists(self):
        with patch("kicad_mcp.utils.symbol_cache.get_cache_path") as mock_path:
            mock_path.return_value = "/nonexistent/path/cache.json"
            result = load_symbol_index()
            assert result is None

    def test_save_and_load_roundtrip(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("kicad_mcp.utils.symbol_cache.get_cache_path") as mock_path,
            patch("kicad_mcp.utils.symbol_cache.get_lock_path") as mock_lock,
        ):
            cache_file = os.path.join(temp_dir, "test_cache.json")
            mock_path.return_value = cache_file
            mock_lock.return_value = os.path.join(temp_dir, "test.lock")

            test_data = {"version": 1, "symbols": [], "symbol_count": 0}
            assert save_symbol_index(test_data) is True

            loaded = load_symbol_index()
            assert loaded["version"] == 1
            assert loaded["symbol_count"] == 0


class TestSearchIndex:
    """Tests for search_index function."""

    @pytest.fixture
    def sample_index(self):
        return {
            "symbols": [
                {
                    "id": "Device:R",
                    "name": "R",
                    "library": "Device",
                    "description": "Resistor",
                    "keywords": "passive R",
                },
                {
                    "id": "Device:C",
                    "name": "C",
                    "library": "Device",
                    "description": "Capacitor",
                    "keywords": "passive C",
                },
                {
                    "id": "MCU:ATmega328",
                    "name": "ATmega328",
                    "library": "MCU",
                    "description": "AVR Microcontroller",
                    "keywords": "AVR MCU",
                },
            ]
        }

    def test_basic_search(self, sample_index):
        result = search_index(sample_index, "Resistor")
        assert result["total_matches"] == 1
        assert result["results"][0]["symbol"] == "Device:R"

    def test_library_filter(self, sample_index):
        result = search_index(sample_index, "passive", library="Device")
        assert result["total_matches"] == 2

    def test_truncation(self, sample_index):
        result = search_index(sample_index, "e", limit=1)  # matches all
        assert result["truncated"] is True
        assert len(result["results"]) == 1

    def test_no_truncation_when_under_limit(self, sample_index):
        result = search_index(sample_index, "ATmega", limit=10)
        assert result["truncated"] is False

    def test_case_insensitive(self, sample_index):
        result = search_index(sample_index, "RESISTOR")
        assert result["total_matches"] == 1

    def test_no_matches(self, sample_index):
        result = search_index(sample_index, "nonexistent")
        assert result["total_matches"] == 0
        assert result["results"] == []


class TestProcessLibrary:
    """Tests for _process_library function."""

    def test_returns_empty_if_read_fails(self):
        read_func = MagicMock(return_value=None)
        parse_func = MagicMock()

        result = _process_library("TestLib", read_func, parse_func)
        assert result == []
        parse_func.assert_not_called()

    def test_returns_symbols_on_success(self):
        read_func = MagicMock(return_value="content")
        parse_func = MagicMock(
            return_value=[{"name": "Sym1", "description": "Desc1", "keywords": "kw1"}]
        )

        result = _process_library("TestLib", read_func, parse_func)
        assert len(result) == 1
        assert result[0]["id"] == "TestLib:Sym1"
        assert result[0]["library"] == "TestLib"

    def test_handles_parse_exception(self):
        read_func = MagicMock(return_value="content")
        parse_func = MagicMock(side_effect=Exception("Parse error"))

        result = _process_library("TestLib", read_func, parse_func)
        assert result == []
