"""Unit tests for symbol_tools module."""

from unittest.mock import patch

import pytest

from kicad_mcp.tools.symbol_tools import (
    find_similar_symbols,
    get_all_symbols_in_library,
    list_available_libraries,
    parse_symbol_identifier,
    read_library_file,
)


class TestParseSymbolIdentifier:
    """Tests for parse_symbol_identifier function."""

    def test_valid_identifier(self):
        lib, sym = parse_symbol_identifier("Device:R")
        assert lib == "Device"
        assert sym == "R"

    def test_valid_identifier_with_colon_in_name(self):
        lib, sym = parse_symbol_identifier("Library:Symbol:WithColon")
        assert lib == "Library"
        assert sym == "Symbol:WithColon"

    def test_missing_colon_raises_error(self):
        with pytest.raises(ValueError, match="Invalid symbol identifier"):
            parse_symbol_identifier("InvalidFormat")

    def test_empty_string_raises_error(self):
        with pytest.raises(ValueError, match="Invalid symbol identifier"):
            parse_symbol_identifier("")


class TestGetAllSymbolsInLibrary:
    """Tests for get_all_symbols_in_library function."""

    def test_parse_simple_symbol(self):
        content = """(kicad_symbol_lib
	(symbol "R"
		(property "Description" "Resistor")
		(property "ki_keywords" "resistor R")
	)
)"""
        symbols = get_all_symbols_in_library(content)
        assert len(symbols) == 1
        assert symbols[0]["name"] == "R"
        assert symbols[0]["description"] == "Resistor"
        assert symbols[0]["keywords"] == "resistor R"

    def test_skip_internal_units(self):
        content = """(kicad_symbol_lib
	(symbol "IC1"
		(property "Description" "Main chip")
	)
	(symbol "IC1_0_1"
		(property "Description" "Internal unit")
	)
	(symbol "IC1_1_1"
		(property "Description" "Another internal unit")
	)
)"""
        symbols = get_all_symbols_in_library(content)
        assert len(symbols) == 1
        assert symbols[0]["name"] == "IC1"

    def test_empty_library(self):
        content = "(kicad_symbol_lib)"
        symbols = get_all_symbols_in_library(content)
        assert symbols == []

    def test_missing_description(self):
        content = """(kicad_symbol_lib
	(symbol "NoDesc"
		(property "ki_keywords" "test")
	)
)"""
        symbols = get_all_symbols_in_library(content)
        assert len(symbols) == 1
        assert symbols[0]["description"] == ""
        assert symbols[0]["keywords"] == "test"


class TestListAvailableLibraries:
    """Tests for list_available_libraries function."""

    @patch("kicad_mcp.tools.symbol_tools.get_symbol_library_path")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_returns_sorted_libraries(self, mock_listdir, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = True
        mock_listdir.return_value = ["Zebra.kicad_sym", "Apple.kicad_sym", "Mango.kicad_sym"]

        libs = list_available_libraries()
        assert libs == ["Apple", "Mango", "Zebra"]

    @patch("kicad_mcp.tools.symbol_tools.get_symbol_library_path")
    @patch("os.path.exists")
    def test_returns_empty_if_path_not_exists(self, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = False

        libs = list_available_libraries()
        assert libs == []

    @patch("kicad_mcp.tools.symbol_tools.get_symbol_library_path")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_filters_non_kicad_files(self, mock_listdir, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = True
        mock_listdir.return_value = ["Valid.kicad_sym", "readme.txt", "other.lib"]

        libs = list_available_libraries()
        assert libs == ["Valid"]


class TestFindSimilarSymbols:
    """Tests for find_similar_symbols function."""

    def test_exact_substring_match(self):
        symbols = ["ATmega328", "ATmega168", "PIC16F84", "STM32F4"]
        similar = find_similar_symbols("mega", symbols)
        assert "ATmega328" in similar
        assert "ATmega168" in similar

    def test_limit_results(self):
        symbols = [f"Symbol{i}" for i in range(20)]
        similar = find_similar_symbols("Symbol", symbols, limit=5)
        assert len(similar) == 5

    def test_fuzzy_match(self):
        symbols = ["Resistor", "Capacitor", "Inductor"]
        similar = find_similar_symbols("Resist", symbols)
        assert "Resistor" in similar


class TestReadLibraryFile:
    """Tests for read_library_file function."""

    @patch("kicad_mcp.tools.symbol_tools.get_symbol_library_path")
    @patch("os.path.exists")
    def test_returns_none_if_not_found(self, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = False

        result = read_library_file("NonExistent")
        assert result is None
