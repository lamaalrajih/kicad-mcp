"""Unit tests for footprint_tools module."""

from unittest.mock import patch

import pytest

from kicad_mcp.tools.footprint_tools import (
    find_similar_footprints,
    get_all_footprints_in_library,
    list_available_libraries,
    parse_footprint_identifier,
    parse_pads_from_footprint,
    read_footprint_file,
)


class TestParseFootprintIdentifier:
    """Tests for parse_footprint_identifier function."""

    def test_valid_identifier(self):
        lib, fp = parse_footprint_identifier("Resistor_SMD:R_0805_2012Metric")
        assert lib == "Resistor_SMD"
        assert fp == "R_0805_2012Metric"

    def test_valid_identifier_with_colon_in_name(self):
        lib, fp = parse_footprint_identifier("Library:Footprint:WithColon")
        assert lib == "Library"
        assert fp == "Footprint:WithColon"

    def test_missing_colon_raises_error(self):
        with pytest.raises(ValueError, match="Invalid footprint identifier"):
            parse_footprint_identifier("InvalidFormat")

    def test_empty_string_raises_error(self):
        with pytest.raises(ValueError, match="Invalid footprint identifier"):
            parse_footprint_identifier("")


class TestParsePadsFromFootprint:
    """Tests for parse_pads_from_footprint function."""

    def test_parse_smd_pad(self):
        content = """(footprint "R_0805"
            (pad "1" smd rect (at -1 0) (size 1 1.2))
            (pad "2" smd rect (at 1 0) (size 1 1.2))
        )"""
        pads = parse_pads_from_footprint(content)
        assert len(pads) == 2
        assert pads[0]["number"] == "1"
        assert pads[0]["type"] == "smd"
        assert pads[0]["shape"] == "rect"
        assert pads[0]["position"] == [-1.0, 0.0]

    def test_parse_thru_hole_pad_with_drill(self):
        content = """(footprint "DIP-8"
            (pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8))
        )"""
        pads = parse_pads_from_footprint(content)
        assert len(pads) == 1
        assert pads[0]["type"] == "thru_hole"
        assert pads[0]["drill"] == [0.8, 0.8]

    def test_parse_pad_with_angle(self):
        content = """(footprint "Rotated"
            (pad "1" smd rect (at 1.5 0 45) (size 1 1))
        )"""
        pads = parse_pads_from_footprint(content)
        assert len(pads) == 1
        assert pads[0]["angle"] == 45.0

    def test_empty_footprint(self):
        content = "(footprint \"Empty\")"
        pads = parse_pads_from_footprint(content)
        assert pads == []


class TestGetAllFootprintsInLibrary:
    """Tests for get_all_footprints_in_library function."""

    def test_returns_empty_for_nonexistent_path(self):
        result = get_all_footprints_in_library("/nonexistent/path")
        assert result == []

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("builtins.open")
    def test_parses_footprint_files(self, mock_open, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["R_0805.kicad_mod", "C_0603.kicad_mod"]

        mock_open.return_value.__enter__.return_value.read.return_value = """
        (footprint "R_0805"
            (descr "Resistor SMD 0805")
            (tags "resistor 0805 SMD")
        )
        """

        result = get_all_footprints_in_library("/fake/path")
        assert len(result) == 2

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_filters_non_kicad_mod_files(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["valid.kicad_mod", "readme.txt", "other.lib"]

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "(footprint \"test\")"
            result = get_all_footprints_in_library("/fake/path")
            # Only the .kicad_mod file should be processed
            assert mock_open.call_count == 1


class TestListAvailableLibraries:
    """Tests for list_available_libraries function."""

    @patch("kicad_mcp.tools.footprint_tools.get_footprint_library_path")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_returns_sorted_libraries(self, mock_isdir, mock_listdir, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = True
        mock_listdir.return_value = ["Zebra.pretty", "Apple.pretty", "Mango.pretty"]
        mock_isdir.return_value = True

        libs = list_available_libraries()
        assert libs == ["Apple", "Mango", "Zebra"]

    @patch("kicad_mcp.tools.footprint_tools.get_footprint_library_path")
    @patch("os.path.exists")
    def test_returns_empty_if_path_not_exists(self, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = False

        libs = list_available_libraries()
        assert libs == []

    @patch("kicad_mcp.tools.footprint_tools.get_footprint_library_path")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_filters_non_pretty_directories(self, mock_isdir, mock_listdir, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = True
        mock_listdir.return_value = ["Valid.pretty", "readme.txt", "other.lib"]
        mock_isdir.return_value = True

        libs = list_available_libraries()
        assert libs == ["Valid"]


class TestFindSimilarFootprints:
    """Tests for find_similar_footprints function."""

    def test_exact_substring_match(self):
        footprints = ["R_0805_2012Metric", "R_0603_1608Metric", "C_0805_2012Metric"]
        similar = find_similar_footprints("0805", footprints)
        assert "R_0805_2012Metric" in similar
        assert "C_0805_2012Metric" in similar

    def test_limit_results(self):
        footprints = [f"Footprint{i}" for i in range(20)]
        similar = find_similar_footprints("Footprint", footprints, limit=5)
        assert len(similar) == 5

    def test_fuzzy_match(self):
        footprints = ["SOIC-8", "SSOP-8", "DIP-8"]
        similar = find_similar_footprints("SOIC", footprints)
        assert "SOIC-8" in similar


class TestReadFootprintFile:
    """Tests for read_footprint_file function."""

    @patch("kicad_mcp.tools.footprint_tools.get_footprint_library_path")
    @patch("os.path.exists")
    def test_returns_none_if_not_found(self, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = False

        result = read_footprint_file("NonExistent", "Footprint")
        assert result is None

    @patch("kicad_mcp.tools.footprint_tools.get_footprint_library_path")
    @patch("os.path.exists")
    @patch("builtins.open")
    def test_reads_file_content(self, mock_open, mock_exists, mock_path):
        mock_path.return_value = "/fake/path"
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "(footprint content)"

        result = read_footprint_file("Resistor_SMD", "R_0805")
        assert result == "(footprint content)"
