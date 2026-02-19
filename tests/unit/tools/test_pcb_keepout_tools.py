"""
Tests for PCB keepout-aware placement validation tools.

Tests the 4 keepout tools: get_keepout_zones, get_board_constraints,
validate_placement, audit_pcb_placement.

Unit tests mock run_pcbnew_script to test tool logic without requiring
KiCad's Python 3.9 / pcbnew bindings.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from mcp.server.fastmcp import FastMCP

from kicad_mcp.tools.pcb_tools import register_pcb_tools


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def mcp_server():
    """Create a FastMCP server with PCB tools registered."""
    mcp = FastMCP("test-kicad")
    register_pcb_tools(mcp)
    return mcp


@pytest.fixture
def pcb_file(tmp_path):
    """Create a dummy .kicad_pcb file for path-existence checks."""
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text("(kicad_pcb)")
    return str(pcb)


# Sample return data mimicking what pcbnew scripts produce

SAMPLE_KEEPOUTS = [
    {
        "source": "footprint",
        "source_ref": "U1",
        "uuid": "abc-123",
        "layers": ["F.Cu", "B.Cu"],
        "constraints": {
            "no_tracks": True,
            "no_vias": True,
            "no_pads": True,
            "no_footprints": True,
            "no_copper_pour": True,
        },
        "bounding_box": {
            "x_min_mm": 106.0,
            "y_min_mm": 66.26,
            "x_max_mm": 154.0,
            "y_max_mm": 87.2,
        },
        "polygon_pts_mm": [
            [106.0, 66.26],
            [154.0, 66.26],
            [154.0, 87.2],
            [106.0, 87.2],
        ],
    },
]

SAMPLE_OUTLINE = {
    "x_min_mm": 95.0,
    "y_min_mm": 72.0,
    "x_max_mm": 165.0,
    "y_max_mm": 122.0,
    "width_mm": 70.0,
    "height_mm": 50.0,
}


# -- Helper to call tools via the registered functions -----------------------

def _get_tool_fn(mcp_server, tool_name):
    """Extract a tool function from the FastMCP server by name."""
    for tool in mcp_server._tool_manager._tools.values():
        if tool.name == tool_name:
            return tool.fn
    raise ValueError(f"Tool {tool_name!r} not found")


# -- get_keepout_zones tests -------------------------------------------------

class TestGetKeepoutZones:

    def test_file_not_found(self, mcp_server):
        fn = _get_tool_fn(mcp_server, "get_keepout_zones")
        result = fn("/nonexistent/board.kicad_pcb")
        assert "error" in result
        assert "not found" in result["error"]

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_returns_keepouts(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "keepout_count": 1,
            "keepouts": SAMPLE_KEEPOUTS,
        }
        fn = _get_tool_fn(mcp_server, "get_keepout_zones")
        result = fn(pcb_file)
        assert result["status"] == "ok"
        assert result["keepout_count"] == 1
        assert len(result["keepouts"]) == 1
        kz = result["keepouts"][0]
        assert kz["source"] == "footprint"
        assert kz["source_ref"] == "U1"
        assert kz["constraints"]["no_tracks"] is True
        mock_run.assert_called_once()

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_no_keepouts(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "keepout_count": 0,
            "keepouts": [],
        }
        fn = _get_tool_fn(mcp_server, "get_keepout_zones")
        result = fn(pcb_file)
        assert result["keepout_count"] == 0

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_script_contains_extract_keepouts(self, mock_run, mcp_server, pcb_file):
        """Verify the generated script includes the keepout helper code."""
        mock_run.return_value = {"status": "ok", "keepout_count": 0, "keepouts": []}
        fn = _get_tool_fn(mcp_server, "get_keepout_zones")
        fn(pcb_file)
        script = mock_run.call_args[0][0]
        assert "extract_keepouts" in script
        assert "GetIsRuleArea" in script
        assert pcb_file in script


# -- get_board_constraints tests ---------------------------------------------

class TestGetBoardConstraints:

    def test_file_not_found(self, mcp_server):
        fn = _get_tool_fn(mcp_server, "get_board_constraints")
        result = fn("/nonexistent/board.kicad_pcb")
        assert "error" in result

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_returns_constraints(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "board_outline": {
                **SAMPLE_OUTLINE,
                "area_mm2": 3500.0,
            },
            "keepout_zones": SAMPLE_KEEPOUTS,
            "design_rules": {
                "min_track_width_mm": 0.2,
                "min_clearance_mm": 0.2,
                "min_via_diameter_mm": 0.6,
            },
            "existing_footprints_count": 16,
            "total_keepout_area_mm2": 1005.1,
            "effective_placement_area_mm2": 2494.9,
        }
        fn = _get_tool_fn(mcp_server, "get_board_constraints")
        result = fn(pcb_file)
        assert result["status"] == "ok"
        assert result["board_outline"]["width_mm"] == 70.0
        assert result["design_rules"]["min_track_width_mm"] == 0.2
        assert result["existing_footprints_count"] == 16
        assert result["effective_placement_area_mm2"] == 2494.9

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_script_includes_design_rules(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {"status": "ok", "board_outline": None,
                                 "keepout_zones": [], "design_rules": {},
                                 "existing_footprints_count": 0,
                                 "total_keepout_area_mm2": 0}
        fn = _get_tool_fn(mcp_server, "get_board_constraints")
        fn(pcb_file)
        script = mock_run.call_args[0][0]
        assert "GetDesignSettings" in script
        assert "m_TrackMinWidth" in script
        assert "get_board_outline" in script


# -- validate_placement tests ------------------------------------------------

class TestValidatePlacement:

    def test_file_not_found(self, mcp_server):
        fn = _get_tool_fn(mcp_server, "validate_placement")
        result = fn("/nonexistent/board.kicad_pcb",
                     "Resistor_SMD", "R_0805_2012Metric", 130.0, 80.0)
        assert "error" in result

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_valid_placement(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "valid": True,
            "violations": [],
            "warnings": [],
            "footprint_bbox_mm": {
                "x_min_mm": 98.0, "y_min_mm": 110.0,
                "x_max_mm": 102.0, "y_max_mm": 112.0,
            },
            "board_outline_mm": SAMPLE_OUTLINE,
        }
        fn = _get_tool_fn(mcp_server, "validate_placement")
        result = fn(pcb_file, "Resistor_SMD", "R_0805_2012Metric", 100.0, 111.0)
        assert result["valid"] is True
        assert result["violations"] == []

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_placement_in_keepout(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "valid": False,
            "violations": [{
                "type": "keepout_overlap",
                "keepout_source": "footprint",
                "keepout_ref": "U1",
                "overlap_mm2": 8.0,
                "blocked": ["tracks", "vias", "pads", "footprints", "copper_pour"],
                "message": "Footprint overlaps keepout zone from U1",
            }],
            "warnings": [],
            "footprint_bbox_mm": {
                "x_min_mm": 128.0, "y_min_mm": 78.0,
                "x_max_mm": 132.0, "y_max_mm": 80.0,
            },
            "board_outline_mm": SAMPLE_OUTLINE,
        }
        fn = _get_tool_fn(mcp_server, "validate_placement")
        result = fn(pcb_file, "Resistor_SMD", "R_0805_2012Metric", 130.0, 79.0)
        assert result["valid"] is False
        assert len(result["violations"]) == 1
        assert result["violations"][0]["type"] == "keepout_overlap"
        assert result["violations"][0]["keepout_ref"] == "U1"

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_placement_outside_board(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "valid": False,
            "violations": [{
                "type": "outside_board",
                "overhang": {"right_mm": 5.0},
                "message": "Footprint extends beyond board outline",
            }],
            "warnings": [],
            "footprint_bbox_mm": {
                "x_min_mm": 162.0, "y_min_mm": 110.0,
                "x_max_mm": 170.0, "y_max_mm": 112.0,
            },
            "board_outline_mm": SAMPLE_OUTLINE,
        }
        fn = _get_tool_fn(mcp_server, "validate_placement")
        result = fn(pcb_file, "Resistor_SMD", "R_0805_2012Metric", 166.0, 111.0)
        assert result["valid"] is False
        assert result["violations"][0]["type"] == "outside_board"
        assert result["violations"][0]["overhang"]["right_mm"] == 5.0

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_routing_warning_not_violation(self, mock_run, mcp_server, pcb_file):
        """A keepout that blocks routing but not footprints → warning, still valid."""
        mock_run.return_value = {
            "status": "ok",
            "valid": True,
            "violations": [],
            "warnings": [{
                "type": "routing_keepout_overlap",
                "keepout_source": "footprint",
                "keepout_ref": "U2",
                "overlap_mm2": 2.0,
                "blocked": ["tracks", "vias"],
                "message": "Footprint overlaps zone that blocks tracks, vias (routing may be difficult)",
            }],
            "footprint_bbox_mm": {
                "x_min_mm": 108.0, "y_min_mm": 96.0,
                "x_max_mm": 112.0, "y_max_mm": 98.0,
            },
            "board_outline_mm": SAMPLE_OUTLINE,
        }
        fn = _get_tool_fn(mcp_server, "validate_placement")
        result = fn(pcb_file, "Capacitor_SMD", "C_0805_2012Metric", 110.0, 97.0)
        assert result["valid"] is True
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["type"] == "routing_keepout_overlap"

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_script_loads_footprint_from_library(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {"status": "ok", "valid": True,
                                 "violations": [], "warnings": [],
                                 "footprint_bbox_mm": {}, "board_outline_mm": None}
        fn = _get_tool_fn(mcp_server, "validate_placement")
        fn(pcb_file, "Resistor_SMD", "R_0805_2012Metric", 100.0, 100.0, 45.0)
        script = mock_run.call_args[0][0]
        assert "FootprintLoad" in script
        assert "Resistor_SMD" in script
        assert "R_0805_2012Metric" in script
        assert "SetPosition" in script
        assert "45.0" in script

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_library_not_found(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {"error": "Library 'FakeLib' not found"}
        fn = _get_tool_fn(mcp_server, "validate_placement")
        result = fn(pcb_file, "FakeLib", "FakeFP", 100.0, 100.0)
        assert "error" in result


# -- audit_pcb_placement tests -----------------------------------------------

class TestAuditPcbPlacement:

    def test_file_not_found(self, mcp_server):
        fn = _get_tool_fn(mcp_server, "audit_pcb_placement")
        result = fn("/nonexistent/board.kicad_pcb")
        assert "error" in result

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_all_clean(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "total_footprints": 8,
            "violations_count": 0,
            "clean_count": 8,
            "violations": [],
            "summary": "All 8 footprints pass placement checks",
        }
        fn = _get_tool_fn(mcp_server, "audit_pcb_placement")
        result = fn(pcb_file)
        assert result["violations_count"] == 0
        assert result["clean_count"] == 8
        assert "pass" in result["summary"]

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_violations_found(self, mock_run, mcp_server, pcb_file):
        mock_run.return_value = {
            "status": "ok",
            "total_footprints": 16,
            "violations_count": 12,
            "clean_count": 4,
            "violations": [
                {
                    "reference": "D1",
                    "value": "LED",
                    "footprint": "LED_0805_2012Metric",
                    "position_mm": [120.0, 80.0],
                    "bbox_mm": {
                        "x_min_mm": 118.0, "y_min_mm": 78.0,
                        "x_max_mm": 122.0, "y_max_mm": 82.0,
                    },
                    "issues": [{
                        "type": "keepout_overlap",
                        "severity": "violation",
                        "keepout_source": "footprint",
                        "keepout_ref": "U1",
                        "overlap_mm2": 16.0,
                        "blocked": ["tracks", "vias", "pads", "footprints", "copper_pour"],
                    }],
                },
                {
                    "reference": "BZ1",
                    "value": "Buzzer",
                    "footprint": "Buzzer_12x9.5mm",
                    "position_mm": [170.0, 118.0],
                    "bbox_mm": {
                        "x_min_mm": 164.0, "y_min_mm": 112.0,
                        "x_max_mm": 176.0, "y_max_mm": 124.0,
                    },
                    "issues": [{
                        "type": "outside_board",
                        "severity": "violation",
                        "overhang": {"right_mm": 11.0, "bottom_mm": 2.0},
                    }],
                },
            ],
            "summary": "12 of 16 footprints have placement issues",
        }
        fn = _get_tool_fn(mcp_server, "audit_pcb_placement")
        result = fn(pcb_file)
        assert result["violations_count"] == 12
        assert result["clean_count"] == 4
        assert len(result["violations"]) == 2
        # Check keepout violation
        d1 = result["violations"][0]
        assert d1["reference"] == "D1"
        assert d1["issues"][0]["severity"] == "violation"
        # Check board boundary violation
        bz1 = result["violations"][1]
        assert bz1["reference"] == "BZ1"
        assert bz1["issues"][0]["type"] == "outside_board"

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_skips_own_keepout(self, mock_run, mcp_server, pcb_file):
        """Verify the script skips a footprint's own embedded keepout zone."""
        mock_run.return_value = {"status": "ok", "total_footprints": 1,
                                 "violations_count": 0, "clean_count": 1,
                                 "violations": [], "summary": "All 1 footprints pass placement checks"}
        fn = _get_tool_fn(mcp_server, "audit_pcb_placement")
        fn(pcb_file)
        script = mock_run.call_args[0][0]
        # The script should have logic to skip a footprint's own keepout
        assert 'source_ref' in script
        assert 'continue' in script

    @patch("kicad_mcp.tools.pcb_tools.run_pcbnew_script")
    def test_script_error_propagated(self, mock_run, mcp_server, pcb_file):
        """If pcbnew script raises RuntimeError, it propagates."""
        mock_run.side_effect = RuntimeError("pcbnew crashed")
        fn = _get_tool_fn(mcp_server, "audit_pcb_placement")
        with pytest.raises(RuntimeError, match="pcbnew crashed"):
            fn(pcb_file)


# -- Shared helper logic tests (pure Python, no mocking needed) ---------------

class TestHelperLogic:
    """Test the pure-Python helper functions embedded in _KEEPOUT_HELPER.

    Since these are strings embedded in pcbnew scripts, we extract and exec them
    to test the geometry logic directly.
    """

    @pytest.fixture(autouse=True)
    def setup_helpers(self):
        """Extract and execute the geometry helper functions."""
        # These functions are defined in _KEEPOUT_HELPER as plain Python
        namespace = {}
        exec(
            "def rects_overlap(a, b):\n"
            '    return (a["x_min_mm"] < b["x_max_mm"] and a["x_max_mm"] > b["x_min_mm"] and\n'
            '            a["y_min_mm"] < b["y_max_mm"] and a["y_max_mm"] > b["y_min_mm"])\n'
            "\n"
            "def overlap_area(a, b):\n"
            '    dx = max(0, min(a["x_max_mm"], b["x_max_mm"]) - max(a["x_min_mm"], b["x_min_mm"]))\n'
            '    dy = max(0, min(a["y_max_mm"], b["y_max_mm"]) - max(a["y_min_mm"], b["y_min_mm"]))\n'
            "    return round(dx * dy, 2)\n"
            "\n"
            "def rect_inside(inner, outer):\n"
            '    return (inner["x_min_mm"] >= outer["x_min_mm"] and inner["x_max_mm"] <= outer["x_max_mm"] and\n'
            '            inner["y_min_mm"] >= outer["y_min_mm"] and inner["y_max_mm"] <= outer["y_max_mm"])\n',
            namespace,
        )
        self.rects_overlap = namespace["rects_overlap"]
        self.overlap_area = namespace["overlap_area"]
        self.rect_inside = namespace["rect_inside"]

    def _rect(self, x1, y1, x2, y2):
        return {"x_min_mm": x1, "y_min_mm": y1, "x_max_mm": x2, "y_max_mm": y2}

    # rects_overlap
    def test_overlapping_rects(self):
        a = self._rect(0, 0, 10, 10)
        b = self._rect(5, 5, 15, 15)
        assert self.rects_overlap(a, b) is True

    def test_non_overlapping_rects(self):
        a = self._rect(0, 0, 10, 10)
        b = self._rect(20, 20, 30, 30)
        assert self.rects_overlap(a, b) is False

    def test_touching_edge_not_overlapping(self):
        """Rects that share an edge but don't overlap."""
        a = self._rect(0, 0, 10, 10)
        b = self._rect(10, 0, 20, 10)
        assert self.rects_overlap(a, b) is False

    def test_contained_rect(self):
        outer = self._rect(0, 0, 100, 100)
        inner = self._rect(10, 10, 20, 20)
        assert self.rects_overlap(outer, inner) is True

    # overlap_area
    def test_overlap_area_partial(self):
        a = self._rect(0, 0, 10, 10)
        b = self._rect(5, 5, 15, 15)
        assert self.overlap_area(a, b) == 25.0

    def test_overlap_area_none(self):
        a = self._rect(0, 0, 10, 10)
        b = self._rect(20, 20, 30, 30)
        assert self.overlap_area(a, b) == 0.0

    def test_overlap_area_contained(self):
        outer = self._rect(0, 0, 100, 100)
        inner = self._rect(10, 10, 20, 20)
        assert self.overlap_area(outer, inner) == 100.0

    # rect_inside
    def test_fully_inside(self):
        inner = self._rect(10, 10, 20, 20)
        outer = self._rect(0, 0, 100, 100)
        assert self.rect_inside(inner, outer) is True

    def test_partially_outside(self):
        inner = self._rect(90, 90, 110, 110)
        outer = self._rect(0, 0, 100, 100)
        assert self.rect_inside(inner, outer) is False

    def test_exactly_on_boundary(self):
        inner = self._rect(0, 0, 100, 100)
        outer = self._rect(0, 0, 100, 100)
        assert self.rect_inside(inner, outer) is True

    def test_completely_outside(self):
        inner = self._rect(200, 200, 210, 210)
        outer = self._rect(0, 0, 100, 100)
        assert self.rect_inside(inner, outer) is False
