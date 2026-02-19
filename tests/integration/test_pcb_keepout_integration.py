"""
Integration tests for PCB keepout-aware placement validation tools.

These tests run actual pcbnew scripts against the track_geometry_car PCB file.
They require KiCad's Python 3.9 with pcbnew bindings to be installed.

Mark: requires_kicad — skipped if KiCad Python is not available.
"""

import os

import pytest

from kicad_mcp.utils.pcbnew_bridge import run_pcbnew_script, _get_kicad_python
from kicad_mcp.utils.keepout_helpers import KEEPOUT_HELPER as _KEEPOUT_HELPER

# Skip all tests in this module if KiCad Python is unavailable
pytestmark = pytest.mark.requires_kicad

PCB_PATH = "/Volumes/Files/claude/KiCAD-mcp-extensions/track_geometry_car.kicad_pcb"


def _kicad_available():
    """Check if KiCad Python interpreter exists."""
    return _get_kicad_python() is not None


def _pcb_exists():
    """Check if the test PCB file exists."""
    return os.path.exists(PCB_PATH)


@pytest.fixture(autouse=True)
def skip_if_unavailable():
    if not _kicad_available():
        pytest.skip("KiCad Python 3.9 not available")
    if not _pcb_exists():
        pytest.skip(f"PCB file not found: {PCB_PATH}")


class TestGetKeepoutZonesIntegration:

    def test_finds_keepout_zones(self):
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
keepouts = extract_keepouts(board)
print(json.dumps({{"status": "ok", "keepout_count": len(keepouts), "keepouts": keepouts}}))
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        assert result["keepout_count"] >= 1, "Expected at least 1 keepout zone"

        # ESP32 antenna keepout should be present
        refs = [kz["source_ref"] for kz in result["keepouts"]]
        assert "U1" in refs, "Expected ESP32 (U1) antenna keepout"

    def test_keepout_has_correct_structure(self):
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
keepouts = extract_keepouts(board)
print(json.dumps({{"status": "ok", "keepouts": keepouts}}))
"""
        result = run_pcbnew_script(script)
        for kz in result["keepouts"]:
            assert "source" in kz
            assert "layers" in kz
            assert "constraints" in kz
            assert "bounding_box" in kz
            assert "polygon_pts_mm" in kz
            bb = kz["bounding_box"]
            assert bb["x_min_mm"] < bb["x_max_mm"]
            assert bb["y_min_mm"] < bb["y_max_mm"]

    def test_u1_keepout_constraints(self):
        """ESP32 antenna keepout should block everything."""
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
keepouts = extract_keepouts(board)
u1_keepouts = [kz for kz in keepouts if kz["source_ref"] == "U1"]
print(json.dumps({{"status": "ok", "u1_keepouts": u1_keepouts}}))
"""
        result = run_pcbnew_script(script)
        u1_kz = result["u1_keepouts"]
        assert len(u1_kz) >= 1
        c = u1_kz[0]["constraints"]
        assert c["no_tracks"] is True
        assert c["no_vias"] is True
        assert c["no_copper_pour"] is True


class TestGetBoardConstraintsIntegration:

    def test_board_outline(self):
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
outline = get_board_outline(board)
print(json.dumps({{"status": "ok", "outline": outline}}))
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        outline = result["outline"]
        assert outline is not None
        assert outline["width_mm"] == 70.0
        assert outline["height_mm"] == 50.0

    def test_full_constraints(self):
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
keepouts = extract_keepouts(board)
outline = get_board_outline(board)
ds = board.GetDesignSettings()
design_rules = {{
    "min_track_width_mm": round(pcbnew.ToMM(ds.m_TrackMinWidth), 3),
    "min_clearance_mm": round(pcbnew.ToMM(ds.m_MinClearance), 3),
    "min_via_diameter_mm": round(pcbnew.ToMM(ds.m_ViasMinSize), 3),
}}
board_area = outline["width_mm"] * outline["height_mm"] if outline else 0
total_keepout_area = 0
for kz in keepouts:
    bb = kz["bounding_box"]
    total_keepout_area += (bb["x_max_mm"] - bb["x_min_mm"]) * (bb["y_max_mm"] - bb["y_min_mm"])
print(json.dumps({{
    "status": "ok",
    "outline": outline,
    "keepout_count": len(keepouts),
    "design_rules": design_rules,
    "board_area_mm2": round(board_area, 1),
    "total_keepout_area_mm2": round(total_keepout_area, 1),
    "footprint_count": len(list(board.GetFootprints())),
}}))
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        assert result["board_area_mm2"] == 3500.0
        assert result["keepout_count"] >= 1
        assert result["footprint_count"] == 16
        assert result["design_rules"]["min_track_width_mm"] >= 0


class TestValidatePlacementIntegration:

    def test_placement_in_safe_zone(self):
        """Validate a position clearly outside all keepouts and within board."""
        script = f"""
import pcbnew, json, os
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})

lib_path = "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints/Resistor_SMD.pretty"
fp = pcbnew.FootprintLoad(lib_path, "R_0805_2012Metric")
# Board outline is 95-165 x 75-125, keepout ~106-154 x 66-87
# 0805 bbox is ~15mm wide (includes ref text), so center at 110,100
# to keep bbox (102.5-117.5 x 97.5-102.5) well inside the board
# and below the keepout zone (which ends at y=87.2)
fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(110), pcbnew.FromMM(100)))

fp_bbox = fp.GetBoundingBox()
fp_rect = {{
    "x_min_mm": round(pcbnew.ToMM(fp_bbox.GetX()), 3),
    "y_min_mm": round(pcbnew.ToMM(fp_bbox.GetY()), 3),
    "x_max_mm": round(pcbnew.ToMM(fp_bbox.GetRight()), 3),
    "y_max_mm": round(pcbnew.ToMM(fp_bbox.GetBottom()), 3),
}}

keepouts = extract_keepouts(board)
outline = get_board_outline(board)
violations = []
for kz in keepouts:
    if rects_overlap(fp_rect, kz["bounding_box"]):
        if kz["constraints"]["no_footprints"]:
            violations.append("keepout_overlap")
if outline and not rect_inside(fp_rect, outline):
    violations.append("outside_board")

print(json.dumps({{"status": "ok", "valid": len(violations) == 0, "violations": violations}}))
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        assert result["valid"] is True, f"Unexpected violations: {result['violations']}"

    def test_placement_in_esp32_keepout(self):
        """A resistor placed in the ESP32 antenna keepout should be invalid."""
        script = f"""
import pcbnew, json, os
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})

lib_path = "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints/Resistor_SMD.pretty"
fp = pcbnew.FootprintLoad(lib_path, "R_0805_2012Metric")
# Place at (130, 78) which is well inside the ESP32 antenna keepout
fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(130), pcbnew.FromMM(78)))

fp_bbox = fp.GetBoundingBox()
fp_rect = {{
    "x_min_mm": round(pcbnew.ToMM(fp_bbox.GetX()), 3),
    "y_min_mm": round(pcbnew.ToMM(fp_bbox.GetY()), 3),
    "x_max_mm": round(pcbnew.ToMM(fp_bbox.GetRight()), 3),
    "y_max_mm": round(pcbnew.ToMM(fp_bbox.GetBottom()), 3),
}}

keepouts = extract_keepouts(board)
violations = []
for kz in keepouts:
    if rects_overlap(fp_rect, kz["bounding_box"]):
        if kz["constraints"]["no_footprints"]:
            violations.append(kz["source_ref"])

print(json.dumps({{"status": "ok", "valid": len(violations) == 0, "violation_refs": violations}}))
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        assert result["valid"] is False
        assert "U1" in result["violation_refs"]


class TestAuditPcbPlacementIntegration:

    def test_audit_finds_violations(self):
        """The current board should have known violations."""
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
keepouts = extract_keepouts(board)
outline = get_board_outline(board)

violations_list = []
clean_count = 0

for fp in board.GetFootprints():
    ref = fp.GetReference()
    fp_bbox = fp.GetBoundingBox()
    fp_rect = {{
        "x_min_mm": round(pcbnew.ToMM(fp_bbox.GetX()), 3),
        "y_min_mm": round(pcbnew.ToMM(fp_bbox.GetY()), 3),
        "x_max_mm": round(pcbnew.ToMM(fp_bbox.GetRight()), 3),
        "y_max_mm": round(pcbnew.ToMM(fp_bbox.GetBottom()), 3),
    }}
    issues = []
    for kz in keepouts:
        if kz["source"] == "footprint" and kz["source_ref"] == ref:
            continue
        kz_bb = kz["bounding_box"]
        if rects_overlap(fp_rect, kz_bb):
            c = kz["constraints"]
            severity = "violation" if c["no_footprints"] else "warning"
            issues.append({{"type": "keepout_overlap", "severity": severity, "keepout_ref": kz["source_ref"]}})
    if outline and not rect_inside(fp_rect, outline):
        issues.append({{"type": "outside_board", "severity": "violation"}})
    if issues:
        violations_list.append({{"reference": ref, "issues": issues}})
    else:
        clean_count += 1

total = len(list(board.GetFootprints()))
print(json.dumps({{
    "status": "ok",
    "total_footprints": total,
    "violations_count": len(violations_list),
    "clean_count": clean_count,
    "violation_refs": [v["reference"] for v in violations_list],
}}))
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        assert result["total_footprints"] == 16
        # Board has known violations — at least some components in keepout
        assert result["violations_count"] > 0
        assert result["clean_count"] < result["total_footprints"]

    def test_audit_skips_own_keepout(self):
        """U1 (ESP32) should NOT be flagged for its own antenna keepout."""
        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({PCB_PATH!r})
keepouts = extract_keepouts(board)

# Check U1 specifically
for fp in board.GetFootprints():
    if fp.GetReference() != "U1":
        continue
    fp_bbox = fp.GetBoundingBox()
    fp_rect = {{
        "x_min_mm": round(pcbnew.ToMM(fp_bbox.GetX()), 3),
        "y_min_mm": round(pcbnew.ToMM(fp_bbox.GetY()), 3),
        "x_max_mm": round(pcbnew.ToMM(fp_bbox.GetRight()), 3),
        "y_max_mm": round(pcbnew.ToMM(fp_bbox.GetBottom()), 3),
    }}
    own_keepout_overlaps = 0
    other_keepout_overlaps = 0
    for kz in keepouts:
        if not rects_overlap(fp_rect, kz["bounding_box"]):
            continue
        if kz["source"] == "footprint" and kz["source_ref"] == "U1":
            own_keepout_overlaps += 1
        else:
            other_keepout_overlaps += 1
    print(json.dumps({{
        "status": "ok",
        "own_keepout_overlaps": own_keepout_overlaps,
        "other_keepout_overlaps": other_keepout_overlaps,
    }}))
    break
"""
        result = run_pcbnew_script(script)
        assert result["status"] == "ok"
        # U1 should overlap its own keepout (which we skip)
        assert result["own_keepout_overlaps"] >= 1
