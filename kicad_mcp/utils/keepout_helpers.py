"""
Shared pcbnew helper code for keepout zone extraction and geometry checks.

This module provides a single source of truth for the Python code that runs
inside KiCad's Python 3.9 subprocess (via pcbnew_bridge). The code is stored
as a string constant because it gets embedded into dynamically generated
pcbnew scripts — it cannot be imported as a normal module.

Used by:
  - kicad_mcp/tools/pcb_tools.py (the 4 keepout validation tools)
  - tests/integration/test_pcb_keepout_integration.py

If you modify these helpers, both consumers pick up the change automatically.
"""

# Python code string that gets embedded in pcbnew subprocess scripts.
# Provides: extract_keepouts(), get_board_outline(), rects_overlap(),
#           overlap_area(), rect_inside()
KEEPOUT_HELPER = """
def extract_keepouts(board):
    import pcbnew
    keepouts = []
    def process_zone(zone, source, source_ref=""):
        if not zone.GetIsRuleArea():
            return None
        bbox = zone.GetBoundingBox()
        layers = []
        for lid in zone.GetLayerSet().Seq():
            layers.append(board.GetLayerName(lid))
        poly_set = zone.Outline()
        pts = []
        if poly_set.OutlineCount() > 0:
            ol = poly_set.Outline(0)
            for i in range(ol.PointCount()):
                pt = ol.CPoint(i)
                pts.append([round(pcbnew.ToMM(pt.x), 3), round(pcbnew.ToMM(pt.y), 3)])
        try:
            uuid_str = zone.m_Uuid.AsString()
        except Exception:
            uuid_str = ""
        return {
            "source": source,
            "source_ref": source_ref,
            "uuid": uuid_str,
            "layers": layers,
            "constraints": {
                "no_tracks": zone.GetDoNotAllowTracks(),
                "no_vias": zone.GetDoNotAllowVias(),
                "no_pads": zone.GetDoNotAllowPads(),
                "no_footprints": zone.GetDoNotAllowFootprints(),
                "no_copper_pour": zone.GetDoNotAllowCopperPour(),
            },
            "bounding_box": {
                "x_min_mm": round(pcbnew.ToMM(bbox.GetX()), 3),
                "y_min_mm": round(pcbnew.ToMM(bbox.GetY()), 3),
                "x_max_mm": round(pcbnew.ToMM(bbox.GetRight()), 3),
                "y_max_mm": round(pcbnew.ToMM(bbox.GetBottom()), 3),
            },
            "polygon_pts_mm": pts,
        }
    for zone in board.Zones():
        info = process_zone(zone, "board")
        if info:
            keepouts.append(info)
    for fp in board.GetFootprints():
        try:
            for zone in fp.Zones():
                info = process_zone(zone, "footprint", fp.GetReference())
                if info:
                    keepouts.append(info)
        except AttributeError:
            pass
    return keepouts

def get_board_outline(board):
    import pcbnew
    edge_cuts_id = board.GetLayerID("Edge.Cuts")
    xs, ys = [], []
    for drawing in board.GetDrawings():
        if drawing.GetLayer() == edge_cuts_id:
            start = drawing.GetStart()
            end = drawing.GetEnd()
            xs.extend([pcbnew.ToMM(start.x), pcbnew.ToMM(end.x)])
            ys.extend([pcbnew.ToMM(start.y), pcbnew.ToMM(end.y)])
    if not xs:
        return None
    return {
        "x_min_mm": round(min(xs), 3),
        "y_min_mm": round(min(ys), 3),
        "x_max_mm": round(max(xs), 3),
        "y_max_mm": round(max(ys), 3),
        "width_mm": round(max(xs) - min(xs), 3),
        "height_mm": round(max(ys) - min(ys), 3),
    }

def rects_overlap(a, b):
    return (a["x_min_mm"] < b["x_max_mm"] and a["x_max_mm"] > b["x_min_mm"] and
            a["y_min_mm"] < b["y_max_mm"] and a["y_max_mm"] > b["y_min_mm"])

def overlap_area(a, b):
    dx = max(0, min(a["x_max_mm"], b["x_max_mm"]) - max(a["x_min_mm"], b["x_min_mm"]))
    dy = max(0, min(a["y_max_mm"], b["y_max_mm"]) - max(a["y_min_mm"], b["y_min_mm"]))
    return round(dx * dy, 2)

def rect_inside(inner, outer):
    return (inner["x_min_mm"] >= outer["x_min_mm"] and inner["x_max_mm"] <= outer["x_max_mm"] and
            inner["y_min_mm"] >= outer["y_min_mm"] and inner["y_max_mm"] <= outer["y_max_mm"])
"""
