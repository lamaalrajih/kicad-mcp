"""
PCB editing tools using KiCad's pcbnew Python API.

These tools manipulate .kicad_pcb files by running pcbnew operations
via KiCad's bundled Python interpreter.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP, Context

from kicad_mcp.utils.pcbnew_bridge import run_pcbnew_script
from kicad_mcp.utils.keepout_helpers import KEEPOUT_HELPER

logger = logging.getLogger(__name__)


def register_pcb_tools(mcp: FastMCP) -> None:
    """Register PCB editing tools with the MCP server."""

    @mcp.tool()
    def load_pcb(pcb_path: str) -> Dict[str, Any]:
        """Load a .kicad_pcb file and return its summary.

        Args:
            pcb_path: Absolute path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"File not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})
footprints = board.GetFootprints()
tracks = board.GetTracks()

fp_list = []
for fp in footprints:
    pos = fp.GetPosition()
    fp_list.append({{
        "reference": fp.GetReference(),
        "value": fp.GetValue(),
        "footprint": fp.GetFPID().GetUniStringLibItemName(),
        "x_mm": pcbnew.ToMM(pos.x),
        "y_mm": pcbnew.ToMM(pos.y),
        "layer": board.GetLayerName(fp.GetLayer()),
    }})

print(json.dumps({{
    "status": "ok",
    "file": {pcb_path!r},
    "footprint_count": len(fp_list),
    "track_count": len(tracks),
    "footprints": fp_list,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def create_pcb(pcb_path: str) -> Dict[str, Any]:
        """Create a new empty .kicad_pcb file.

        Args:
            pcb_path: Absolute path for the new PCB file.
        """
        script = f"""
import pcbnew, json

board = pcbnew.CreateEmptyBoard()
board.SetFileName({pcb_path!r})
board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "file": {pcb_path!r},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def place_footprint(
        pcb_path: str,
        library: str,
        footprint_name: str,
        reference: str,
        value: str,
        x_mm: float,
        y_mm: float,
        rotation_deg: float = 0.0,
        layer: str = "F.Cu",
    ) -> Dict[str, Any]:
        """Place a footprint on the PCB from a KiCad library.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            library: Footprint library name (e.g., "Resistor_THT").
            footprint_name: Footprint name within the library.
            reference: Component reference (e.g., "R1").
            value: Component value (e.g., "330").
            x_mm: X position in millimeters.
            y_mm: Y position in millimeters.
            rotation_deg: Rotation angle in degrees (default 0).
            layer: PCB layer, "F.Cu" or "B.Cu" (default "F.Cu").
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json, os, glob

board = pcbnew.LoadBoard({pcb_path!r})

# Find the footprint library path
lib_name = {library!r}
fp_name = {footprint_name!r}

# Search standard KiCad footprint library locations
lib_search_paths = [
    "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints",
    os.path.expanduser("~/Documents/KiCad/footprints"),
    "/usr/share/kicad/footprints",
]

lib_path = None
for search_path in lib_search_paths:
    candidate = os.path.join(search_path, lib_name + ".pretty")
    if os.path.isdir(candidate):
        lib_path = candidate
        break

if not lib_path:
    print(json.dumps({{"error": f"Library '{{lib_name}}' not found"}}))
    raise SystemExit(0)

fp = pcbnew.FootprintLoad(lib_path, fp_name)
if fp is None:
    print(json.dumps({{"error": f"Footprint '{{fp_name}}' not found in '{{lib_name}}'"}}))
    raise SystemExit(0)

fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pcbnew.FromMM({y_mm})))
fp.SetReference({reference!r})
fp.SetValue({value!r})

if {rotation_deg} != 0:
    fp.SetOrientationDegrees({rotation_deg})

if {layer!r} == "B.Cu":
    fp.Flip(fp.GetPosition(), False)

board.Add(fp)
board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "placed": {{
        "reference": {reference!r},
        "footprint": f"{{lib_name}}:{{fp_name}}",
        "x_mm": {x_mm},
        "y_mm": {y_mm},
        "rotation": {rotation_deg},
        "layer": {layer!r},
    }},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def add_trace(
        pcb_path: str,
        start_x_mm: float,
        start_y_mm: float,
        end_x_mm: float,
        end_y_mm: float,
        width_mm: float = 0.25,
        layer: str = "F.Cu",
        net_name: str = "",
    ) -> Dict[str, Any]:
        """Add a copper trace between two points on the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            start_x_mm: Start X position in mm.
            start_y_mm: Start Y position in mm.
            end_x_mm: End X position in mm.
            end_y_mm: End Y position in mm.
            width_mm: Trace width in mm (default 0.25).
            layer: Copper layer name (default "F.Cu").
            net_name: Net name to assign (empty for unassigned).
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

track = pcbnew.PCB_TRACK(board)
track.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM({start_x_mm}), pcbnew.FromMM({start_y_mm})))
track.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM({end_x_mm}), pcbnew.FromMM({end_y_mm})))
track.SetWidth(pcbnew.FromMM({width_mm}))
track.SetLayer(board.GetLayerID({layer!r}))

net_name = {net_name!r}
if net_name:
    net = board.FindNet(net_name)
    if net:
        track.SetNet(net)

board.Add(track)
board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "trace": {{
        "start": [{start_x_mm}, {start_y_mm}],
        "end": [{end_x_mm}, {end_y_mm}],
        "width_mm": {width_mm},
        "layer": {layer!r},
        "net": net_name,
    }},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def add_board_outline(
        pcb_path: str,
        x_mm: float,
        y_mm: float,
        width_mm: float,
        height_mm: float,
    ) -> Dict[str, Any]:
        """Add a rectangular board outline (Edge.Cuts) to the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            x_mm: Top-left X position in mm.
            y_mm: Top-left Y position in mm.
            width_mm: Board width in mm.
            height_mm: Board height in mm.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

x = pcbnew.FromMM({x_mm})
y = pcbnew.FromMM({y_mm})
w = pcbnew.FromMM({width_mm})
h = pcbnew.FromMM({height_mm})

corners = [
    (x, y), (x + w, y), (x + w, y + h), (x, y + h)
]

for i in range(4):
    seg = pcbnew.PCB_SHAPE(board)
    seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
    seg.SetStart(pcbnew.VECTOR2I(corners[i][0], corners[i][1]))
    seg.SetEnd(pcbnew.VECTOR2I(corners[(i+1)%4][0], corners[(i+1)%4][1]))
    seg.SetLayer(pcbnew.Edge_Cuts)
    seg.SetWidth(pcbnew.FromMM(0.1))
    board.Add(seg)

board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "outline": {{
        "x_mm": {x_mm},
        "y_mm": {y_mm},
        "width_mm": {width_mm},
        "height_mm": {height_mm},
    }},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def add_net(
        pcb_path: str,
        net_name: str,
    ) -> Dict[str, Any]:
        """Add a named net to the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            net_name: Name for the net (e.g., "+3V3", "GND", "SDA").
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

existing = board.FindNet({net_name!r})
if existing and existing.GetNetCode() > 0:
    print(json.dumps({{
        "status": "ok",
        "net": {net_name!r},
        "net_code": existing.GetNetCode(),
        "note": "Net already exists",
    }}))
    raise SystemExit(0)

netinfo = pcbnew.NETINFO_ITEM(board, {net_name!r})
board.Add(netinfo)
board.Save(board.GetFileName())

net = board.FindNet({net_name!r})
print(json.dumps({{
    "status": "ok",
    "net": {net_name!r},
    "net_code": net.GetNetCode() if net else -1,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def assign_pad_net(
        pcb_path: str,
        reference: str,
        pad_number: str,
        net_name: str,
    ) -> Dict[str, Any]:
        """Assign a net to a specific pad on a footprint.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            reference: Component reference (e.g., "R1").
            pad_number: Pad number (e.g., "1", "2").
            net_name: Net name to assign.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

fp = board.FindFootprintByReference({reference!r})
if fp is None:
    print(json.dumps({{"error": f"Footprint {reference!r} not found"}}))
    raise SystemExit(0)

net = board.FindNet({net_name!r})
if net is None or net.GetNetCode() == 0:
    print(json.dumps({{"error": f"Net {net_name!r} not found"}}))
    raise SystemExit(0)

pad_found = False
for pad in fp.Pads():
    if pad.GetNumber() == {pad_number!r}:
        pad.SetNet(net)
        pad_found = True
        break

if not pad_found:
    print(json.dumps({{"error": f"Pad {pad_number!r} not found on {reference!r}"}}))
    raise SystemExit(0)

board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "reference": {reference!r},
    "pad": {pad_number!r},
    "net": {net_name!r},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def list_pcb_footprints(pcb_path: str) -> Dict[str, Any]:
        """List all footprints currently placed on the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

fp_list = []
for fp in board.GetFootprints():
    pos = fp.GetPosition()
    pads = []
    for pad in fp.Pads():
        pad_pos = pad.GetPosition()
        pads.append({{
            "number": pad.GetNumber(),
            "x_mm": round(pcbnew.ToMM(pad_pos.x), 3),
            "y_mm": round(pcbnew.ToMM(pad_pos.y), 3),
            "net": pad.GetNetname(),
        }})
    fp_list.append({{
        "reference": fp.GetReference(),
        "value": fp.GetValue(),
        "footprint": fp.GetFPID().GetUniStringLibItemName(),
        "x_mm": round(pcbnew.ToMM(pos.x), 3),
        "y_mm": round(pcbnew.ToMM(pos.y), 3),
        "rotation": fp.GetOrientationDegrees(),
        "layer": board.GetLayerName(fp.GetLayer()),
        "pads": pads,
    }})

print(json.dumps({{
    "status": "ok",
    "footprint_count": len(fp_list),
    "footprints": fp_list,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def list_pcb_nets(pcb_path: str) -> Dict[str, Any]:
        """List all nets in the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

nets = []
for code, net in board.GetNetsByNetcode().items():
    if code > 0:
        nets.append({{
            "code": code,
            "name": net.GetNetname(),
        }})

print(json.dumps({{
    "status": "ok",
    "net_count": len(nets),
    "nets": nets,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def add_via(
        pcb_path: str,
        x_mm: float,
        y_mm: float,
        drill_mm: float = 0.3,
        size_mm: float = 0.6,
        net_name: str = "",
        via_type: str = "through",
    ) -> Dict[str, Any]:
        """Add a via to the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            x_mm: X position in millimeters.
            y_mm: Y position in millimeters.
            drill_mm: Drill diameter in mm (default 0.3).
            size_mm: Via pad diameter in mm (default 0.6).
            net_name: Net name to assign (empty for unassigned).
            via_type: Via type - "through", "blind_buried", or "micro" (default "through").
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

via = pcbnew.PCB_VIA(board)
via.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pcbnew.FromMM({y_mm})))
via.SetDrill(pcbnew.FromMM({drill_mm}))
via.SetWidth(pcbnew.FromMM({size_mm}))

via_type = {via_type!r}
if via_type == "blind_buried":
    via.SetViaType(pcbnew.VIATYPE_BLIND_BURIED)
elif via_type == "micro":
    via.SetViaType(pcbnew.VIATYPE_MICROVIA)
else:
    via.SetViaType(pcbnew.VIATYPE_THROUGH)

net_name = {net_name!r}
if net_name:
    net = board.FindNet(net_name)
    if net:
        via.SetNet(net)

board.Add(via)
board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "via": {{
        "x_mm": {x_mm},
        "y_mm": {y_mm},
        "drill_mm": {drill_mm},
        "size_mm": {size_mm},
        "type": via_type,
        "net": net_name,
    }},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def add_copper_zone(
        pcb_path: str,
        net_name: str,
        layer: str = "F.Cu",
        corners: List[List[float]] = [],
        clearance_mm: float = 0.3,
        min_width_mm: float = 0.2,
        connect_pads: str = "thermal",
        priority: int = 0,
    ) -> Dict[str, Any]:
        """Add a copper zone (pour/fill) to the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            net_name: Net name for the zone (e.g., "GND").
            layer: Copper layer (default "F.Cu").
            corners: List of [x_mm, y_mm] corner points defining the zone outline.
            clearance_mm: Clearance to other nets in mm (default 0.3).
            min_width_mm: Minimum fill width in mm (default 0.2).
            connect_pads: Pad connection type - "thermal", "solid", or "none" (default "thermal").
            priority: Zone priority (higher fills first, default 0).
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        if len(corners) < 3:
            return {"error": "Zone needs at least 3 corner points"}

        corners_repr = repr(corners)

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

net = board.FindNet({net_name!r})
if net is None or net.GetNetCode() == 0:
    print(json.dumps({{"error": f"Net {net_name!r} not found"}}))
    raise SystemExit(0)

zone = pcbnew.ZONE(board)
zone.SetNet(net)
zone.SetLayer(board.GetLayerID({layer!r}))
zone.SetAssignedPriority({priority})

# Set clearance and minimum width
zone.SetLocalClearance(pcbnew.FromMM({clearance_mm}))
zone.SetMinThickness(pcbnew.FromMM({min_width_mm}))

# Set pad connection type
connect = {connect_pads!r}
if connect == "solid":
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
elif connect == "none":
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_NONE)
else:
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)

# Build outline
outline = zone.Outline()
outline.NewOutline()
corners = {corners_repr}
for i, (cx, cy) in enumerate(corners):
    outline.Append(pcbnew.FromMM(cx), pcbnew.FromMM(cy))

board.Add(zone)

# Fill the zone
filler = pcbnew.ZONE_FILLER(board)
zones = board.Zones()
zone_list = pcbnew.ZONES()
for z in zones:
    zone_list.append(z)
filler.Fill(zone_list)

board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "zone": {{
        "net": {net_name!r},
        "layer": {layer!r},
        "corners": corners,
        "clearance_mm": {clearance_mm},
        "min_width_mm": {min_width_mm},
        "connect_pads": connect,
        "priority": {priority},
    }},
}}))
"""
        return run_pcbnew_script(script, timeout=60.0)

    @mcp.tool()
    def set_design_rules(
        pcb_path: str,
        min_track_width_mm: float = 0.2,
        min_clearance_mm: float = 0.2,
        min_via_diameter_mm: float = 0.6,
        min_via_drill_mm: float = 0.3,
        min_hole_to_hole_mm: float = 0.25,
    ) -> Dict[str, Any]:
        """Set PCB design rules (DRC constraints).

        Args:
            pcb_path: Path to the .kicad_pcb file.
            min_track_width_mm: Minimum track width in mm (default 0.2).
            min_clearance_mm: Minimum clearance between copper in mm (default 0.2).
            min_via_diameter_mm: Minimum via pad diameter in mm (default 0.6).
            min_via_drill_mm: Minimum via drill diameter in mm (default 0.3).
            min_hole_to_hole_mm: Minimum hole-to-hole distance in mm (default 0.25).
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})
ds = board.GetDesignSettings()

ds.SetCopperLayerCount(2)
ds.m_TrackMinWidth = pcbnew.FromMM({min_track_width_mm})
ds.m_MinClearance = pcbnew.FromMM({min_clearance_mm})
ds.m_ViasMinSize = pcbnew.FromMM({min_via_diameter_mm})
ds.m_ViasMinDrill = pcbnew.FromMM({min_via_drill_mm})
ds.m_HoleToHoleMin = pcbnew.FromMM({min_hole_to_hole_mm})

board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "design_rules": {{
        "min_track_width_mm": {min_track_width_mm},
        "min_clearance_mm": {min_clearance_mm},
        "min_via_diameter_mm": {min_via_diameter_mm},
        "min_via_drill_mm": {min_via_drill_mm},
        "min_hole_to_hole_mm": {min_hole_to_hole_mm},
    }},
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def bulk_assign_pad_nets(
        pcb_path: str,
        assignments: List[Dict[str, str]] = [],
    ) -> Dict[str, Any]:
        """Assign nets to multiple pads in a single operation.

        Each assignment is a dict with "reference", "pad", and "net" keys.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            assignments: List of {"reference": "R1", "pad": "1", "net": "GND"} dicts.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        if not assignments:
            return {"error": "No assignments provided"}

        assignments_repr = repr(assignments)

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

assignments = {assignments_repr}
results = []
errors = []

for a in assignments:
    ref = a["reference"]
    pad_num = a["pad"]
    net_name = a["net"]

    fp = board.FindFootprintByReference(ref)
    if fp is None:
        errors.append(f"Footprint {{ref}} not found")
        continue

    net = board.FindNet(net_name)
    if net is None or net.GetNetCode() == 0:
        errors.append(f"Net {{net_name}} not found")
        continue

    pad_found = False
    for pad in fp.Pads():
        if pad.GetNumber() == pad_num:
            pad.SetNet(net)
            pad_found = True
            results.append({{"reference": ref, "pad": pad_num, "net": net_name}})
            break

    if not pad_found:
        errors.append(f"Pad {{pad_num}} not found on {{ref}}")

board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "assigned": len(results),
    "errors": errors,
    "results": results,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def move_footprint(
        pcb_path: str,
        reference: str,
        x_mm: float,
        y_mm: float,
        rotation_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Move a footprint to a new position on the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            reference: Component reference (e.g., "R1").
            x_mm: New X position in millimeters.
            y_mm: New Y position in millimeters.
            rotation_deg: New rotation in degrees (None to keep current).
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        rot_code = f"fp.SetOrientationDegrees({rotation_deg})" if rotation_deg is not None else ""

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

fp = board.FindFootprintByReference({reference!r})
if fp is None:
    print(json.dumps({{"error": f"Footprint {reference!r} not found"}}))
    raise SystemExit(0)

fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pcbnew.FromMM({y_mm})))
{rot_code}

board.Save(board.GetFileName())

pos = fp.GetPosition()
print(json.dumps({{
    "status": "ok",
    "reference": {reference!r},
    "x_mm": round(pcbnew.ToMM(pos.x), 3),
    "y_mm": round(pcbnew.ToMM(pos.y), 3),
    "rotation": fp.GetOrientationDegrees(),
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def get_pad_positions(
        pcb_path: str,
        reference: str,
    ) -> Dict[str, Any]:
        """Get all pad positions for a footprint.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            reference: Component reference (e.g., "R1").
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

fp = board.FindFootprintByReference({reference!r})
if fp is None:
    print(json.dumps({{"error": f"Footprint {reference!r} not found"}}))
    raise SystemExit(0)

pads = []
for pad in fp.Pads():
    pos = pad.GetPosition()
    pads.append({{
        "number": pad.GetNumber(),
        "x_mm": round(pcbnew.ToMM(pos.x), 3),
        "y_mm": round(pcbnew.ToMM(pos.y), 3),
        "net": pad.GetNetname(),
        "shape": str(pad.GetShape()),
    }})

print(json.dumps({{
    "status": "ok",
    "reference": {reference!r},
    "pad_count": len(pads),
    "pads": pads,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def add_text_to_pcb(
        pcb_path: str,
        text: str,
        x_mm: float,
        y_mm: float,
        layer: str = "F.SilkS",
        size_mm: float = 1.0,
        thickness_mm: float = 0.15,
        rotation_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """Add text to the PCB.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            text: Text content.
            x_mm: X position in millimeters.
            y_mm: Y position in millimeters.
            layer: PCB layer (default "F.SilkS").
            size_mm: Text height in mm (default 1.0).
            thickness_mm: Text line thickness in mm (default 0.15).
            rotation_deg: Text rotation in degrees (default 0).
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

txt = pcbnew.PCB_TEXT(board)
txt.SetText({text!r})
txt.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pcbnew.FromMM({y_mm})))
txt.SetLayer(board.GetLayerID({layer!r}))
txt.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM({size_mm}), pcbnew.FromMM({size_mm})))
txt.SetTextThickness(pcbnew.FromMM({thickness_mm}))
if {rotation_deg} != 0:
    txt.SetTextAngleDegrees({rotation_deg})

board.Add(txt)
board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "text": {text!r},
    "x_mm": {x_mm},
    "y_mm": {y_mm},
    "layer": {layer!r},
}}))
"""
        return run_pcbnew_script(script)

    # =========================================================================
    # Keepout-aware placement validation tools
    # =========================================================================

    # Keepout helper code is imported from kicad_mcp.utils.keepout_helpers.
    # It's a string constant that gets embedded in each pcbnew subprocess script.
    _KEEPOUT_HELPER = KEEPOUT_HELPER

    @mcp.tool()
    def get_keepout_zones(pcb_path: str) -> Dict[str, Any]:
        """List all keepout/rule areas on the PCB with their boundaries and constraints.

        Returns keepout zones from both board-level and footprint-embedded sources.
        Useful for understanding placement constraints before placing components.

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({pcb_path!r})
keepouts = extract_keepouts(board)
print(json.dumps({{"status": "ok", "keepout_count": len(keepouts), "keepouts": keepouts}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def get_board_constraints(pcb_path: str) -> Dict[str, Any]:
        """Get a complete summary of board outline, keepout zones, design rules, and placement area.

        Returns all information needed to make informed placement decisions:
        board dimensions, keepout zone locations and restrictions, design rules,
        and the effective available area for component placement.

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({pcb_path!r})
keepouts = extract_keepouts(board)
outline = get_board_outline(board)

ds = board.GetDesignSettings()
design_rules = {{
    "min_track_width_mm": round(pcbnew.ToMM(ds.m_TrackMinWidth), 3),
    "min_clearance_mm": round(pcbnew.ToMM(ds.m_MinClearance), 3),
    "min_via_diameter_mm": round(pcbnew.ToMM(ds.m_ViasMinSize), 3),
}}

board_area = 0
if outline:
    board_area = round(outline["width_mm"] * outline["height_mm"], 1)
    outline["area_mm2"] = board_area

total_keepout_area = 0
for kz in keepouts:
    bb = kz["bounding_box"]
    kz_area = round((bb["x_max_mm"] - bb["x_min_mm"]) * (bb["y_max_mm"] - bb["y_min_mm"]), 1)
    kz["area_mm2"] = kz_area
    if board_area > 0:
        kz["board_coverage_pct"] = round(100 * kz_area / board_area, 1)
    total_keepout_area += kz_area

result = {{
    "status": "ok",
    "board_outline": outline,
    "keepout_zones": keepouts,
    "design_rules": design_rules,
    "existing_footprints_count": len(list(board.GetFootprints())),
    "total_keepout_area_mm2": round(total_keepout_area, 1),
}}
if board_area > 0:
    result["effective_placement_area_mm2"] = round(board_area - total_keepout_area, 1)
print(json.dumps(result))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def validate_placement(
        pcb_path: str,
        library: str,
        footprint_name: str,
        x_mm: float,
        y_mm: float,
        rotation_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """Check if placing a footprint at the given position would violate keepout zones or board boundaries.

        Loads the footprint from the library to get its bounding box, then checks
        for overlap with all keepout zones and whether it fits within the board outline.
        Does NOT modify the PCB file.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            library: Footprint library name (e.g., "Resistor_SMD").
            footprint_name: Footprint name within the library.
            x_mm: Proposed X position in millimeters.
            y_mm: Proposed Y position in millimeters.
            rotation_deg: Proposed rotation in degrees (default 0).
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json, os
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({pcb_path!r})

lib_search_paths = [
    "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints",
    os.path.expanduser("~/Documents/KiCad/footprints"),
    "/usr/share/kicad/footprints",
]
lib_path = None
for sp in lib_search_paths:
    candidate = os.path.join(sp, {library!r} + ".pretty")
    if os.path.isdir(candidate):
        lib_path = candidate
        break
if not lib_path:
    print(json.dumps({{"error": "Library '{library}' not found"}}))
    raise SystemExit(0)

fp = pcbnew.FootprintLoad(lib_path, {footprint_name!r})
if fp is None:
    print(json.dumps({{"error": "Footprint '{footprint_name}' not found in '{library}'"}}))
    raise SystemExit(0)

fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pcbnew.FromMM({y_mm})))
if {rotation_deg} != 0:
    fp.SetOrientationDegrees({rotation_deg})

fp_bbox = fp.GetBoundingBox(False, False)  # exclude text for accurate body bbox
fp_rect = {{
    "x_min_mm": round(pcbnew.ToMM(fp_bbox.GetX()), 3),
    "y_min_mm": round(pcbnew.ToMM(fp_bbox.GetY()), 3),
    "x_max_mm": round(pcbnew.ToMM(fp_bbox.GetRight()), 3),
    "y_max_mm": round(pcbnew.ToMM(fp_bbox.GetBottom()), 3),
}}

keepouts = extract_keepouts(board)
outline = get_board_outline(board)
violations = []
warnings = []

for kz in keepouts:
    kz_bb = kz["bounding_box"]
    if not rects_overlap(fp_rect, kz_bb):
        continue
    area = overlap_area(fp_rect, kz_bb)
    c = kz["constraints"]
    if c["no_footprints"]:
        violations.append({{
            "type": "keepout_overlap",
            "keepout_source": kz["source"],
            "keepout_ref": kz["source_ref"],
            "overlap_mm2": area,
            "blocked": [k.replace("no_", "") for k, v in c.items() if v],
            "message": "Footprint overlaps keepout zone"
                       + (f" from {{kz['source_ref']}}" if kz["source_ref"] else ""),
        }})
    else:
        blocked = [k.replace("no_", "") for k, v in c.items() if v]
        if blocked:
            warnings.append({{
                "type": "routing_keepout_overlap",
                "keepout_source": kz["source"],
                "keepout_ref": kz["source_ref"],
                "overlap_mm2": area,
                "blocked": blocked,
                "message": f"Footprint overlaps zone that blocks {{', '.join(blocked)}} (routing may be difficult)",
            }})

if outline and not rect_inside(fp_rect, outline):
    overhang = {{}}
    if fp_rect["x_min_mm"] < outline["x_min_mm"]:
        overhang["left_mm"] = round(outline["x_min_mm"] - fp_rect["x_min_mm"], 3)
    if fp_rect["x_max_mm"] > outline["x_max_mm"]:
        overhang["right_mm"] = round(fp_rect["x_max_mm"] - outline["x_max_mm"], 3)
    if fp_rect["y_min_mm"] < outline["y_min_mm"]:
        overhang["top_mm"] = round(outline["y_min_mm"] - fp_rect["y_min_mm"], 3)
    if fp_rect["y_max_mm"] > outline["y_max_mm"]:
        overhang["bottom_mm"] = round(fp_rect["y_max_mm"] - outline["y_max_mm"], 3)
    violations.append({{
        "type": "outside_board",
        "overhang": overhang,
        "message": "Footprint extends beyond board outline",
    }})

print(json.dumps({{
    "status": "ok",
    "valid": len(violations) == 0,
    "violations": violations,
    "warnings": warnings,
    "footprint_bbox_mm": fp_rect,
    "board_outline_mm": outline,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def audit_pcb_placement(pcb_path: str) -> Dict[str, Any]:
        """Audit all footprint placements for keepout zone violations and board boundary issues.

        Checks every placed footprint against all keepout/rule areas and the board outline.
        Reports violations with overlap details. Does NOT modify the PCB file.

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json
{_KEEPOUT_HELPER}
board = pcbnew.LoadBoard({pcb_path!r})
keepouts = extract_keepouts(board)
outline = get_board_outline(board)

violations_list = []
clean_count = 0

for fp in board.GetFootprints():
    ref = fp.GetReference()
    fp_bbox = fp.GetBoundingBox(False, False)  # exclude text for accurate body bbox
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
        if not rects_overlap(fp_rect, kz_bb):
            continue
        area = overlap_area(fp_rect, kz_bb)
        c = kz["constraints"]
        blocked = [k.replace("no_", "") for k, v in c.items() if v]
        severity = "violation" if c["no_footprints"] else "warning"
        issues.append({{
            "type": "keepout_overlap",
            "severity": severity,
            "keepout_source": kz["source"],
            "keepout_ref": kz["source_ref"],
            "overlap_mm2": area,
            "blocked": blocked,
        }})

    if outline and not rect_inside(fp_rect, outline):
        overhang = {{}}
        if fp_rect["x_min_mm"] < outline["x_min_mm"]:
            overhang["left_mm"] = round(outline["x_min_mm"] - fp_rect["x_min_mm"], 3)
        if fp_rect["x_max_mm"] > outline["x_max_mm"]:
            overhang["right_mm"] = round(fp_rect["x_max_mm"] - outline["x_max_mm"], 3)
        if fp_rect["y_min_mm"] < outline["y_min_mm"]:
            overhang["top_mm"] = round(outline["y_min_mm"] - fp_rect["y_min_mm"], 3)
        if fp_rect["y_max_mm"] > outline["y_max_mm"]:
            overhang["bottom_mm"] = round(fp_rect["y_max_mm"] - outline["y_max_mm"], 3)
        issues.append({{
            "type": "outside_board",
            "severity": "violation",
            "overhang": overhang,
        }})

    if issues:
        pos = fp.GetPosition()
        violations_list.append({{
            "reference": ref,
            "value": fp.GetValue(),
            "footprint": fp.GetFPID().GetUniStringLibItemName(),
            "position_mm": [round(pcbnew.ToMM(pos.x), 3), round(pcbnew.ToMM(pos.y), 3)],
            "bbox_mm": fp_rect,
            "issues": issues,
        }})
    else:
        clean_count += 1

total = len(list(board.GetFootprints()))
vcount = len(violations_list)
summary = f"{{vcount}} of {{total}} footprints have placement issues" if vcount > 0 else f"All {{total}} footprints pass placement checks"

print(json.dumps({{
    "status": "ok",
    "total_footprints": total,
    "violations_count": vcount,
    "clean_count": clean_count,
    "violations": violations_list,
    "summary": summary,
}}))
"""
        return run_pcbnew_script(script)

    # =========================================================================
    # Zone fill tools
    # =========================================================================

    @mcp.tool()
    def fill_zones(pcb_path: str) -> Dict[str, Any]:
        """Fill all copper zones on the PCB.

        Runs the zone filler headlessly to compute copper fills for all
        non-rule-area zones. This replaces the manual "press B in KiCad" step.
        Zones must already exist on the board (use add_copper_zone first).

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

# Count copper zones (skip rule areas)
copper_zones = []
for z in board.Zones():
    if not z.GetIsRuleArea():
        copper_zones.append(z)

if not copper_zones:
    print(json.dumps({{"status": "ok", "message": "No copper zones to fill", "zones_filled": 0}}))
    raise SystemExit(0)

# Unfill first to force recomputation
for z in copper_zones:
    z.UnFill()

# Fill all zones (never pass aCheck=True headlessly — it hangs)
filler = pcbnew.ZONE_FILLER(board)
zones = board.Zones()
success = filler.Fill(zones)

# Collect results
zone_info = []
for z in copper_zones:
    layer_set = z.GetLayerSet()
    layer_name = "F.Cu" if layer_set.Contains(pcbnew.F_Cu) else "B.Cu" if layer_set.Contains(pcbnew.B_Cu) else "unknown"
    zone_info.append({{
        "net": z.GetNetname(),
        "layer": layer_name,
        "filled": z.IsFilled(),
        "filled_area_mm2": round(pcbnew.ToMM(pcbnew.ToMM(z.GetFilledArea())), 1),
    }})

board.Save(board.GetFileName())

print(json.dumps({{
    "status": "ok",
    "fill_success": success,
    "zones_filled": len(copper_zones),
    "zones": zone_info,
}}))
"""
        return run_pcbnew_script(script, timeout=60.0)

    # =========================================================================
    # Silkscreen tools
    # =========================================================================

    @mcp.tool()
    def list_silkscreen_items(pcb_path: str) -> Dict[str, Any]:
        """List all silkscreen text items on the PCB.

        Returns reference designators, values, and standalone text with their
        positions, sizes, visibility, and layers.

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

items = []

# Footprint text (references and values)
for fp in board.GetFootprints():
    ref = fp.GetReference()
    for field_type, field_obj in [("reference", fp.Reference()), ("value", fp.Value())]:
        pos = field_obj.GetPosition()
        size = field_obj.GetTextSize()
        rel_pos = field_obj.GetFPRelativePosition()
        items.append({{
            "type": field_type,
            "component": ref,
            "text": field_obj.GetText(),
            "visible": field_obj.IsVisible(),
            "layer": board.GetLayerName(field_obj.GetLayer()),
            "x_mm": round(pcbnew.ToMM(pos.x), 3),
            "y_mm": round(pcbnew.ToMM(pos.y), 3),
            "rel_x_mm": round(pcbnew.ToMM(rel_pos.x), 3),
            "rel_y_mm": round(pcbnew.ToMM(rel_pos.y), 3),
            "size_mm": round(pcbnew.ToMM(size.x), 3),
            "thickness_mm": round(pcbnew.ToMM(field_obj.GetTextThickness()), 3),
            "angle_deg": field_obj.GetTextAngle().AsDegrees(),
        }})

# Standalone text items on silkscreen layers
silk_layers = [board.GetLayerID("F.SilkS"), board.GetLayerID("B.SilkS")]
for drawing in board.GetDrawings():
    if hasattr(drawing, 'GetText') and drawing.GetLayer() in silk_layers:
        pos = drawing.GetPosition()
        size = drawing.GetTextSize()
        items.append({{
            "type": "standalone",
            "component": None,
            "text": drawing.GetText(),
            "visible": drawing.IsVisible() if hasattr(drawing, 'IsVisible') else True,
            "layer": board.GetLayerName(drawing.GetLayer()),
            "x_mm": round(pcbnew.ToMM(pos.x), 3),
            "y_mm": round(pcbnew.ToMM(pos.y), 3),
            "rel_x_mm": 0,
            "rel_y_mm": 0,
            "size_mm": round(pcbnew.ToMM(size.x), 3),
            "thickness_mm": round(pcbnew.ToMM(drawing.GetTextThickness()), 3),
            "angle_deg": drawing.GetTextAngle().AsDegrees(),
        }})

print(json.dumps({{
    "status": "ok",
    "item_count": len(items),
    "items": items,
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def update_silkscreen_item(
        pcb_path: str,
        reference: str,
        field: str = "reference",
        visible: Optional[bool] = None,
        x_mm: Optional[float] = None,
        y_mm: Optional[float] = None,
        rel_x_mm: Optional[float] = None,
        rel_y_mm: Optional[float] = None,
        size_mm: Optional[float] = None,
        thickness_mm: Optional[float] = None,
        angle_deg: Optional[float] = None,
        layer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a silkscreen text item's properties.

        Modify visibility, position, size, rotation, or layer of a footprint's
        reference designator or value text.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            reference: Component reference (e.g., "R1").
            field: Which field to update - "reference" or "value" (default "reference").
            visible: Set visibility (True/False). None to keep current.
            x_mm: New absolute X position in mm. None to keep current.
            y_mm: New absolute Y position in mm. None to keep current.
            rel_x_mm: New X position relative to footprint center in mm. None to keep current.
            rel_y_mm: New Y position relative to footprint center in mm. None to keep current.
            size_mm: New text height in mm. None to keep current.
            thickness_mm: New text stroke thickness in mm. None to keep current.
            angle_deg: New text rotation in degrees. None to keep current.
            layer: New layer name (e.g., "F.SilkS", "B.SilkS"). None to keep current.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        if field not in ("reference", "value"):
            return {"error": f"field must be 'reference' or 'value', got {field!r}"}

        # Build the modification statements
        mods = []
        if visible is not None:
            mods.append(f"text.SetVisible({visible!r})")
        if x_mm is not None and y_mm is not None:
            mods.append(f"text.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pcbnew.FromMM({y_mm})))")
        elif x_mm is not None or y_mm is not None:
            if x_mm is not None:
                mods.append(f"pos = text.GetPosition(); text.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM({x_mm}), pos.y))")
            else:
                mods.append(f"pos = text.GetPosition(); text.SetPosition(pcbnew.VECTOR2I(pos.x, pcbnew.FromMM({y_mm})))")
        if rel_x_mm is not None and rel_y_mm is not None:
            mods.append(f"text.SetFPRelativePosition(pcbnew.VECTOR2I(pcbnew.FromMM({rel_x_mm}), pcbnew.FromMM({rel_y_mm})))")
        if size_mm is not None:
            mods.append(f"text.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM({size_mm}), pcbnew.FromMM({size_mm})))")
        if thickness_mm is not None:
            mods.append(f"text.SetTextThickness(pcbnew.FromMM({thickness_mm}))")
        if angle_deg is not None:
            mods.append(f"text.SetTextAngle(pcbnew.EDA_ANGLE({angle_deg}, pcbnew.DEGREES_T))")
        if layer is not None:
            mods.append(f"text.SetLayer(board.GetLayerID({layer!r}))")

        if not mods:
            return {"error": "No modifications specified"}

        mods_code = "\n".join(mods)

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

fp = board.FindFootprintByReference({reference!r})
if fp is None:
    print(json.dumps({{"error": f"Footprint {reference!r} not found"}}))
    raise SystemExit(0)

text = fp.Reference() if {field!r} == "reference" else fp.Value()

{mods_code}

board.Save(board.GetFileName())

pos = text.GetPosition()
size = text.GetTextSize()
rel_pos = text.GetFPRelativePosition()
print(json.dumps({{
    "status": "ok",
    "reference": {reference!r},
    "field": {field!r},
    "text": text.GetText(),
    "visible": text.IsVisible(),
    "layer": board.GetLayerName(text.GetLayer()),
    "x_mm": round(pcbnew.ToMM(pos.x), 3),
    "y_mm": round(pcbnew.ToMM(pos.y), 3),
    "rel_x_mm": round(pcbnew.ToMM(rel_pos.x), 3),
    "rel_y_mm": round(pcbnew.ToMM(rel_pos.y), 3),
    "size_mm": round(pcbnew.ToMM(size.x), 3),
    "thickness_mm": round(pcbnew.ToMM(text.GetTextThickness()), 3),
    "angle_deg": text.GetTextAngle().AsDegrees(),
}}))
"""
        return run_pcbnew_script(script)

    @mcp.tool()
    def check_silkscreen_overlaps(pcb_path: str) -> Dict[str, Any]:
        """Find silkscreen text items that overlap copper pads.

        Checks all visible silkscreen text (reference designators, values,
        standalone text) against all pads on the board. Reports overlaps
        between different components that could cause manufacturing issues.
        Skips text overlapping its own component's pads (which is normal).

        Args:
            pcb_path: Path to the .kicad_pcb file.
        """
        if not os.path.exists(pcb_path):
            return {"error": f"PCB file not found: {pcb_path}"}

        script = f"""
import pcbnew, json

board = pcbnew.LoadBoard({pcb_path!r})

silk_layer_ids = [board.GetLayerID("F.SilkS"), board.GetLayerID("B.SilkS")]

# Collect all visible silkscreen text items with bounding boxes
silk_items = []
for fp in board.GetFootprints():
    ref = fp.GetReference()
    for field_type, field_obj in [("reference", fp.Reference()), ("value", fp.Value())]:
        if not field_obj.IsVisible():
            continue
        if field_obj.GetLayer() not in silk_layer_ids:
            continue
        bbox = field_obj.GetBoundingBox()
        silk_items.append({{
            "type": field_type,
            "component": ref,
            "text": field_obj.GetText(),
            "layer": board.GetLayerName(field_obj.GetLayer()),
            "bbox_x_min": bbox.GetX(),
            "bbox_y_min": bbox.GetY(),
            "bbox_x_max": bbox.GetRight(),
            "bbox_y_max": bbox.GetBottom(),
        }})

# Also check standalone text
for drawing in board.GetDrawings():
    if hasattr(drawing, 'GetText') and drawing.GetLayer() in silk_layer_ids:
        if hasattr(drawing, 'IsVisible') and not drawing.IsVisible():
            continue
        bbox = drawing.GetBoundingBox()
        silk_items.append({{
            "type": "standalone",
            "component": None,
            "text": drawing.GetText(),
            "layer": board.GetLayerName(drawing.GetLayer()),
            "bbox_x_min": bbox.GetX(),
            "bbox_y_min": bbox.GetY(),
            "bbox_x_max": bbox.GetRight(),
            "bbox_y_max": bbox.GetBottom(),
        }})

# Collect all pads
pads = []
for fp in board.GetFootprints():
    for pad in fp.Pads():
        sz = pad.GetBoundingBox()
        pads.append({{
            "reference": fp.GetReference(),
            "pad_number": pad.GetNumber(),
            "x_min": sz.GetX(),
            "y_min": sz.GetY(),
            "x_max": sz.GetRight(),
            "y_max": sz.GetBottom(),
        }})

# Check for AABB overlaps (skip text overlapping own component)
overlaps = []
for si in silk_items:
    for pad in pads:
        if si["component"] == pad["reference"]:
            continue
        if (si["bbox_x_min"] < pad["x_max"] and si["bbox_x_max"] > pad["x_min"] and
            si["bbox_y_min"] < pad["y_max"] and si["bbox_y_max"] > pad["y_min"]):
            overlaps.append({{
                "silk_type": si["type"],
                "silk_component": si["component"],
                "silk_text": si["text"],
                "silk_layer": si["layer"],
                "pad_component": pad["reference"],
                "pad_number": pad["pad_number"],
            }})

print(json.dumps({{
    "status": "ok",
    "silk_items_checked": len(silk_items),
    "pads_checked": len(pads),
    "overlap_count": len(overlaps),
    "overlaps": overlaps,
}}))
"""
        return run_pcbnew_script(script)
