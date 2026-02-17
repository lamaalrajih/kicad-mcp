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
