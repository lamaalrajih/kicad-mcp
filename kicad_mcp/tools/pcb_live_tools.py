"""
Live PCB control tools via KiCad's IPC API (kipy).

Unlike the file-based tools in this package, these talk to a *running* KiCad
PCB Editor over its IPC socket and modify the open board in real time. This
requires:

1. KiCad's API server enabled (Preferences -> Plugins -> "Enable the KiCad
   API"; stored as ``api.enable_server: true`` in ``kicad_common.json``), and
2. the PCB Editor open on the board you want to edit.

Each running editor exposes its own socket at ``/tmp/kicad/api-<pid>.sock``
(macOS/Linux). The project-manager socket (``api.sock``) does not serve board
documents, so these helpers probe the per-editor sockets and use the first one
that returns a board.
"""
import glob
import os
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP, Context

try:  # kipy is an optional runtime dependency
    from kipy import KiCad
    import kipy.board_types as bt
    from kipy.util.units import from_mm, to_mm
    from kipy.geometry import Vector2
    _KIPY_AVAILABLE = True
    _KIPY_IMPORT_ERROR = ""
except Exception as e:  # pragma: no cover - only when kipy missing
    _KIPY_AVAILABLE = False
    _KIPY_IMPORT_ERROR = str(e)


_NOT_AVAILABLE = {
    "success": False,
    "error": (
        "kipy (kicad-python) is not installed. Install it with "
        "'uv pip install kicad-python' to enable live PCB control."
    ),
}


def _candidate_sockets() -> List[str]:
    """Return candidate IPC socket paths, per-editor sockets first."""
    env = os.environ.get("KICAD_API_SOCKET")
    if env:
        return [env]
    editor = sorted(glob.glob("/tmp/kicad/api-*.sock"))
    generic = glob.glob("/tmp/kicad/api.sock")
    # newest editor socket first
    return [f"ipc://{p}" for p in reversed(editor)] + [f"ipc://{p}" for p in generic]


class _LiveError(Exception):
    pass


def _connect_board():
    """Connect to a running PCB Editor and return ``(kicad, board)``.

    Raises :class:`_LiveError` with an actionable message when no editor with
    an open board can be reached.
    """
    sockets = _candidate_sockets()
    if not sockets:
        raise _LiveError(
            "No KiCad API socket found. Is KiCad running with the API server "
            "enabled (Preferences -> Plugins -> Enable the KiCad API)?"
        )
    last_err = None
    for sp in sockets:
        try:
            k = KiCad(socket_path=sp)
            board = k.get_board()  # raises if this process has no open board
            return k, board
        except Exception as e:  # try the next socket
            last_err = e
            continue
    raise _LiveError(
        "Connected to KiCad but no open board was found. Open the PCB Editor "
        f"on your board and try again. (last error: {last_err})"
    )


def _V(x_mm: float, y_mm: float) -> "Vector2":
    v = Vector2()
    v.x = from_mm(float(x_mm))
    v.y = from_mm(float(y_mm))
    return v


def _layer_by_name(name: str):
    """Map a friendly layer name to a BoardLayer enum (default F.Cu)."""
    mapping = {
        "F.Cu": bt.BoardLayer.BL_F_Cu,
        "B.Cu": bt.BoardLayer.BL_B_Cu,
        "F.SilkS": bt.BoardLayer.BL_F_SilkS,
        "B.SilkS": bt.BoardLayer.BL_B_SilkS,
        "Edge.Cuts": bt.BoardLayer.BL_Edge_Cuts,
        "F.Fab": bt.BoardLayer.BL_F_Fab,
        "B.Fab": bt.BoardLayer.BL_B_Fab,
    }
    return mapping.get(name, bt.BoardLayer.BL_F_Cu)


def register_pcb_live_tools(mcp: FastMCP) -> None:
    """Register live (IPC) PCB control tools with the MCP server."""

    @mcp.tool()
    async def pcb_live_status(ctx: Context = None) -> Dict[str, Any]:
        """Check whether a live KiCad PCB Editor session is reachable.

        Returns connection info and a summary of the open board (item counts,
        board size). Use this first to confirm the IPC link before editing.
        """
        if not _KIPY_AVAILABLE:
            return {**_NOT_AVAILABLE, "import_error": _KIPY_IMPORT_ERROR}
        try:
            k, board = _connect_board()
            tracks = board.get_tracks()
            fps = board.get_footprints()
            shapes = board.get_shapes()
            texts = board.get_text()
            return {
                "success": True,
                "connected": True,
                "kicad_version": str(k.get_version()),
                "footprints": len(fps),
                "tracks": len(tracks),
                "shapes": len(shapes),
                "texts": len(texts),
                "copper_layers": board.get_copper_layer_count(),
            }
        except _LiveError as e:
            return {"success": False, "connected": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "connected": False, "error": str(e)}

    @mcp.tool()
    async def pcb_add_board_outline(x: float, y: float,
                                    width: float, height: float,
                                    ctx: Context = None) -> Dict[str, Any]:
        """Draw a rectangular board outline on Edge.Cuts of the open board.

        Args:
            x, y: Top-left corner in mm.
            width, height: Rectangle size in mm.
        """
        if not _KIPY_AVAILABLE:
            return _NOT_AVAILABLE
        try:
            _, board = _connect_board()
            rect = bt.BoardRectangle()
            rect.top_left = _V(x, y)
            rect.bottom_right = _V(x + width, y + height)
            rect.layer = bt.BoardLayer.BL_Edge_Cuts
            created = board.create_items([rect])
            return {"success": True, "created": len(created),
                    "outline_mm": {"x": x, "y": y, "w": width, "h": height}}
        except _LiveError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def pcb_add_track(start_x: float, start_y: float,
                            end_x: float, end_y: float,
                            width: float = 0.25, layer: str = "F.Cu",
                            ctx: Context = None) -> Dict[str, Any]:
        """Add a copper track segment to the open board.

        Args:
            start_x, start_y: Start point in mm.
            end_x, end_y: End point in mm.
            width: Track width in mm (default 0.25).
            layer: Copper layer, ``"F.Cu"`` or ``"B.Cu"`` (default F.Cu).
        """
        if not _KIPY_AVAILABLE:
            return _NOT_AVAILABLE
        try:
            _, board = _connect_board()
            tr = bt.Track()
            tr.start = _V(start_x, start_y)
            tr.end = _V(end_x, end_y)
            tr.width = from_mm(float(width))
            tr.layer = _layer_by_name(layer)
            created = board.create_items([tr])
            return {"success": True, "created": len(created),
                    "start": [start_x, start_y], "end": [end_x, end_y],
                    "width_mm": width, "layer": layer}
        except _LiveError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def pcb_add_text(text: str, x: float, y: float,
                           layer: str = "F.SilkS",
                           ctx: Context = None) -> Dict[str, Any]:
        """Add a text label to the open board.

        Args:
            text: The text to place.
            x, y: Position in mm.
            layer: Layer name (default ``"F.SilkS"``).
        """
        if not _KIPY_AVAILABLE:
            return _NOT_AVAILABLE
        try:
            _, board = _connect_board()
            t = bt.BoardText()
            t.value = text
            t.position = _V(x, y)
            t.layer = _layer_by_name(layer)
            created = board.create_items([t])
            return {"success": True, "created": len(created),
                    "text": text, "position": [x, y], "layer": layer}
        except _LiveError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def pcb_save(ctx: Context = None) -> Dict[str, Any]:
        """Save the open board to disk (via the KiCad API)."""
        if not _KIPY_AVAILABLE:
            return _NOT_AVAILABLE
        try:
            k, _ = _connect_board()
            result = k.run_action("common.Control.save")
            return {"success": True, "action_status": str(result)}
        except _LiveError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}
