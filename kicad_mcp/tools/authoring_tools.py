"""
Schematic authoring tools for KiCad projects.

Unlike the other tools in this package (which read and analyze existing
designs), these tools *create and modify* schematics: a new project, symbol
placement, wires and net labels. They are built on
:mod:`kicad_mcp.utils.sexpr_author` and require only the installed KiCad
symbol libraries -- no running KiCad instance.
"""
import os
from typing import Dict, Any

from fastmcp import FastMCP, Context

from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils import sexpr_author as author


def _resolve_schematic(project_path: str) -> str:
    """Return the .kicad_sch path for a project (.kicad_pro) or a direct .kicad_sch."""
    if project_path.endswith(".kicad_sch"):
        return project_path
    files = get_project_files(project_path)
    if "schematic" not in files:
        raise FileNotFoundError("No schematic (.kicad_sch) found for project")
    return files["schematic"]


def register_authoring_tools(mcp: FastMCP) -> None:
    """Register schematic authoring tools with the MCP server.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.tool()
    async def create_project(project_dir: str, name: str,
                             with_pcb: bool = True, ctx: Context = None) -> Dict[str, Any]:
        """Create a new, empty KiCad project.

        Creates ``<name>.kicad_pro``, ``<name>.kicad_sch`` and (optionally)
        ``<name>.kicad_pcb`` inside ``project_dir``.

        Args:
            project_dir: Directory to create the project in (created if missing)
            name: Base name for the project files (no extension)
            with_pcb: Also create an empty board file (default True)

        Returns:
            Dictionary describing the created files.
        """
        try:
            if ctx:
                ctx.info(f"Creating project '{name}' in {project_dir}")
            result = author.create_project(project_dir, name, with_pcb=with_pcb)
            result["success"] = True
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def add_schematic_component(project_path: str, lib_id: str,
                                      reference: str, value: str,
                                      x: float, y: float,
                                      rotation: float = 0.0,
                                      footprint: str = "",
                                      ctx: Context = None) -> Dict[str, Any]:
        """Place a component (symbol) on a project's schematic.

        The symbol definition is loaded from KiCad's standard libraries by
        ``lib_id`` (e.g. ``"Device:R"``, ``"Device:LED"``, ``"Device:C"``)
        and embedded into the schematic automatically.

        Args:
            project_path: Path to the .kicad_pro (or .kicad_sch) file
            lib_id: Symbol identifier, ``"Library:Symbol"`` (e.g. ``"Device:R"``)
            reference: Reference designator (e.g. ``"R1"``)
            value: Component value (e.g. ``"10k"``)
            x: X position on the sheet, in mm
            y: Y position on the sheet, in mm
            rotation: Rotation in degrees (0/90/180/270; default 0)
            footprint: Optional footprint (e.g. ``"Resistor_SMD:R_0603_1608Metric"``)

        Returns:
            Details of the placed component, including the absolute schematic
            coordinates of every pin (use these for ``add_schematic_wire`` /
            ``add_schematic_label``).
        """
        try:
            sch = _resolve_schematic(project_path)
            if ctx:
                ctx.info(f"Adding {reference} ({lib_id}) at ({x}, {y})")
            root = author.load_schematic(sch)
            comp = author.add_component(root, lib_id, reference, value,
                                        float(x), float(y),
                                        rotation=float(rotation),
                                        footprint=footprint)
            author.save_schematic(sch, root)
            comp["success"] = True
            comp["schematic_path"] = sch
            return comp
        except author.SymbolNotFound as e:
            return {"success": False, "error": f"Symbol not found: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def add_schematic_wire(project_path: str,
                                 start_x: float, start_y: float,
                                 end_x: float, end_y: float,
                                 ctx: Context = None) -> Dict[str, Any]:
        """Draw a wire between two points on a project's schematic.

        To connect two component pins, pass the pin coordinates returned by
        ``add_schematic_component``.

        Args:
            project_path: Path to the .kicad_pro (or .kicad_sch) file
            start_x, start_y: Wire start point, in mm
            end_x, end_y: Wire end point, in mm
        """
        try:
            sch = _resolve_schematic(project_path)
            root = author.load_schematic(sch)
            uid = author.add_wire(root, float(start_x), float(start_y),
                                  float(end_x), float(end_y))
            author.save_schematic(sch, root)
            return {"success": True, "uuid": uid, "schematic_path": sch,
                    "start": [start_x, start_y], "end": [end_x, end_y]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def add_schematic_label(project_path: str, text: str,
                                  x: float, y: float, rotation: float = 0.0,
                                  ctx: Context = None) -> Dict[str, Any]:
        """Add a net label at a point on a project's schematic.

        Two items sharing a label name are electrically connected. Place the
        label on a pin or wire endpoint (use pin coordinates from
        ``add_schematic_component``).

        Args:
            project_path: Path to the .kicad_pro (or .kicad_sch) file
            text: Net/label name (e.g. ``"VIN"``, ``"GND"``)
            x: X position, in mm
            y: Y position, in mm
            rotation: Label rotation in degrees (default 0)
        """
        try:
            sch = _resolve_schematic(project_path)
            root = author.load_schematic(sch)
            uid = author.add_label(root, text, float(x), float(y),
                                   rotation=float(rotation))
            author.save_schematic(sch, root)
            return {"success": True, "uuid": uid, "schematic_path": sch,
                    "label": text, "position": [x, y]}
        except Exception as e:
            return {"success": False, "error": str(e)}
