"""
Fabrication output tools: Gerbers, drill files, and a fab zip.

Uses ``kicad-cli`` to plot manufacturing files from a project's board.
"""
import os
import glob
import subprocess
import zipfile
from typing import Dict, Any, Optional

from fastmcp import FastMCP, Context

from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils.kicad_cli import get_cli_manager


def _resolve_pcb(project_path: str) -> str:
    if project_path.endswith(".kicad_pcb"):
        return project_path
    files = get_project_files(project_path)
    if "pcb" not in files:
        raise FileNotFoundError("Project has no board (.kicad_pcb)")
    return files["pcb"]


def register_fabrication_tools(mcp: FastMCP) -> None:
    """Register fabrication output tools with the MCP server."""

    @mcp.tool()
    async def export_position_file(project_path: str, output_path: str = "",
                                   side: str = "both", fmt: str = "csv",
                                   units: str = "mm",
                                   ctx: Context = None) -> Dict[str, Any]:
        """Export a component placement (pick-and-place / centroid) file.

        Args:
            project_path: Path to the .kicad_pro (or .kicad_pcb) file.
            output_path: Output file; defaults to ``<board>-pos.<ext>`` next to
                the board.
            side: ``front``, ``back`` or ``both`` (default both).
            fmt: ``csv`` or ``ascii`` (default csv).
            units: ``mm`` or ``in`` (default mm).
        """
        try:
            cli = get_cli_manager().find_kicad_cli()
            if not cli:
                return {"success": False, "error": "kicad-cli not found"}
            pcb = _resolve_pcb(project_path)
            base = os.path.splitext(pcb)[0]
            ext = "csv" if fmt == "csv" else "pos"
            out = output_path or f"{base}-pos.{ext}"
            if ctx:
                ctx.info(f"Exporting placement file ({side}, {fmt}) -> {out}")
            p = subprocess.run(
                [cli, "pcb", "export", "pos", "--output", out,
                 "--side", side, "--format", fmt, "--units", units, pcb],
                capture_output=True, text=True, timeout=120,
            )
            if not os.path.exists(out):
                return {"success": False,
                        "error": (p.stderr or p.stdout).strip()[-400:]}
            return {"success": True, "board": pcb, "position_file": out,
                    "side": side, "format": fmt, "units": units}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def export_step(project_path: str, output_path: str = "",
                          ctx: Context = None) -> Dict[str, Any]:
        """Export a 3D STEP model of the board (with component 3D models).

        Args:
            project_path: Path to the .kicad_pro (or .kicad_pcb) file.
            output_path: Output ``.step`` file; defaults to ``<board>.step``.
        """
        try:
            cli = get_cli_manager().find_kicad_cli()
            if not cli:
                return {"success": False, "error": "kicad-cli not found"}
            pcb = _resolve_pcb(project_path)
            out = output_path or f"{os.path.splitext(pcb)[0]}.step"
            if ctx:
                ctx.info(f"Exporting 3D STEP -> {out}")
            p = subprocess.run(
                [cli, "pcb", "export", "step", "--force",
                 "--subst-models", "--no-dnp", "--output", out, pcb],
                capture_output=True, text=True, timeout=300,
            )
            if not os.path.exists(out):
                return {"success": False,
                        "error": (p.stderr or p.stdout).strip()[-400:]}
            return {"success": True, "board": pcb, "step_file": out,
                    "size_bytes": os.path.getsize(out)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def export_gerbers(project_path: str, output_dir: str = "",
                             make_zip: bool = True,
                             ctx: Context = None) -> Dict[str, Any]:
        """Export Gerber + drill files for a project's board, optionally zipped.

        Produces a standard fabrication set (copper, mask, silk, edge, plus NC
        drill files) ready to send to a PCB manufacturer.

        Args:
            project_path: Path to the .kicad_pro (or .kicad_pcb) file.
            output_dir: Where to write the files. Defaults to a ``gerbers``
                folder next to the board.
            make_zip: Also create ``<board>-gerbers.zip`` (default True).
        """
        try:
            cli = get_cli_manager().find_kicad_cli()
            if not cli:
                return {"success": False, "error": "kicad-cli not found"}

            if project_path.endswith(".kicad_pcb"):
                pcb = project_path
            else:
                files = get_project_files(project_path)
                if "pcb" not in files:
                    return {"success": False, "error": "Project has no board (.kicad_pcb)"}
                pcb = files["pcb"]

            board_dir = os.path.dirname(pcb)
            base = os.path.splitext(os.path.basename(pcb))[0]
            out = output_dir or os.path.join(board_dir, "gerbers")
            os.makedirs(out, exist_ok=True)

            if ctx:
                ctx.info(f"Plotting Gerbers to {out}")
            g = subprocess.run(
                [cli, "pcb", "export", "gerbers", "--output", out, pcb],
                capture_output=True, text=True, timeout=180,
            )
            if ctx:
                ctx.info("Generating drill files")
            d = subprocess.run(
                [cli, "pcb", "export", "drill", "--output", out + os.sep, pcb],
                capture_output=True, text=True, timeout=180,
            )

            produced = sorted(
                os.path.basename(p) for p in glob.glob(os.path.join(out, "*"))
                if os.path.isfile(p) and not p.endswith(".zip")
            )
            if not produced:
                return {"success": False, "stage": "export",
                        "error": (g.stderr or d.stderr or "no files produced").strip()[-400:]}

            result = {
                "success": True,
                "board": pcb,
                "output_dir": out,
                "files": produced,
                "file_count": len(produced),
            }

            if make_zip:
                zip_path = os.path.join(board_dir, f"{base}-gerbers.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name in produced:
                        zf.write(os.path.join(out, name), name)
                result["zip"] = zip_path

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
