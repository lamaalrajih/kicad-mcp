"""
PCB generation tools: update a board from its schematic, and autoroute.

These bridge the schematic and PCB worlds using KiCad's bundled ``pcbnew``
(via :mod:`kicad_mcp.utils.pcbnew_runner`) for operations that ``kicad-cli``
does not provide -- notably "update PCB from schematic" (create footprints and
assign nets) and Specctra DSN/SES import/export for external autorouting.

Autorouting itself is delegated to Freerouting (an external Java tool). It is
optional and located via the ``FREEROUTING_JAR`` environment variable or common
install paths; the tool never downloads it.
"""
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Any, Optional

from fastmcp import FastMCP, Context

from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils import pcbnew_runner as pr


def _project_files(project_path: str) -> Dict[str, str]:
    files = get_project_files(project_path)
    if "schematic" not in files:
        raise FileNotFoundError("Project has no schematic (.kicad_sch)")
    if "pcb" not in files:
        raise FileNotFoundError("Project has no board (.kicad_pcb)")
    return files


def _find_freerouting() -> Optional[str]:
    env = os.environ.get("FREEROUTING_JAR")
    if env and os.path.isfile(env):
        return env
    for base in (os.path.expanduser("~"), "/Applications", "/opt", "/usr/local"):
        for name in ("freerouting.jar", "freerouting-executable.jar"):
            cand = os.path.join(base, name)
            if os.path.isfile(cand):
                return cand
    return None


def register_pcb_update_tools(mcp: FastMCP) -> None:
    """Register PCB update / autoroute tools with the MCP server."""

    @mcp.tool()
    async def update_pcb_from_schematic(project_path: str,
                                        ctx: Context = None) -> Dict[str, Any]:
        """Create/refresh board footprints from the schematic and assign nets.

        This is the KiCad "Update PCB from Schematic" step, done headlessly:
        it exports the netlist, loads each component's footprint from the
        standard libraries, places new ones on a grid, and assigns pad nets so
        the ratsnest appears. Components already on the board are kept in place.

        Requires KiCad's bundled Python (for pcbnew). Do not run while the board
        is open in the PCB Editor, or the editor may overwrite the result.

        Args:
            project_path: Path to the .kicad_pro file.
        """
        try:
            files = _project_files(project_path)
            if ctx:
                ctx.info("Exporting netlist from schematic...")
            with tempfile.TemporaryDirectory() as tmp:
                net_path = os.path.join(tmp, "netlist.net")
                pr.export_netlist(files["schematic"], net_path)
                netlist = pr.parse_kicad_netlist(net_path)
                if ctx:
                    ctx.info(
                        f"{len(netlist['components'])} components; "
                        f"{len(netlist['nets'])} nets. Updating board..."
                    )
                result = pr.update_pcb_from_netlist(files["pcb"], netlist)
            result.setdefault("success", True)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def add_pcb_outline(project_path: str, x: float, y: float,
                              width: float, height: float,
                              ctx: Context = None) -> Dict[str, Any]:
        """Add a rectangular board outline (Edge.Cuts) to the project's board.

        A board outline is required before autorouting. Do not run while the
        board is open in the PCB Editor.

        Args:
            project_path: Path to the .kicad_pro file.
            x, y: Top-left corner in mm.
            width, height: Outline size in mm.
        """
        try:
            files = _project_files(project_path)
            return pr.add_outline(files["pcb"], float(x), float(y),
                                  float(width), float(height))
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def autoroute_pcb(project_path: str, passes: int = 100,
                            ctx: Context = None) -> Dict[str, Any]:
        """Autoroute the board with Freerouting (export DSN -> route -> import SES).

        Requires Java and a Freerouting jar (set ``FREEROUTING_JAR`` or place
        ``freerouting.jar`` in your home / /Applications / /opt). The board must
        already have footprints and nets (run ``update_pcb_from_schematic``
        first). Do not run while the board is open in the PCB Editor.

        Args:
            project_path: Path to the .kicad_pro file.
            passes: Freerouting optimization passes (default 100).
        """
        try:
            files = _project_files(project_path)
            board = files["pcb"]

            java = shutil.which("java")
            jar = _find_freerouting()
            if not java:
                return {"success": False, "stage": "precheck",
                        "error": "Java runtime not found. Install a JRE to use Freerouting."}
            if not jar:
                return {"success": False, "stage": "precheck",
                        "error": ("Freerouting jar not found. Download it from "
                                  "https://github.com/freerouting/freerouting/releases "
                                  "and set FREEROUTING_JAR to its path.")}

            with tempfile.TemporaryDirectory() as tmp:
                dsn = os.path.join(tmp, "board.dsn")
                ses = os.path.join(tmp, "board.ses")
                if ctx:
                    ctx.info("Exporting Specctra DSN...")
                d = pr.export_dsn(board, dsn)
                if not d.get("success") or not os.path.exists(dsn):
                    return {"success": False, "stage": "export_dsn", "detail": d}

                if ctx:
                    ctx.info(f"Running Freerouting ({passes} passes)...")
                # Force headless, and neutralize Freerouting's startup/finish
                # telemetry. It POSTs analytics on a code path that BLOCKS the
                # job: a real but unreachable proxy makes it hang, and no proxy
                # makes the direct connect hang too. Pointing the proxy at a
                # closed local port makes that connect fail *instantly*
                # (ECONNREFUSED) so routing proceeds and the process exits.
                cmd = [java,
                       "-Djava.awt.headless=true",
                       "-jar", jar, "-de", dsn, "-do", ses,
                       "-mp", str(passes)]
                env = dict(os.environ)
                env["HTTP_PROXY"] = env["HTTPS_PROXY"] = "http://127.0.0.1:9"
                env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
                rp = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=1800, env=env)
                if not os.path.exists(ses):
                    return {"success": False, "stage": "freerouting",
                            "error": rp.stderr.strip()[-500:] or rp.stdout.strip()[-500:]}

                if ctx:
                    ctx.info("Importing routed SES back into the board...")
                s = pr.import_ses(board, ses)
                if not s.get("success"):
                    return {"success": False, "stage": "import_ses", "detail": s}

            return {"success": True, "board": board, "routed": True,
                    "freerouting_jar": jar}
        except Exception as e:
            return {"success": False, "error": str(e)}
