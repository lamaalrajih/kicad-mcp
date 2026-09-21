"""
Batch reporting tools: run DRC across many projects at once.
"""
import os
import glob
import json
import subprocess
import tempfile
from typing import Dict, Any, List

from fastmcp import FastMCP, Context

from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils.kicad_cli import get_cli_manager


def _find_projects(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.kicad_pro"), recursive=True))


def register_report_tools(mcp: FastMCP) -> None:
    """Register batch reporting tools with the MCP server."""

    @mcp.tool()
    async def batch_drc_report(directory: str, ctx: Context = None) -> Dict[str, Any]:
        """Run DRC on every KiCad project under a directory and summarize.

        Recursively finds ``*.kicad_pro`` files, runs the Design Rule Check on
        each board, and returns per-project violation counts plus a roll-up.

        Args:
            directory: Folder to scan recursively for projects.
        """
        try:
            cli = get_cli_manager().find_kicad_cli()
            if not cli:
                return {"success": False, "error": "kicad-cli not found"}
            if not os.path.isdir(directory):
                return {"success": False, "error": f"Not a directory: {directory}"}

            projects = _find_projects(directory)
            if ctx:
                ctx.info(f"Found {len(projects)} project(s); running DRC...")

            results = []
            total_violations = 0
            clean = 0
            for pro in projects:
                name = os.path.splitext(os.path.basename(pro))[0]
                try:
                    files = get_project_files(pro)
                    if "pcb" not in files:
                        results.append({"project": name, "path": pro,
                                        "status": "no_board"})
                        continue
                    with tempfile.TemporaryDirectory() as tmp:
                        report = os.path.join(tmp, "drc.json")
                        subprocess.run(
                            [cli, "pcb", "drc", "--format", "json",
                             "--output", report, files["pcb"]],
                            capture_output=True, text=True, timeout=180,
                        )
                        violations = 0
                        if os.path.exists(report):
                            data = json.load(open(report))
                            violations = len(data.get("violations", []))
                        total_violations += violations
                        if violations == 0:
                            clean += 1
                        results.append({"project": name, "path": pro,
                                        "violations": violations,
                                        "status": "clean" if violations == 0 else "violations"})
                except Exception as e:
                    results.append({"project": name, "path": pro,
                                    "status": "error", "error": str(e)})

            return {
                "success": True,
                "directory": directory,
                "project_count": len(projects),
                "clean": clean,
                "total_violations": total_violations,
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
