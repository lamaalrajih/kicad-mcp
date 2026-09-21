"""
Library discovery tools: find symbols and footprints in KiCad's libraries.

These help pick valid ``lib_id`` values for ``add_schematic_component`` and
footprint names for placement, without leaving the assistant.
"""
import os
import re
import glob
from typing import Dict, Any, List

from fastmcp import FastMCP, Context

from kicad_mcp.utils import sexpr_author as author

# Top-level symbols sit at a single-tab indent; sub-symbols (e.g. "R_0_1")
# are deeper, so this only matches the real symbol names.
_SYMBOL_RE = re.compile(r'\n\t\(symbol "([^"]+)"')


def _footprint_library_dirs() -> List[str]:
    shared = author._shared_support_dir()
    dirs = []
    if shared:
        dirs.append(os.path.join(shared, "footprints"))
    env = os.environ.get("KICAD_FOOTPRINT_DIR")
    if env:
        dirs = [p for p in env.split(os.pathsep) if p] + dirs
    return [d for d in dirs if os.path.isdir(d)]


def register_library_tools(mcp: FastMCP) -> None:
    """Register library discovery tools with the MCP server."""

    @mcp.tool()
    async def list_symbol_libraries(ctx: Context = None) -> Dict[str, Any]:
        """List the available KiCad symbol libraries (``.kicad_sym``)."""
        libs = []
        for d in author.symbol_library_dirs():
            for p in sorted(glob.glob(os.path.join(d, "*.kicad_sym"))):
                libs.append(os.path.splitext(os.path.basename(p))[0])
        libs = sorted(set(libs))
        return {"success": True, "count": len(libs), "libraries": libs}

    @mcp.tool()
    async def search_symbols(query: str, library: str = "",
                             limit: int = 50, ctx: Context = None) -> Dict[str, Any]:
        """Search KiCad symbol libraries for symbols matching ``query``.

        Returns fully-qualified ``lib_id`` values (``Library:Symbol``) usable
        directly with ``add_schematic_component``.

        Args:
            query: Case-insensitive substring to match against symbol names.
            library: Restrict to one library (e.g. ``"Device"``); empty = all.
            limit: Maximum results (default 50).
        """
        q = query.lower()
        results: List[str] = []
        for d in author.symbol_library_dirs():
            pattern = f"{library}.kicad_sym" if library else "*.kicad_sym"
            for path in sorted(glob.glob(os.path.join(d, pattern))):
                lib = os.path.splitext(os.path.basename(path))[0]
                try:
                    text = open(path, "r", encoding="utf-8").read()
                except OSError:
                    continue
                for name in _SYMBOL_RE.findall(text):
                    if q in name.lower():
                        results.append(f"{lib}:{name}")
                        if len(results) >= limit:
                            return {"success": True, "count": len(results),
                                    "truncated": True, "lib_ids": results}
        return {"success": True, "count": len(results),
                "truncated": False, "lib_ids": results}

    @mcp.tool()
    async def search_footprints(query: str, limit: int = 50,
                                ctx: Context = None) -> Dict[str, Any]:
        """Search KiCad footprint libraries for footprints matching ``query``.

        Returns ``Library:Footprint`` identifiers usable as the ``footprint``
        argument of ``add_schematic_component``.

        Args:
            query: Case-insensitive substring to match against footprint names.
            limit: Maximum results (default 50).
        """
        q = query.lower()
        results: List[str] = []
        for base in _footprint_library_dirs():
            for pretty in sorted(glob.glob(os.path.join(base, "*.pretty"))):
                lib = os.path.splitext(os.path.basename(pretty))[0]
                for mod in sorted(glob.glob(os.path.join(pretty, "*.kicad_mod"))):
                    name = os.path.splitext(os.path.basename(mod))[0]
                    if q in name.lower() or q in lib.lower():
                        results.append(f"{lib}:{name}")
                        if len(results) >= limit:
                            return {"success": True, "count": len(results),
                                    "truncated": True, "footprints": results}
        return {"success": True, "count": len(results),
                "truncated": False, "footprints": results}
