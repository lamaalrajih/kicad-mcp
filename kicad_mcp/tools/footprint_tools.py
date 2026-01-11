"""
Footprint library tools for KiCad.

Provides tools for querying footprint libraries, validating footprints,
and retrieving pad information.
"""

from difflib import SequenceMatcher
import os
import re
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from kicad_mcp.config import KICAD_APP_PATH, system
from kicad_mcp.utils.footprint_cache import (
    get_cache_path,
    get_or_build_index,
    search_index,
)


def get_footprint_library_path() -> str:
    """Get the path to KiCad's footprint libraries.

    Returns:
        Path to the footprint libraries directory
    """
    if system == "Darwin":
        return os.path.join(KICAD_APP_PATH, "Contents/SharedSupport/footprints")
    elif system == "Windows":
        return os.path.join(KICAD_APP_PATH, "share", "kicad", "footprints")
    else:  # Linux
        return "/usr/share/kicad/footprints"


def list_available_libraries() -> list[str]:
    """List all available footprint library names.

    Returns:
        Sorted list of library names (without .pretty extension)
    """
    lib_path = get_footprint_library_path()
    if not os.path.exists(lib_path):
        return []

    libraries = []
    for d in os.listdir(lib_path):
        if d.endswith(".pretty") and os.path.isdir(os.path.join(lib_path, d)):
            libraries.append(d[:-7])  # Remove .pretty extension
    return sorted(libraries)


def parse_footprint_identifier(footprint_id: str) -> tuple[str, str]:
    """Parse library:footprint format into (library, footprint) tuple.

    Args:
        footprint_id: Footprint identifier in format 'Library:FootprintName'

    Returns:
        Tuple of (library_name, footprint_name)

    Raises:
        ValueError: If footprint_id is not in correct format
    """
    if ":" not in footprint_id:
        raise ValueError(
            f"Invalid footprint identifier '{footprint_id}'. Expected format: 'Library:FootprintName'"
        )
    parts = footprint_id.split(":", 1)
    return parts[0], parts[1]


def read_footprint_file(library_name: str, footprint_name: str) -> str | None:
    """Read a footprint file content.

    Args:
        library_name: Name of the library (without .pretty extension)
        footprint_name: Name of the footprint (without .kicad_mod extension)

    Returns:
        Footprint file content as string, or None if not found
    """
    lib_path = get_footprint_library_path()
    file_path = os.path.join(lib_path, f"{library_name}.pretty", f"{footprint_name}.kicad_mod")

    if not os.path.exists(file_path):
        return None

    with open(file_path, encoding="utf-8") as f:
        return f.read()


# Pre-compiled patterns for footprint parsing
_PAD_PATTERN = re.compile(
    r'\(pad\s+"?([^"\s]+)"?\s+(\w+)\s+(\w+)'  # number, type, shape
    r".*?\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\)",  # at X Y [angle]
    re.DOTALL,
)
_PAD_SIZE_PATTERN = re.compile(r"\(size\s+([\d.-]+)\s+([\d.-]+)\)")
_PAD_DRILL_PATTERN = re.compile(r"\(drill\s+([\d.-]+)(?:\s+([\d.-]+))?\)")
_DESC_PATTERN = re.compile(r'\(descr\s+"([^"]*)"')
_TAGS_PATTERN = re.compile(r'\(tags\s+"([^"]*)"')
_FP_NAME_PATTERN = re.compile(r'\(footprint\s+"([^"]+)"')


def parse_pads_from_footprint(footprint_content: str) -> list[dict[str, Any]]:
    """Parse pad information from footprint S-expression content.

    Args:
        footprint_content: S-expression content of the footprint

    Returns:
        List of pad dictionaries with number, type, shape, position, size, etc.
    """
    pads = []

    for match in _PAD_PATTERN.finditer(footprint_content):
        pad_num, pad_type, pad_shape, x, y, angle = match.groups()

        # Get the pad block for additional info
        pad_start = match.start()
        depth = 0
        pad_end = pad_start
        for i, char in enumerate(footprint_content[pad_start:], pad_start):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    pad_end = i + 1
                    break

        pad_block = footprint_content[pad_start:pad_end]

        # Extract size
        size_match = _PAD_SIZE_PATTERN.search(pad_block)
        size = [float(size_match.group(1)), float(size_match.group(2))] if size_match else [0, 0]

        # Extract drill (for through-hole pads)
        drill_match = _PAD_DRILL_PATTERN.search(pad_block)
        drill = None
        if drill_match:
            drill_x = float(drill_match.group(1))
            drill_y = float(drill_match.group(2)) if drill_match.group(2) else drill_x
            drill = [drill_x, drill_y]

        pads.append(
            {
                "number": pad_num,
                "type": pad_type,  # smd, thru_hole, np_thru_hole, connect
                "shape": pad_shape,  # circle, rect, oval, roundrect, trapezoid, custom
                "position": [float(x), float(y)],
                "angle": float(angle) if angle else 0.0,
                "size": size,
                "drill": drill,
            }
        )

    return pads


def get_all_footprints_in_library(library_path: str) -> list[dict[str, str]]:
    """Extract all footprint names and descriptions from a library directory.

    Args:
        library_path: Full path to the .pretty directory

    Returns:
        List of dictionaries with name, description, and keywords
    """
    footprints = []

    if not os.path.exists(library_path):
        return []

    for filename in os.listdir(library_path):
        if not filename.endswith(".kicad_mod"):
            continue

        footprint_name = filename[:-10]  # Remove .kicad_mod
        filepath = os.path.join(library_path, filename)

        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            description = ""
            keywords = ""

            desc_match = _DESC_PATTERN.search(content)
            if desc_match:
                description = desc_match.group(1)

            tags_match = _TAGS_PATTERN.search(content)
            if tags_match:
                keywords = tags_match.group(1)

            footprints.append(
                {
                    "name": footprint_name,
                    "description": description,
                    "keywords": keywords,
                }
            )
        except Exception:
            # Skip unreadable files
            continue

    return footprints


def find_similar_footprints(target: str, all_footprints: list[str], limit: int = 5) -> list[str]:
    """Find footprints similar to the target name.

    Args:
        target: Footprint name to find similar matches for
        all_footprints: List of all available footprint names
        limit: Maximum number of suggestions to return

    Returns:
        List of similar footprint names, sorted by similarity
    """
    scored = []
    target_lower = target.lower()

    for fp in all_footprints:
        fp_lower = fp.lower()
        # Exact substring match gets high score
        if target_lower in fp_lower:
            score = 0.8 + (len(target) / len(fp)) * 0.2
        else:
            score = SequenceMatcher(None, target_lower, fp_lower).ratio()
        scored.append((fp, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:limit]]


def register_footprint_tools(mcp: FastMCP) -> None:
    """Register footprint library tools with the MCP server.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.tool()
    def get_footprint_pads(footprint_id: str) -> dict[str, Any]:
        """Get pad information for a KiCad footprint.

        Args:
            footprint_id: Footprint identifier in format 'Library:FootprintName'
                         (e.g., 'Resistor_SMD:R_0805_2012Metric', 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm')

        Returns:
            Dictionary with pad information including numbers, types, positions, sizes
        """
        print(f"Getting pads for footprint: {footprint_id}")

        try:
            library, footprint_name = parse_footprint_identifier(footprint_id)
        except ValueError as e:
            print(f"Invalid footprint identifier: {footprint_id}")
            return {"success": False, "error": str(e)}

        footprint_content = read_footprint_file(library, footprint_name)
        if footprint_content is None:
            print(f"Footprint not found: {footprint_id}")
            available = list_available_libraries()
            similar = find_similar_footprints(library, available)
            return {
                "success": False,
                "error": f"Footprint '{footprint_name}' not found in library '{library}'",
                "suggestions": similar,
            }

        pads = parse_pads_from_footprint(footprint_content)
        print(f"Found {len(pads)} pads for {footprint_id}")

        return {
            "success": True,
            "footprint": footprint_id,
            "pad_count": len(pads),
            "pads": pads,
        }

    @mcp.tool()
    def validate_footprint_exists(footprint_id: str) -> dict[str, Any]:
        """Check if a KiCad footprint exists and suggest alternatives if not.

        Args:
            footprint_id: Footprint identifier in format 'Library:FootprintName'

        Returns:
            Dictionary with exists status and suggestions if not found
        """
        print(f"Validating footprint: {footprint_id}")

        try:
            library, footprint_name = parse_footprint_identifier(footprint_id)
        except ValueError as e:
            return {"exists": False, "error": str(e), "suggestions": []}

        footprint_content = read_footprint_file(library, footprint_name)
        if footprint_content is None:
            print(f"Footprint not found: {footprint_id}")
            lib_path = get_footprint_library_path()
            pretty_path = os.path.join(lib_path, f"{library}.pretty")

            if not os.path.exists(pretty_path):
                # Library doesn't exist
                available = list_available_libraries()
                similar = find_similar_footprints(library, available)
                return {
                    "exists": False,
                    "error": f"Library '{library}' not found",
                    "suggestions": [f"{lib}:{footprint_name}" for lib in similar[:3]],
                }

            # Library exists but footprint doesn't
            all_fps = get_all_footprints_in_library(pretty_path)
            fp_names = [fp["name"] for fp in all_fps]
            similar = find_similar_footprints(footprint_name, fp_names)
            return {
                "exists": False,
                "error": f"Footprint '{footprint_name}' not found in library '{library}'",
                "suggestions": [f"{library}:{fp}" for fp in similar],
            }

        # Extract additional info for valid footprint
        desc_match = _DESC_PATTERN.search(footprint_content)
        tags_match = _TAGS_PATTERN.search(footprint_content)

        print(f"Footprint exists: {footprint_id}")
        return {
            "exists": True,
            "footprint": footprint_id,
            "description": desc_match.group(1) if desc_match else "",
            "keywords": tags_match.group(1) if tags_match else "",
        }

    @mcp.tool()
    def search_footprints(
        query: str, library: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search for KiCad footprints by keyword.

        Args:
            query: Search keyword (e.g., 'SOIC', '0805', 'QFP')
            library: Optional library name to search within (searches all if not specified)
            limit: Maximum number of results to return (default 20)

        Returns:
            List of matching footprints with their library:footprint identifiers
        """
        from kicad_mcp.utils.footprint_cache import is_cache_valid, load_footprint_index

        print(f"Searching footprints for: {query}")

        library_path = get_footprint_library_path()
        cached = load_footprint_index()

        # Check if cache exists and is valid
        if not cached:
            return {
                "success": False,
                "cache_status": "missing",
                "message": "Footprint index not built yet. Run rebuild_footprint_index first.",
            }

        if not is_cache_valid(cached, library_path):
            return {
                "success": False,
                "cache_status": "stale",
                "message": "Footprint libraries have changed. Run rebuild_footprint_index to update.",
            }

        # Search the index
        search_result = search_index(cached, query, library, limit)

        print(f"Found {search_result['total_matches']} matching footprints")
        return {
            "success": True,
            "cache_status": "valid",
            "query": query,
            "total_matches": search_result["total_matches"],
            "result_count": len(search_result["results"]),
            "truncated": search_result["truncated"],
            "results": search_result["results"],
        }

    @mcp.tool()
    async def rebuild_footprint_index(ctx: Context | None = None) -> dict[str, Any]:
        """Rebuild the footprint index cache.

        Force a rebuild of the cached footprint index. Use this when:
        - KiCad libraries have been updated
        - New footprint libraries have been installed
        - Footprint index appears corrupted or stale

        Returns:
            Dictionary with rebuild results
        """
        print("Rebuilding footprint index...")

        library_path = get_footprint_library_path()
        index_data = await get_or_build_index(library_path, force_rebuild=True, ctx=ctx)

        return {
            "success": True,
            "message": "Footprint index rebuilt successfully",
            "footprint_count": index_data["footprint_count"],
            "library_count": index_data["library_count"],
            "cache_path": get_cache_path(),
            "timestamp": index_data["timestamp"],
        }

    @mcp.tool()
    def list_footprint_libraries() -> dict[str, Any]:
        """List all available KiCad footprint libraries.

        Returns:
            List of library names that can be used with other footprint tools
        """
        print("Listing footprint libraries")
        libraries = list_available_libraries()
        return {
            "success": True,
            "library_count": len(libraries),
            "libraries": libraries,
            "library_path": get_footprint_library_path(),
        }
