"""
Symbol library tools for KiCad.

Provides tools for querying symbol libraries, validating symbols,
and retrieving pin information.
"""

from difflib import SequenceMatcher
import os
import re
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from kicad_mcp.config import KICAD_APP_PATH, system
from kicad_mcp.utils.symbol_cache import (
    get_cache_path,
    get_or_build_index,
    search_index,
)


def get_symbol_library_path() -> str:
    """Get the path to KiCad's symbol libraries.

    Returns:
        Path to the symbol libraries directory
    """
    if system == "Darwin":
        return os.path.join(KICAD_APP_PATH, "Contents/SharedSupport/symbols")
    elif system == "Windows":
        return os.path.join(KICAD_APP_PATH, "share", "kicad", "symbols")
    else:  # Linux
        return "/usr/share/kicad/symbols"


def list_available_libraries() -> list[str]:
    """List all available symbol library names.

    Returns:
        Sorted list of library names (without .kicad_sym extension)
    """
    lib_path = get_symbol_library_path()
    if not os.path.exists(lib_path):
        return []

    libraries = []
    for f in os.listdir(lib_path):
        if f.endswith(".kicad_sym"):
            libraries.append(f[:-10])  # Remove .kicad_sym extension
    return sorted(libraries)


def parse_symbol_identifier(symbol_id: str) -> tuple[str, str]:
    """Parse library:symbol format into (library, symbol) tuple.

    Args:
        symbol_id: Symbol identifier in format 'Library:SymbolName'

    Returns:
        Tuple of (library_name, symbol_name)

    Raises:
        ValueError: If symbol_id is not in correct format
    """
    if ":" not in symbol_id:
        raise ValueError(
            f"Invalid symbol identifier '{symbol_id}'. Expected format: 'Library:SymbolName'"
        )
    parts = symbol_id.split(":", 1)
    return parts[0], parts[1]


def read_library_file(library_name: str) -> str | None:
    """Read a symbol library file content.

    Args:
        library_name: Name of the library (without .kicad_sym extension)

    Returns:
        Library file content as string, or None if not found
    """
    lib_path = get_symbol_library_path()
    file_path = os.path.join(lib_path, f"{library_name}.kicad_sym")

    if not os.path.exists(file_path):
        return None

    with open(file_path, encoding="utf-8") as f:
        return f.read()


def extract_symbol_content(library_content: str, symbol_name: str) -> str | None:
    """Extract a specific symbol's S-expression block from library content.

    Args:
        library_content: Full content of the library file
        symbol_name: Name of the symbol to extract

    Returns:
        Symbol S-expression block, or None if not found
    """
    # Match symbol block - handles nested parentheses
    pattern = rf'\(symbol "{re.escape(symbol_name)}"'
    match = re.search(pattern, library_content)

    if not match:
        return None

    start = match.start()
    depth = 0
    end = start

    for i, char in enumerate(library_content[start:], start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    return library_content[start:end]


def parse_pins_from_symbol(symbol_content: str, library_content: str) -> list[dict[str, Any]]:
    """Parse pin information from symbol S-expression content.

    Handles symbol inheritance via 'extends'.

    Args:
        symbol_content: S-expression content of the symbol
        library_content: Full library content (for resolving extends)

    Returns:
        List of pin dictionaries with name, number, position, etc.
    """
    pins = []

    # Check for extends clause
    extends_match = re.search(r'\(extends "([^"]+)"\)', symbol_content)
    if extends_match:
        base_symbol = extends_match.group(1)
        base_content = extract_symbol_content(library_content, base_symbol)
        if base_content:
            pins = parse_pins_from_symbol(base_content, library_content)

    # Parse pins from this symbol (may override inherited pins)
    # Pin format: (pin <type> <shape> (at X Y ANGLE) (length L) (name "NAME" ...) (number "NUM" ...))
    pin_pattern = re.compile(
        r"\(pin\s+(\w+)\s+(\w+)\s*"  # type and shape
        r"\(at\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\)"  # at X Y ANGLE
        r"\s*\(length\s+([\d.-]+)\)"  # length
        r'\s*\(name\s+"([^"]*)"\s*'  # name
        r'.*?\(number\s+"([^"]+)"',  # number
        re.DOTALL,
    )

    for match in pin_pattern.finditer(symbol_content):
        pin_type, pin_shape, x, y, angle, length, name, number = match.groups()
        pins.append(
            {
                "name": name if name != "~" else "",
                "number": number,
                "position": [float(x), float(y)],
                "angle": float(angle),
                "length": float(length),
                "type": pin_type,
                "shape": pin_shape,
            }
        )

    return pins


# Pre-compiled patterns for symbol extraction
_SYMBOL_HEADER_PATTERN = re.compile(r'\n\t\(symbol "([^"]+)"')
_DESC_PATTERN = re.compile(r'\(property "Description"\s+"([^"]*)"')
_KEYWORDS_PATTERN = re.compile(r'\(property "ki_keywords"\s+"([^"]*)"')


def get_all_symbols_in_library(library_content: str) -> list[dict[str, str]]:
    """Extract all symbol names and descriptions from a library.

    Uses single-pass parsing for efficiency.

    Args:
        library_content: Full content of the library file

    Returns:
        List of dictionaries with name, description, and keywords
    """
    symbols = []

    # Find all symbol start positions first
    symbol_matches = list(_SYMBOL_HEADER_PATTERN.finditer(library_content))

    for i, match in enumerate(symbol_matches):
        symbol_name = match.group(1)

        # Skip internal symbol units (e.g., Symbol_0_1, Symbol_1_1)
        if "_" in symbol_name:
            parts = symbol_name.split("_")
            if parts[-1].isdigit():
                continue
            if len(parts) >= 2 and parts[-2].isdigit():
                continue

        # Get block boundaries - from this symbol to next (or end)
        block_start = match.start()
        if i + 1 < len(symbol_matches):
            block_end = symbol_matches[i + 1].start()
        else:
            block_end = len(library_content)

        block = library_content[block_start:block_end]

        # Extract description and keywords from block
        description = ""
        keywords = ""

        desc_match = _DESC_PATTERN.search(block)
        if desc_match:
            description = desc_match.group(1)

        kw_match = _KEYWORDS_PATTERN.search(block)
        if kw_match:
            keywords = kw_match.group(1)

        symbols.append(
            {
                "name": symbol_name,
                "description": description,
                "keywords": keywords,
            }
        )

    return symbols


def find_similar_symbols(target: str, all_symbols: list[str], limit: int = 5) -> list[str]:
    """Find symbols similar to the target name.

    Args:
        target: Symbol name to find similar matches for
        all_symbols: List of all available symbol names
        limit: Maximum number of suggestions to return

    Returns:
        List of similar symbol names, sorted by similarity
    """
    scored = []
    target_lower = target.lower()

    for sym in all_symbols:
        sym_lower = sym.lower()
        # Exact substring match gets high score
        if target_lower in sym_lower:
            score = 0.8 + (len(target) / len(sym)) * 0.2
        else:
            score = SequenceMatcher(None, target_lower, sym_lower).ratio()
        scored.append((sym, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:limit]]


def register_symbol_tools(mcp: FastMCP) -> None:
    """Register symbol library tools with the MCP server.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.tool()
    def get_symbol_pins(symbol_id: str) -> dict[str, Any]:
        """Get pin information for a KiCad symbol.

        Args:
            symbol_id: Symbol identifier in format 'Library:SymbolName'
                       (e.g., 'MCU_Microchip_ATtiny:ATtiny85-20P', 'Device:R')

        Returns:
            Dictionary with pin information including names, numbers, positions
        """
        print(f"Getting pins for symbol: {symbol_id}")

        try:
            library, symbol_name = parse_symbol_identifier(symbol_id)
        except ValueError as e:
            print(f"Invalid symbol identifier: {symbol_id}")
            return {"success": False, "error": str(e)}

        library_content = read_library_file(library)
        if library_content is None:
            print(f"Library not found: {library}")
            available = list_available_libraries()
            similar = find_similar_symbols(library, available)
            return {
                "success": False,
                "error": f"Library '{library}' not found",
                "suggestions": similar,
            }

        symbol_content = extract_symbol_content(library_content, symbol_name)
        if symbol_content is None:
            print(f"Symbol not found: {symbol_name} in library {library}")
            # Find similar symbols in this library
            all_syms = get_all_symbols_in_library(library_content)
            sym_names = [s["name"] for s in all_syms]
            similar = find_similar_symbols(symbol_name, sym_names)
            return {
                "success": False,
                "error": f"Symbol '{symbol_name}' not found in library '{library}'",
                "suggestions": [f"{library}:{s}" for s in similar],
            }

        pins = parse_pins_from_symbol(symbol_content, library_content)
        print(f"Found {len(pins)} pins for {symbol_id}")

        return {
            "success": True,
            "symbol": symbol_id,
            "pin_count": len(pins),
            "pins": pins,
        }

    @mcp.tool()
    def validate_symbol_exists(symbol_id: str) -> dict[str, Any]:
        """Check if a KiCad symbol exists and suggest alternatives if not.

        Args:
            symbol_id: Symbol identifier in format 'Library:SymbolName'

        Returns:
            Dictionary with exists status and suggestions if not found
        """
        print(f"Validating symbol: {symbol_id}")

        try:
            library, symbol_name = parse_symbol_identifier(symbol_id)
        except ValueError as e:
            return {"exists": False, "error": str(e), "suggestions": []}

        library_content = read_library_file(library)
        if library_content is None:
            print(f"Library not found: {library}")
            available = list_available_libraries()
            similar = find_similar_symbols(library, available)
            return {
                "exists": False,
                "error": f"Library '{library}' not found",
                "suggestions": [f"{lib}:{symbol_name}" for lib in similar[:3]],
            }

        symbol_content = extract_symbol_content(library_content, symbol_name)
        if symbol_content is None:
            print(f"Symbol not found: {symbol_name}")
            all_syms = get_all_symbols_in_library(library_content)
            sym_names = [s["name"] for s in all_syms]
            similar = find_similar_symbols(symbol_name, sym_names)
            return {
                "exists": False,
                "error": f"Symbol '{symbol_name}' not found in library '{library}'",
                "suggestions": [f"{library}:{s}" for s in similar],
            }

        # Extract additional info for valid symbol
        desc_match = re.search(r'\(property "Description"\s+"([^"]*)"', symbol_content)
        footprint_match = re.search(r'\(property "Footprint"\s+"([^"]*)"', symbol_content)

        print(f"Symbol exists: {symbol_id}")
        return {
            "exists": True,
            "symbol": symbol_id,
            "description": desc_match.group(1) if desc_match else "",
            "default_footprint": footprint_match.group(1) if footprint_match else "",
        }

    @mcp.tool()
    def search_symbols(query: str, library: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Search for KiCad symbols by keyword.

        Args:
            query: Search keyword (e.g., 'ESP32', 'voltage regulator', 'ATmega')
            library: Optional library name to search within (searches all if not specified)
            limit: Maximum number of results to return (default 20)

        Returns:
            List of matching symbols with their library:symbol identifiers
        """
        from kicad_mcp.utils.symbol_cache import is_cache_valid, load_symbol_index

        print(f"Searching symbols for: {query}")

        library_path = get_symbol_library_path()
        cached = load_symbol_index()

        # Check if cache exists and is valid
        if not cached:
            return {
                "success": False,
                "cache_status": "missing",
                "message": "Symbol index not built yet. Run rebuild_symbol_index first (takes ~2-3 min).",
            }

        if not is_cache_valid(cached, library_path):
            return {
                "success": False,
                "cache_status": "stale",
                "message": "Symbol libraries have changed. Run rebuild_symbol_index to update (~2-3 min).",
            }

        # Search the index
        search_result = search_index(cached, query, library, limit)

        print(f"Found {search_result['total_matches']} matching symbols")
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
    async def rebuild_symbol_index(ctx: Context | None = None) -> dict[str, Any]:
        """Rebuild the symbol index cache.

        Force a rebuild of the cached symbol index. Use this when:
        - KiCad libraries have been updated
        - New symbol libraries have been installed
        - Symbol index appears corrupted or stale

        Returns:
            Dictionary with rebuild results
        """
        print("Rebuilding symbol index...")

        library_path = get_symbol_library_path()
        index_data = await get_or_build_index(library_path, force_rebuild=True, ctx=ctx)

        return {
            "success": True,
            "message": "Symbol index rebuilt successfully",
            "symbol_count": index_data["symbol_count"],
            "library_count": index_data["library_count"],
            "cache_path": get_cache_path(),
            "timestamp": index_data["timestamp"],
        }

    @mcp.tool()
    def list_symbol_libraries() -> dict[str, Any]:
        """List all available KiCad symbol libraries.

        Returns:
            List of library names that can be used with other symbol tools
        """
        print("Listing symbol libraries")
        libraries = list_available_libraries()
        return {
            "success": True,
            "library_count": len(libraries),
            "libraries": libraries,
            "library_path": get_symbol_library_path(),
        }
