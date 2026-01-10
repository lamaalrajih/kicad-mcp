"""
Symbol index cache management for fast symbol searches.

Builds and maintains a cached index of all KiCad symbols to avoid
parsing 223 library files on every search.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import logging
import os
from typing import Any

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


def get_cache_dir() -> str:
    """Get the cache directory path.

    Returns:
        Path to cache directory (created if doesn't exist)
    """
    if os.name == "nt":  # Windows
        cache_base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        cache_dir = os.path.join(cache_base, "kicad-mcp", "cache")
    else:  # Linux/macOS
        cache_base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(cache_base, "kicad-mcp")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_path() -> str:
    """Get the symbol index cache file path.

    Returns:
        Full path to symbol_index.json
    """
    return os.path.join(get_cache_dir(), "symbol_index.json")


def get_lock_path() -> str:
    """Get the lock file path for cache operations.

    Returns:
        Full path to symbol_index.lock
    """
    return os.path.join(get_cache_dir(), "symbol_index.lock")


def compute_library_hash(library_path: str) -> str:
    """Compute hash of library directory to detect changes.

    Args:
        library_path: Path to KiCad symbols directory

    Returns:
        MD5 hash of all .kicad_sym file modification times
    """
    if not os.path.exists(library_path):
        return ""

    hash_input = []
    for filename in sorted(os.listdir(library_path)):
        if filename.endswith(".kicad_sym"):
            filepath = os.path.join(library_path, filename)
            mtime = os.path.getmtime(filepath)
            hash_input.append(f"{filename}:{mtime}")

    combined = "|".join(hash_input)
    return hashlib.md5(combined.encode()).hexdigest()


def is_cache_valid(cache_data: dict[str, Any], library_path: str) -> bool:
    """Check if cached index is still valid.

    Args:
        cache_data: Loaded cache data
        library_path: Current KiCad symbols directory path

    Returns:
        True if cache is valid, False if needs rebuild
    """
    # Check version
    if cache_data.get("version") != 1:
        return False

    # Check if library path changed
    if cache_data.get("library_path") != library_path:
        return False

    # Check if library files changed
    current_hash = compute_library_hash(library_path)
    cached_hash: str = cache_data.get("library_hash", "")

    return current_hash == cached_hash


def load_symbol_index() -> dict[str, Any] | None:
    """Load symbol index from cache.

    Returns:
        Cached index data, or None if cache doesn't exist
    """
    cache_path = get_cache_path()

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    except Exception as e:
        logger.warning(f"Error loading symbol cache: {e}")
        return None


def save_symbol_index(index_data: dict[str, Any]) -> bool:
    """Save symbol index to cache with file locking.

    Args:
        index_data: Index data to save

    Returns:
        True if saved successfully, False otherwise
    """
    cache_path = get_cache_path()
    lock = FileLock(get_lock_path(), timeout=300)

    try:
        with lock:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2)
            logger.info(f"Symbol index saved to: {cache_path}")
        return True
    except Timeout:
        logger.error("Timeout waiting for cache lock")
        return False
    except Exception as e:
        logger.error(f"Error saving symbol cache: {e}")
        return False


def _process_library(lib_name: str, read_func, parse_func) -> list[dict[str, Any]]:
    """Process a single library file. Used by ThreadPoolExecutor."""
    try:
        lib_content = read_func(lib_name)
        if lib_content is None:
            logger.debug(f"Library not found: {lib_name}")
            return []

        lib_symbols = parse_func(lib_content)
        return [
            {
                "library": lib_name,
                "name": sym["name"],
                "description": sym["description"],
                "keywords": sym["keywords"],
                "id": f"{lib_name}:{sym['name']}",
            }
            for sym in lib_symbols
        ]
    except Exception as e:
        logger.error(f"Error processing library {lib_name}: {e}")
        return []


async def build_symbol_index_data(library_path: str, ctx=None) -> dict[str, Any]:
    """Build symbol index data structure using parallel I/O.

    Args:
        library_path: Path to KiCad symbols directory
        ctx: Optional MCP context for progress reporting

    Returns:
        Index data structure ready to cache
    """
    import importlib

    st = importlib.import_module("kicad_mcp.tools.symbol_tools")

    logger.info(f"Building symbol index from: {library_path}")

    libraries = st.list_available_libraries()
    total = len(libraries)
    symbols = []

    logger.info(f"Indexing {total} libraries (parallel)...")

    # Use ThreadPoolExecutor for parallel file I/O
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _process_library, lib_name, st.read_library_file, st.get_all_symbols_in_library
            ): lib_name
            for lib_name in libraries
        }

        for completed, future in enumerate(as_completed(futures), 1):
            if completed % 20 == 0:
                logger.debug(f"Processed {completed}/{total} libraries...")

            if ctx:
                progress = int((completed / total) * 90) + 5
                await ctx.report_progress(progress, 100)

            lib_symbols = future.result()
            symbols.extend(lib_symbols)

    logger.info(f"Indexed {len(symbols)} symbols from {total} libraries")

    index_data = {
        "version": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "library_path": library_path,
        "library_hash": compute_library_hash(library_path),
        "symbol_count": len(symbols),
        "library_count": len(libraries),
        "symbols": symbols,
    }

    return index_data


async def get_or_build_index(
    library_path: str, force_rebuild: bool = False, ctx=None
) -> dict[str, Any]:
    """Get symbol index, building if needed.

    Args:
        library_path: Path to KiCad symbols directory
        force_rebuild: Force rebuild even if cache is valid
        ctx: Optional MCP context for progress reporting

    Returns:
        Symbol index data
    """
    if ctx:
        await ctx.report_progress(1, 100)

    if not force_rebuild:
        cached = load_symbol_index()
        if cached and is_cache_valid(cached, library_path):
            logger.debug(f"Using cached symbol index ({cached['symbol_count']} symbols)")
            if ctx:
                await ctx.report_progress(100, 100)
            return cached

    logger.info("Symbol index cache invalid or missing, rebuilding...")

    index_data = await build_symbol_index_data(library_path, ctx)

    # Save to cache
    save_symbol_index(index_data)

    if ctx:
        await ctx.report_progress(100, 100)

    return index_data


def search_index(
    index_data: dict[str, Any], query: str, library: str | None = None, limit: int = 20
) -> dict[str, Any]:
    """Search the symbol index.

    Args:
        index_data: Symbol index data
        query: Search query
        library: Optional library filter
        limit: Maximum results (0 for count only)

    Returns:
        Dict with total_matches, results list, and truncated flag
    """
    query_lower = query.lower()
    results = []

    for sym in index_data["symbols"]:
        if library and sym["library"] != library:
            continue

        searchable = f"{sym['name']} {sym['description']} {sym['keywords']}".lower()

        if query_lower in searchable:
            score = 0
            if query_lower in sym["name"].lower():
                score += 10
            if query_lower in sym["description"].lower():
                score += 5
            if query_lower in sym["keywords"].lower():
                score += 3

            results.append(
                {
                    "symbol": sym["id"],
                    "description": sym["description"],
                    "keywords": sym["keywords"],
                    "_score": score,
                }
            )

    results.sort(key=lambda x: x["_score"], reverse=True)
    total_matches = len(results)
    truncated = total_matches > limit and limit > 0

    if limit > 0:
        results = results[:limit]

    for r in results:
        del r["_score"]

    return {"total_matches": total_matches, "truncated": truncated, "results": results}
