"""
Lifespan context management for KiCad MCP Server.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Any
import logging  # Import logging
import os  # Added for PID

from mcp.server.fastmcp import FastMCP

# Get PID for logging
# _PID = os.getpid()


@dataclass
class KiCadAppContext:
    """Type-safe context for KiCad MCP server."""

    kicad_modules_available: bool

    # Optional cache for expensive operations
    cache: Dict[str, Any]


@asynccontextmanager
async def kicad_lifespan(
    server: FastMCP, kicad_modules_available: bool = False
) -> AsyncIterator[KiCadAppContext]:
    """Manage KiCad MCP server lifecycle with type-safe context.

    This function handles:
    1. Initializing shared resources when the server starts
    2. Providing a typed context object to all request handlers
    3. Properly cleaning up resources when the server shuts down

    Args:
        server: The FastMCP server instance
        kicad_modules_available: Flag indicating if Python modules were found (passed from create_server)

    Yields:
        KiCadAppContext: A typed context object shared across all handlers
    """
    logging.info(f"Starting KiCad MCP server initialization")

    # Resources initialization - Python path setup removed
    # print("Setting up KiCad Python modules")
    # kicad_modules_available = setup_kicad_python_path() # Now passed as arg
    logging.info(
        f"KiCad Python module availability: {kicad_modules_available} (Setup logic removed)"
    )

    # Create in-memory cache for expensive operations
    cache: Dict[str, Any] = {}

    # Initialize any other resources that need cleanup later
    created_temp_dirs = []  # Assuming this is managed elsewhere or not needed for now

    try:
        # --- Removed Python module preloading section ---
        # if kicad_modules_available:
        #     try:
        #         print("Preloading KiCad Python modules")
        #         ...
        #     except ImportError as e:
        #         print(f"Failed to preload some KiCad modules: {str(e)}")

        # Yield the context to the server - server runs during this time
        logging.info(f"KiCad MCP server initialization complete")
        yield KiCadAppContext(
            kicad_modules_available=kicad_modules_available,  # Pass the flag through
            cache=cache,
        )
    finally:
        # Clean up resources when server shuts down
        logging.info(f"Shutting down KiCad MCP server")

        # Clear the cache
        if cache:
            logging.info(f"Clearing cache with {len(cache)} entries")
            cache.clear()

        # Clean up any temporary directories
        import shutil

        for temp_dir in created_temp_dirs:
            try:
                logging.info(f"Removing temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")

        logging.info(f"KiCad MCP server shutdown complete")
