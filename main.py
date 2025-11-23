#!/usr/bin/env python3
"""
KiCad MCP Server - A Model Context Protocol server for KiCad on macOS.
This server allows Claude and other MCP clients to interact with KiCad projects.
"""
import os
import sys
import logging # Import logging module

# Must import config BEFORE env potentially overrides it via os.environ
from kicad_mcp import config
from kicad_mcp.server import main as server_main
from kicad_mcp.utils.env import load_dotenv

# --- Setup Logging --- 
log_file = os.path.join(os.path.dirname(__file__), 'kicad-mcp.log') 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'), # Use 'w' to overwrite log on each start
        # logging.StreamHandler() # Optionally keep logging to console if needed
    ]
)
# ---------------------

logging.info("--- Server Starting --- ")
logging.info(f"Initial KICAD_USER_DIR from config.py: {config.KICAD_USER_DIR}")
logging.info(f"Initial ADDITIONAL_SEARCH_PATHS from config.py: {config.ADDITIONAL_SEARCH_PATHS}")

# Get PID for logging (already used by basicConfig)
_PID = os.getpid()

# Load environment variables from .env file if present
# This attempts to update os.environ
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
logging.info(f"Attempting to load .env file from: {dotenv_path}")
found_dotenv = load_dotenv() # Assuming this returns True/False or similar
logging.info(f".env file found and loaded: {found_dotenv}")

# Log effective values AFTER load_dotenv attempt
# Note: The config values might not automatically re-read from os.environ 
# depending on how config.py is written. Let's check os.environ directly.
effective_user_dir = os.getenv('KICAD_USER_DIR')
effective_search_paths = os.getenv('KICAD_SEARCH_PATHS')
logging.info(f"os.environ['KICAD_USER_DIR'] after load_dotenv: {effective_user_dir}")
logging.info(f"os.environ['KICAD_SEARCH_PATHS'] after load_dotenv: {effective_search_paths}")

# Re-log the values imported from config.py to see if they reflect os.environ changes
# (This depends on config.py using os.getenv internally AFTER load_dotenv runs)
try:
    from kicad_mcp import config
    import importlib
    importlib.reload(config) # Attempt to force re-reading config
    logging.info(f"Effective KICAD_USER_DIR from config.py after reload: {config.KICAD_USER_DIR}")
    logging.info(f"Effective ADDITIONAL_SEARCH_PATHS from config.py after reload: {config.ADDITIONAL_SEARCH_PATHS}")
except Exception as e:
    logging.error(f"Could not reload config: {e}")
    logging.info(f"Using potentially stale KICAD_USER_DIR from initial import: {config.KICAD_USER_DIR}")
    logging.info(f"Using potentially stale ADDITIONAL_SEARCH_PATHS from initial import: {config.ADDITIONAL_SEARCH_PATHS}")

if __name__ == "__main__":
    try:
        logging.info(f"Starting KiCad MCP server process") 

        # Print search paths from config
        logging.info(f"Using KiCad user directory: {config.KICAD_USER_DIR}") # Changed print to logging
        if config.ADDITIONAL_SEARCH_PATHS:
            logging.info(f"Additional search paths: {', '.join(config.ADDITIONAL_SEARCH_PATHS)}") # Changed print to logging
        else:
            logging.info(f"No additional search paths configured") # Changed print to logging

        # Run server
        logging.info(f"Running server with stdio transport") # Changed print to logging
        server_main()
    except Exception as e:
        logging.exception(f"Unhandled exception in main") # Log exception details
        raise
