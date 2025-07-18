"""
MCP server creation and configuration.
"""
import atexit
import os
import signal
import logging
import functools
from typing import Callable
from fastmcp import FastMCP

# Import resource handlers
from kicad_mcp.resources.projects import register_project_resources
from kicad_mcp.resources.files import register_file_resources
from kicad_mcp.resources.drc_resources import register_drc_resources
from kicad_mcp.resources.bom_resources import register_bom_resources
from kicad_mcp.resources.netlist_resources import register_netlist_resources
from kicad_mcp.resources.pattern_resources import register_pattern_resources


# Import tool handlers
from kicad_mcp.tools.project_tools import register_project_tools
from kicad_mcp.tools.analysis_tools import register_analysis_tools
from kicad_mcp.tools.export_tools import register_export_tools
from kicad_mcp.tools.drc_tools import register_drc_tools
from kicad_mcp.tools.bom_tools import register_bom_tools
from kicad_mcp.tools.netlist_tools import register_netlist_tools
from kicad_mcp.tools.pattern_tools import register_pattern_tools

# Import prompt handlers
from kicad_mcp.prompts.templates import register_prompts
from kicad_mcp.prompts.drc_prompt import register_drc_prompts
from kicad_mcp.prompts.bom_prompts import register_bom_prompts
from kicad_mcp.prompts.pattern_prompts import register_pattern_prompts

# Import context management
from kicad_mcp.context import kicad_lifespan

# Track cleanup handlers
cleanup_handlers = []

# Flag to track whether we're already in shutdown process
_shutting_down = False

# Store server instance for clean shutdown
_server_instance = None

def add_cleanup_handler(handler: Callable) -> None:
    """Register a function to be called during cleanup.
    
    Args:
        handler: Function to call during cleanup
    """
    cleanup_handlers.append(handler)

def run_cleanup_handlers() -> None:
    """Run all registered cleanup handlers."""
    logging.info(f"Running cleanup handlers...")

    global _shutting_down
    
    # Prevent running cleanup handlers multiple times
    if _shutting_down:
        return

    _shutting_down = True
    logging.info(f"Running cleanup handlers...")
    
    for handler in cleanup_handlers:
        try:
            handler()
            logging.info(f"Cleanup handler {handler.__name__} completed successfully")
        except Exception as e:
            logging.error(f"Error in cleanup handler {handler.__name__}: {str(e)}", exc_info=True)

def shutdown_server():
    """Properly shutdown the server if it exists."""
    global _server_instance
    
    if _server_instance:
        try:
            logging.info(f"Shutting down KiCad MCP server")
            _server_instance = None
            logging.info(f"KiCad MCP server shutdown complete")
        except Exception as e:
            logging.error(f"Error shutting down server: {str(e)}", exc_info=True)


def register_signal_handlers(server: FastMCP) -> None:
    """Register handlers for system signals to ensure clean shutdown.
    
    Args:
        server: The FastMCP server instance
    """
    def handle_exit_signal(signum, frame):
        logging.info(f"Received signal {signum}, initiating shutdown...")
        
        # Run cleanup first
        run_cleanup_handlers()
        
        # Then shutdown server
        shutdown_server()
        
        # Exit without waiting for stdio processes which might be blocking
        os._exit(0)
    
    # Register for common termination signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_exit_signal)
            logging.info(f"Registered handler for signal {sig}")
        except (ValueError, AttributeError) as e:
            # Some signals may not be available on all platforms
            logging.error(f"Could not register handler for signal {sig}: {str(e)}")


def create_server() -> FastMCP:
    """Create and configure the KiCad MCP server."""
    logging.info(f"Initializing KiCad MCP server")

    # Try to set up KiCad Python path - Removed
    # kicad_modules_available = setup_kicad_python_path()
    kicad_modules_available = False # Set to False as we removed the setup logic

    # if kicad_modules_available:
    #     print("KiCad Python modules successfully configured")
    # else:
    # Always print this now, as we rely on CLI
    logging.info(f"KiCad Python module setup removed; relying on kicad-cli for external operations.")

    # Build a lifespan callable with the kwarg baked in (FastMCP 2.x dropped lifespan_kwargs)
    lifespan_factory = functools.partial(kicad_lifespan, kicad_modules_available=kicad_modules_available)

    # Initialize FastMCP server
    mcp = FastMCP("KiCad", lifespan=lifespan_factory)
    logging.info(f"Created FastMCP server instance with lifespan management")
    
    # Register resources
    logging.info(f"Registering resources...")
    register_project_resources(mcp)
    register_file_resources(mcp)
    register_drc_resources(mcp)
    register_bom_resources(mcp)
    register_netlist_resources(mcp)
    register_pattern_resources(mcp)
    
    # Register tools
    logging.info(f"Registering tools...")
    register_project_tools(mcp)
    register_analysis_tools(mcp)
    register_export_tools(mcp)
    register_drc_tools(mcp)
    register_bom_tools(mcp)
    register_netlist_tools(mcp)
    register_pattern_tools(mcp)
    
    # Register prompts
    logging.info(f"Registering prompts...")
    register_prompts(mcp)
    register_drc_prompts(mcp)
    register_bom_prompts(mcp)
    register_pattern_prompts(mcp)

    # Register signal handlers and cleanup
    register_signal_handlers(mcp)
    atexit.register(run_cleanup_handlers)
    
    # Add specific cleanup handlers
    add_cleanup_handler(lambda: logging.info(f"KiCad MCP server shutdown complete"))

    # Add temp directory cleanup
    def cleanup_temp_dirs():
        """Clean up any temporary directories created by the server."""
        import shutil
        from kicad_mcp.utils.temp_dir_manager import get_temp_dirs
        
        temp_dirs = get_temp_dirs()
        logging.info(f"Cleaning up {len(temp_dirs)} temporary directories")
        
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logging.info(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")
    
    add_cleanup_handler(cleanup_temp_dirs)
    
    logging.info(f"Server initialization complete")
    return mcp


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    # Signal handlers are set up in register_signal_handlers
    pass


def cleanup_handler() -> None:
    """Handle cleanup during shutdown."""
    run_cleanup_handlers()


def setup_logging() -> None:
    """Configure logging for the server."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main() -> None:
    """Start the KiCad MCP server (blocking)."""
    setup_logging()
    logging.info("Starting KiCad MCP server...")
    
    server = create_server()
    
    try:
        server.run()  # FastMCP manages its own event loop
    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        logging.info("Server shutdown complete")


if __name__ == "__main__":
    main()
