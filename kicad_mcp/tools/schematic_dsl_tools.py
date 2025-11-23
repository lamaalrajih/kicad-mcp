"""
Schematic DSL Analysis Tools

Provides high-level schematic analysis using the schematic_core library.
These tools generate LLM-optimized DSL representations of KiCAD schematics.
"""
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import Context

from ..schematic_core.adapters.kicad_sch import KiCADSchematicAdapter
from ..schematic_core.librarian import Librarian


async def get_schematic_index(
    project_path: str,
    ctx: Context | None = None
) -> str:
    """
    Get a high-level index/overview of all schematic pages

    This tool provides a bird's-eye view of the entire schematic design:
    - List of all schematic pages with component/net counts
    - Power rails and their distribution
    - Inter-page signals and connectivity
    - Design hierarchy

    Use this first to understand the overall structure before diving into specific pages.

    Args:
        project_path: Path to directory containing .kicad_sch files

    Returns:
        Markdown-formatted index of the schematic design

    Example:
        get_schematic_index("/path/to/kicad/project")
    """
    try:
        adapter = KiCADSchematicAdapter(project_path)
        librarian = Librarian(adapter)
        index = librarian.get_index()

        return index

    except Exception as e:
        return f"Error generating schematic index: {str(e)}\n\nPlease check that the project path contains .kicad_sch files."


async def get_schematic_page(
    project_path: str,
    page_name: str,
    ctx: Context | None = None
) -> str:
    """
    Get detailed DSL representation of a specific schematic page

    This tool extracts a single schematic page in a compact, LLM-optimized format:
    - Component definitions with part numbers and values
    - Pin-to-net connectivity
    - Hierarchical structure

    The DSL format is ~10x more compact than raw KiCAD data while preserving
    all essential information for circuit analysis.

    Args:
        project_path: Path to directory containing .kicad_sch files
        page_name: Name of the schematic page (without .kicad_sch extension)

    Returns:
        DSL representation of the schematic page

    Example:
        get_schematic_page("/path/to/project", "battery_charger")
    """
    try:
        adapter = KiCADSchematicAdapter(project_path)
        librarian = Librarian(adapter)
        page_dsl = librarian.get_page(page_name)

        return page_dsl

    except Exception as e:
        return f"Error generating schematic page DSL: {str(e)}\n\nAvailable pages: Use get_schematic_index to see all pages."


async def get_schematic_context(
    project_path: str,
    component_ref: Optional[str] = None,
    net_name: Optional[str] = None,
    ctx: Context | None = None
) -> str:
    """
    Get contextual information about a component or net

    This tool provides focused context for analysis:
    - Component mode: Shows the component's page, connections, and related nets
    - Net mode: Shows all components connected to a net and which pages it spans
    - Useful for tracing signals, debugging connections, or understanding subcircuits

    Args:
        project_path: Path to directory containing .kicad_sch files
        component_ref: Component designator (e.g., "Q1", "U200") - optional
        net_name: Net name to trace (e.g., "GND", "VBUS") - optional

    Returns:
        Contextual information in DSL format

    Example:
        get_schematic_context("/path/to/project", component_ref="U200")
        get_schematic_context("/path/to/project", net_name="VBUS")
    """
    try:
        adapter = KiCADSchematicAdapter(project_path)
        librarian = Librarian(adapter)

        if component_ref:
            context = librarian.get_context([component_ref])
        elif net_name:
            # For net context, find all components connected to the net
            # First refresh to get data
            librarian.refresh()

            # Find the net
            net = librarian.get_net(net_name)
            if not net:
                return f"Error: Net '{net_name}' not found in schematic"

            # Get all component refdes connected to this net
            connected_components = list(set(refdes for refdes, pin in net.members))

            # Generate context for all connected components
            context = librarian.get_context(connected_components)
        else:
            return "Error: Please specify either component_ref or net_name"

        return context

    except Exception as e:
        return f"Error generating schematic context: {str(e)}"


# Register tools with MCP server
def register_schematic_dsl_tools(mcp):
    """Register all schematic DSL tools with the MCP server"""

    mcp.tool()(get_schematic_index)
    mcp.tool()(get_schematic_page)
    mcp.tool()(get_schematic_context)
