"""
Simplified Netlist extraction tools for KiCad schematics.
"""

import os
import logging
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils.netlist_parser import extract_netlist, analyze_netlist


def register_netlist_tools(mcp: FastMCP) -> None:
    """Register netlist-related tools with the MCP server.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.tool()
    def extract_schematic_netlist(schematic_path: str) -> Dict[str, Any]:
        """Extract netlist information from a KiCad schematic.

        This tool parses a KiCad schematic file and extracts comprehensive
        netlist information including components, connections, and labels.

        Args:
            schematic_path: Path to the KiCad schematic file (.kicad_sch)

        Returns:
            Dictionary with netlist information
        """
        logging.info(f"Extracting netlist from schematic: {schematic_path}")

        if not os.path.exists(schematic_path):
            error_msg = f"Schematic file not found: {schematic_path}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}

        logging.info(f"Loading schematic file: {os.path.basename(schematic_path)}")

        try:
            logging.info("Parsing schematic structure...")
            netlist_data = extract_netlist(schematic_path)

            if "error" in netlist_data:
                error_msg = f"Error extracting netlist: {netlist_data['error']}"
                logging.error(error_msg)
                return {"success": False, "error": netlist_data["error"]}

            logging.info(
                f"Extracted {netlist_data['component_count']} components and {netlist_data['net_count']} nets"
            )

            logging.info("Analyzing netlist data...")
            analysis_results = analyze_netlist(netlist_data)

            result = {
                "success": True,
                "schematic_path": schematic_path,
                "component_count": netlist_data["component_count"],
                "net_count": netlist_data["net_count"],
                "components": netlist_data["components"],
                "nets": netlist_data["nets"],
                "analysis": analysis_results,
            }

            logging.info("Netlist extraction complete")
            return result

        except Exception as e:
            error_msg = f"Error extracting netlist: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def find_component_connections(project_path: str, component_ref: str) -> Dict[str, Any]:
        """Find all connections for a specific component in a KiCad project.

        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            component_ref: Component reference (e.g., "R1", "U3")

        Returns:
            Dictionary with component connection information
        """
        logging.info(f"Finding connections for component {component_ref} in project: {project_path}")

        if not os.path.exists(project_path):
            error_msg = f"Project not found: {project_path}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}

        try:
            files = get_project_files(project_path)

            if "schematic" not in files:
                error_msg = "Schematic file not found in project"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}

            schematic_path = files["schematic"]
            logging.info(f"Found schematic file: {os.path.basename(schematic_path)}")

            # Extract netlist
            netlist_data = extract_netlist(schematic_path)

            if "error" in netlist_data:
                error_msg = f"Failed to extract netlist: {netlist_data['error']}"
                logging.error(error_msg)
                return {"success": False, "error": netlist_data["error"]}

            # Check if component exists
            components = netlist_data.get("components", {})
            if component_ref not in components:
                error_msg = f"Component {component_ref} not found in schematic"
                logging.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "available_components": list(components.keys()),
                }

            # Find connections (simplified version)
            component_info = components[component_ref]
            nets = netlist_data.get("nets", {})
            connections = []
            connected_nets = []

            for net_name, pins in nets.items():
                component_pins = [pin for pin in pins if pin.get("component") == component_ref]
                if component_pins:
                    connected_nets.append(net_name)
                    for pin in component_pins:
                        pin_num = pin.get("pin", "Unknown")
                        connected_components = [
                            {"component": other_pin.get("component"), "pin": other_pin.get("pin", "Unknown")}
                            for other_pin in pins
                            if other_pin.get("component") and other_pin.get("component") != component_ref
                        ]
                        connections.append({
                            "pin": pin_num,
                            "net": net_name,
                            "connected_to": connected_components
                        })

            result = {
                "success": True,
                "project_path": project_path,
                "schematic_path": schematic_path,
                "component": component_ref,
                "component_info": component_info,
                "connections": connections,
                "connected_nets": connected_nets,
                "total_connections": len(connections),
            }

            logging.info(f"Found {len(connections)} connections for component {component_ref}")
            return result

        except Exception as e:
            error_msg = f"Error finding component connections: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": str(e)}