"""
Netlist extraction and analysis tools for KiCad schematics.
"""
import os
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP, Context

from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils.netlist_xml_extractor import export_and_parse_netlist_xml
from kicad_mcp.utils.netlist_parser import analyze_netlist

def register_netlist_tools(mcp: FastMCP) -> None:
    """Register netlist-related tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    
    @mcp.tool()
    async def extract_schematic_netlist(schematic_path: str, ctx: Context | None = None) -> Dict[str, Any]:
        """Extract netlist information from a KiCad schematic.
        
        This tool parses a KiCad schematic file and extracts comprehensive
        netlist information including components, connections, and labels.
        
        Args:
            schematic_path: Path to the KiCad schematic file (.kicad_sch)
            ctx: MCP context for progress reporting
            
        Returns:
            Dictionary with netlist information
        """
        print(f"Extracting netlist from schematic: {schematic_path}")
        
        if not os.path.exists(schematic_path):
            print(f"Schematic file not found: {schematic_path}")
            if ctx:
                ctx.info(f"Schematic file not found: {schematic_path}")
            return {"success": False, "error": f"Schematic file not found: {schematic_path}"}
        
        # Report progress
        if ctx:
            await ctx.report_progress(10, 100)
            ctx.info(f"Loading schematic file: {os.path.basename(schematic_path)}")
        
        # Extract netlist information
        try:
            if ctx:
                await ctx.report_progress(20, 100)
                ctx.info("Parsing schematic structure...")
            
            netlist_data = export_and_parse_netlist_xml(schematic_path)
            
            if "error" in netlist_data:
                print(f"Error extracting netlist: {netlist_data['error']}")
                if ctx:
                    ctx.info(f"Error extracting netlist: {netlist_data['error']}")
                return {"success": False, "error": netlist_data['error']}
            
            if ctx:
                await ctx.report_progress(60, 100)
                ctx.info(f"Extracted {netlist_data['component_count']} components and {netlist_data['net_count']} nets")
            
            # Analyze the netlist
            if ctx:
                await ctx.report_progress(70, 100)
                ctx.info("Analyzing netlist data...")
            
            analysis_results = analyze_netlist(netlist_data)
            
            if ctx:
                await ctx.report_progress(90, 100)
            
            # Build result
            result = {
                "success": True,
                "schematic_path": schematic_path,
                "component_count": netlist_data["component_count"],
                "net_count": netlist_data["net_count"],
                "components": netlist_data["components"],
                "nets": netlist_data["nets"],
                "analysis": analysis_results
            }
            
            # Complete progress
            if ctx:
                await ctx.report_progress(100, 100)
                ctx.info("Netlist extraction complete")
            
            return result
            
        except Exception as e:
            print(f"Error extracting netlist: {str(e)}")
            if ctx:
                ctx.info(f"Error extracting netlist: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def extract_project_netlist(project_path: str, ctx: Context | None = None) -> Dict[str, Any]:
        """Extract netlist from a KiCad project's schematic.
        
        This tool finds the schematic associated with a KiCad project
        and extracts its netlist information.
        
        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            ctx: MCP context for progress reporting
            
        Returns:
            Dictionary with netlist information
        """
        print(f"Extracting netlist for project: {project_path}")
        
        if not os.path.exists(project_path):
            print(f"Project not found: {project_path}")
            if ctx:
                ctx.info(f"Project not found: {project_path}")
            return {"success": False, "error": f"Project not found: {project_path}"}
        
        # Report progress
        if ctx:
            await ctx.report_progress(10, 100)
        
        # Get the schematic file
        try:
            files = get_project_files(project_path)
            
            if "schematic" not in files:
                print("Schematic file not found in project")
                if ctx:
                    ctx.info("Schematic file not found in project")
                return {"success": False, "error": "Schematic file not found in project"}
            
            schematic_path = files["schematic"]
            print(f"Found schematic file: {schematic_path}")
            if ctx:
                ctx.info(f"Found schematic file: {os.path.basename(schematic_path)}")

            # Extract netlist directly using utility functions
            if ctx:
                await ctx.report_progress(20, 100)
                ctx.info("Parsing schematic structure...")

            netlist_data = export_and_parse_netlist_xml(schematic_path)

            if "error" in netlist_data:
                print(f"Error extracting netlist: {netlist_data['error']}")
                if ctx:
                    ctx.info(f"Error extracting netlist: {netlist_data['error']}")
                return {"success": False, "error": netlist_data['error']}

            if ctx:
                await ctx.report_progress(60, 100)
                ctx.info(f"Extracted {netlist_data['component_count']} components and {netlist_data['net_count']} nets")

            # Analyze the netlist
            if ctx:
                await ctx.report_progress(70, 100)
                ctx.info("Analyzing netlist data...")

            analysis_results = analyze_netlist(netlist_data)

            if ctx:
                await ctx.report_progress(90, 100)

            # Build result
            result = {
                "success": True,
                "project_path": project_path,
                "schematic_path": schematic_path,
                "component_count": netlist_data["component_count"],
                "net_count": netlist_data["net_count"],
                "components": netlist_data["components"],
                "nets": netlist_data["nets"],
                "analysis": analysis_results
            }

            # Complete progress
            if ctx:
                await ctx.report_progress(100, 100)
                ctx.info("Netlist extraction complete")

            return result
            
        except Exception as e:
            print(f"Error extracting project netlist: {str(e)}")
            if ctx:
                ctx.info(f"Error extracting project netlist: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def analyze_schematic_connections(schematic_path: str, ctx: Context | None = None) -> Dict[str, Any]:
        """Analyze connections in a KiCad schematic.
        
        This tool provides detailed analysis of component connections,
        including power nets, signal paths, and potential issues.
        
        Args:
            schematic_path: Path to the KiCad schematic file (.kicad_sch)
            ctx: MCP context for progress reporting
            
        Returns:
            Dictionary with connection analysis
        """
        print(f"Analyzing connections in schematic: {schematic_path}")
        
        if not os.path.exists(schematic_path):
            print(f"Schematic file not found: {schematic_path}")
            if ctx:
                ctx.info(f"Schematic file not found: {schematic_path}")
            return {"success": False, "error": f"Schematic file not found: {schematic_path}"}
        
        # Report progress
        if ctx:
            await ctx.report_progress(10, 100)
            ctx.info(f"Extracting netlist from: {os.path.basename(schematic_path)}")
        
        # Extract netlist information
        try:
            netlist_data = export_and_parse_netlist_xml(schematic_path)
            
            if "error" in netlist_data:
                print(f"Error extracting netlist: {netlist_data['error']}")
                if ctx:
                    ctx.info(f"Error extracting netlist: {netlist_data['error']}")
                return {"success": False, "error": netlist_data['error']}
            
            if ctx:
                await ctx.report_progress(40, 100)
            
            # Advanced connection analysis
            if ctx:
                ctx.info("Performing connection analysis...")
            
            analysis = {
                "component_count": netlist_data["component_count"],
                "net_count": netlist_data["net_count"],
                "component_types": {},
                "power_nets": [],
                "signal_nets": [],
                "potential_issues": []
            }
            
            # Analyze component types
            components = netlist_data.get("components", {})
            for ref, component in components.items():
                # Extract component type from reference (e.g., R1 -> R)
                import re
                comp_type_match = re.match(r'^([A-Za-z_]+)', ref)
                if comp_type_match:
                    comp_type = comp_type_match.group(1)
                    if comp_type not in analysis["component_types"]:
                        analysis["component_types"][comp_type] = 0
                    analysis["component_types"][comp_type] += 1
            
            if ctx:
                await ctx.report_progress(60, 100)
            
            # Identify power nets
            nets = netlist_data.get("nets", {})
            for net_name, pins in nets.items():
                if any(net_name.startswith(prefix) for prefix in ["VCC", "VDD", "GND", "+5V", "+3V3", "+12V"]):
                    analysis["power_nets"].append({
                        "name": net_name,
                        "pin_count": len(pins)
                    })
                else:
                    analysis["signal_nets"].append({
                        "name": net_name,
                        "pin_count": len(pins)
                    })
            
            if ctx:
                await ctx.report_progress(80, 100)
            
            # Check for potential issues
            # 1. Nets with only one connection (floating)
            for net_name, pins in nets.items():
                if len(pins) <= 1 and not any(net_name.startswith(prefix) for prefix in ["VCC", "VDD", "GND", "+5V", "+3V3", "+12V"]):
                    analysis["potential_issues"].append({
                        "type": "floating_net",
                        "net": net_name,
                        "description": f"Net '{net_name}' appears to be floating (only has {len(pins)} connection)"
                    })
            
            # 2. Power pins without connections
            # This would require more detailed parsing of the schematic
            
            if ctx:
                await ctx.report_progress(90, 100)
            
            # Build result
            result = {
                "success": True,
                "schematic_path": schematic_path,
                "analysis": analysis
            }
            
            # Complete progress
            if ctx:
                await ctx.report_progress(100, 100)
                ctx.info("Connection analysis complete")
            
            return result
            
        except Exception as e:
            print(f"Error analyzing connections: {str(e)}")
            if ctx:
                ctx.info(f"Error analyzing connections: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def find_component_connections(project_path: str, component_ref: str, ctx: Context | None = None) -> Dict[str, Any]:
        """Find all connections for a specific component in a KiCad project.
        
        This tool extracts information about how a specific component
        is connected to other components in the schematic.
        
        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            component_ref: Component reference (e.g., "R1", "U3")
            ctx: MCP context for progress reporting
            
        Returns:
            Dictionary with component connection information
        """
        print(f"Finding connections for component {component_ref} in project: {project_path}")
        
        if not os.path.exists(project_path):
            print(f"Project not found: {project_path}")
            if ctx:
                ctx.info(f"Project not found: {project_path}")
            return {"success": False, "error": f"Project not found: {project_path}"}
        
        # Report progress
        if ctx:
            await ctx.report_progress(10, 100)
        
        # Get the schematic file
        try:
            files = get_project_files(project_path)
            
            if "schematic" not in files:
                print("Schematic file not found in project")
                if ctx:
                    ctx.info("Schematic file not found in project")
                return {"success": False, "error": "Schematic file not found in project"}
            
            schematic_path = files["schematic"]
            print(f"Found schematic file: {schematic_path}")
            if ctx:
                ctx.info(f"Found schematic file: {os.path.basename(schematic_path)}")
            
            # Extract netlist
            if ctx:
                await ctx.report_progress(30, 100)
                ctx.info(f"Extracting netlist to find connections for {component_ref}...")
            
            netlist_data = export_and_parse_netlist_xml(schematic_path)
            
            if "error" in netlist_data:
                print(f"Failed to extract netlist: {netlist_data['error']}")
                if ctx:
                    ctx.info(f"Failed to extract netlist: {netlist_data['error']}")
                return {"success": False, "error": netlist_data['error']}
            
            # Check if component exists in the netlist
            components = netlist_data.get("components", {})
            if component_ref not in components:
                print(f"Component {component_ref} not found in schematic")
                if ctx:
                    ctx.info(f"Component {component_ref} not found in schematic")
                return {
                    "success": False, 
                    "error": f"Component {component_ref} not found in schematic",
                    "available_components": list(components.keys())
                }
            
            # Get component information
            component_info = components[component_ref]
            
            # Find connections
            if ctx:
                await ctx.report_progress(50, 100)
                ctx.info("Finding connections...")
            
            nets = netlist_data.get("nets", {})
            connections = []
            connected_nets = []
            
            for net_name, pins in nets.items():
                # Check if any pin belongs to our component
                component_pins = []
                for pin in pins:
                    if pin.get('component') == component_ref:
                        component_pins.append(pin)
                        
                if component_pins:
                    # This net has connections to our component
                    net_connections = []
                    
                    for pin in component_pins:
                        pin_num = pin.get('pin', 'Unknown')
                        # Find other components connected to this pin
                        connected_components = []
                        
                        for other_pin in pins:
                            other_comp = other_pin.get('component')
                            if other_comp and other_comp != component_ref:
                                connected_components.append({
                                    "component": other_comp,
                                    "pin": other_pin.get('pin', 'Unknown')
                                })
                        
                        net_connections.append({
                            "pin": pin_num,
                            "net": net_name,
                            "connected_to": connected_components
                        })
                    
                    connections.extend(net_connections)
                    connected_nets.append(net_name)
            
            # Analyze the connections
            if ctx:
                await ctx.report_progress(70, 100)
                ctx.info("Analyzing connections...")
            
            # Categorize connections by pin function (if possible)
            pin_functions = {}
            if "pins" in component_info:
                for pin in component_info["pins"]:
                    pin_num = pin.get('num')
                    pin_name = pin.get('name', '')
                    
                    # Try to categorize based on pin name
                    pin_type = "unknown"
                    
                    if any(power_term in pin_name.upper() for power_term in ["VCC", "VDD", "VEE", "VSS", "GND", "PWR", "POWER"]):
                        pin_type = "power"
                    elif any(io_term in pin_name.upper() for io_term in ["IO", "I/O", "GPIO"]):
                        pin_type = "io"
                    elif any(input_term in pin_name.upper() for input_term in ["IN", "INPUT"]):
                        pin_type = "input"
                    elif any(output_term in pin_name.upper() for output_term in ["OUT", "OUTPUT"]):
                        pin_type = "output"
                    
                    pin_functions[pin_num] = {
                        "name": pin_name,
                        "type": pin_type
                    }
            
            # Build result
            result = {
                "success": True,
                "project_path": project_path,
                "schematic_path": schematic_path,
                "component": component_ref,
                "component_info": component_info,
                "connections": connections,
                "connected_nets": connected_nets,
                "pin_functions": pin_functions,
                "total_connections": len(connections)
            }
            
            if ctx:
                await ctx.report_progress(100, 100)
                ctx.info(f"Found {len(connections)} connections for component {component_ref}")
            
            return result
            
        except Exception as e:
            print(f"Error finding component connections: {str(e)}", exc_info=True)
            if ctx:
                ctx.info(f"Error finding component connections: {str(e)}")
            return {"success": False, "error": str(e)}
