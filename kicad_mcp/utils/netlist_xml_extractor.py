"""
XML netlist extraction using kicad-cli and kicad_netlist_reader.

This module exports netlists from KiCad schematics using the kicad-cli tool
and parses them using the official kicad_netlist_reader module.
"""
import os
import subprocess
import tempfile
from typing import Dict, Any
from pathlib import Path

from .kicad_cli import find_kicad_cli
from .kicad_netlist_reader import netlist as KicadNetlist


def export_and_parse_netlist_xml(schematic_path: str) -> Dict[str, Any]:
    """Export netlist as XML using kicad-cli and parse it.

    Args:
        schematic_path: Path to the .kicad_sch file

    Returns:
        Dictionary with parsed netlist information including pin-to-net mappings
    """
    if not os.path.exists(schematic_path):
        raise FileNotFoundError(f"Schematic file not found: {schematic_path}")

    # Find kicad-cli
    kicad_cli = find_kicad_cli()
    if not kicad_cli:
        raise RuntimeError("kicad-cli not found. Please ensure KiCad 8.0+ is installed.")

    # Create temporary file for netlist XML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
        netlist_xml_path = tmp_file.name

    try:
        # Export netlist as XML using kicad-cli
        cmd = [
            kicad_cli,
            "sch",
            "export",
            "netlist",
            "--format", "kicadxml",
            "--output", netlist_xml_path,
            schematic_path
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"kicad-cli failed: {result.stderr}")

        if not os.path.exists(netlist_xml_path):
            raise RuntimeError("Netlist XML file was not created")

        # Parse the XML netlist using kicad_netlist_reader
        print("Parsing XML netlist...")
        net = KicadNetlist(netlist_xml_path)
        print(f"Loaded netlist with {len(net.components)} components")

        # Extract components with pin-to-net mappings
        components = {}
        print("Extracting component pin-to-net mappings...")
        for i, comp in enumerate(net.components):
            if i % 50 == 0:
                print(f"  Processing component {i}/{len(net.components)}")
            ref = comp.getRef()
            value = comp.getValue()
            footprint = comp.getFootprint()

            # Get pin-to-net mappings
            pins = {}
            libpart = comp.getLibPart()
            if libpart:
                pin_list = libpart.getPinList()
                for pin in pin_list:
                    pin_num = pin.get("num")
                    if pin_num:
                        # Get the net name for this pin
                        net_name = comp.getPinNetname(pin_num, net, False)
                        pins[pin_num] = {
                            "net": net_name,
                            "name": pin.get("name", ""),
                            "type": pin.get("type", "")
                        }

            components[ref] = {
                "reference": ref,
                "value": value,
                "footprint": footprint,
                "pins": pins
            }

        # Extract nets
        print(f"Extracted {len(components)} components")
        print("Extracting nets...")
        nets = {}
        for net_element in net.getNets():
            net_name = net_element.get("net", "name")
            if net_name:
                # Get all nodes (pins) connected to this net
                pins = []
                for node in net_element.children:
                    component_ref = node.get("node", "ref")
                    pin_num = node.get("node", "pin")
                    if component_ref and pin_num:
                        pins.append({
                            "component": component_ref,
                            "pin": pin_num
                        })
                nets[net_name] = pins

        print(f"Extracted {len(nets)} nets")
        print("Building result dictionary...")

        return {
            "success": True,
            "schematic_path": schematic_path,
            "components": components,
            "nets": nets,
            "component_count": len(components),
            "net_count": len(nets)
        }

    finally:
        # Clean up temporary netlist file
        if os.path.exists(netlist_xml_path):
            try:
                os.unlink(netlist_xml_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary netlist file: {e}")
