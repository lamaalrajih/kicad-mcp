"""Test the XML netlist parser."""
import sys
import os

# Minimal imports to avoid dependency issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kicad_mcp', 'utils'))

from kicad_netlist_reader import netlist as KicadNetlist

netlist_xml_path = "test_output.xml"

if not os.path.exists(netlist_xml_path):
    print(f"ERROR: {netlist_xml_path} not found. Run test_kicad_cli_simple.py first")
    exit(1)

print(f"Parsing {netlist_xml_path}...")
print(f"File size: {os.path.getsize(netlist_xml_path)} bytes")

try:
    net = KicadNetlist(netlist_xml_path)
    print("Netlist loaded successfully!")

    print(f"\nExtracting components...")
    component_count = 0
    for comp in net.components:
        component_count += 1
        if component_count <= 3:  # Show first 3 components
            ref = comp.getRef()
            value = comp.getValue()
            print(f"  {ref}: {value}")

    print(f"Total components: {component_count}")

    print(f"\nExtracting nets...")
    net_count = 0
    for net_elem in net.getNets():
        net_count += 1

    print(f"Total nets: {net_count}")
    print("\nSUCCESS: Parser works!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
