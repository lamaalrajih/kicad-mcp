"""Test the full netlist XML extractor."""
import sys
import os
import json
import importlib.util

# Load module directly without going through __init__.py
spec = importlib.util.spec_from_file_location(
    "netlist_xml_extractor",
    os.path.join(os.path.dirname(__file__), "kicad_mcp", "utils", "netlist_xml_extractor.py")
)
netlist_module = importlib.util.module_from_spec(spec)

# We need to handle the relative imports in netlist_xml_extractor
# Add the utils directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kicad_mcp", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kicad_mcp"))

spec.loader.exec_module(netlist_module)
export_and_parse_netlist_xml = netlist_module.export_and_parse_netlist_xml

schematic_path = r"c:\Users\geoff\Desktop\projects\kicad-astro-daughterboard2\Astro-DB_rev00005\Astro-DB_rev00005.kicad_sch"

print("Running export_and_parse_netlist_xml...")
print(f"Schematic: {schematic_path}")

try:
    result = export_and_parse_netlist_xml(schematic_path)

    print(f"\nResult keys: {list(result.keys())}")
    print(f"Success: {result.get('success')}")
    print(f"Components: {result.get('component_count')}")
    print(f"Nets: {result.get('net_count')}")

    # Show first few components
    print("\nFirst 3 components:")
    for i, (ref, data) in enumerate(list(result['components'].items())[:3]):
        print(f"  {ref}: {data.get('value')} - {len(data.get('pins', {}))} pins")

    print("\nSUCCESS!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
