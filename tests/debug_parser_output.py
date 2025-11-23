#!/usr/bin/env python3
"""Debug: Check what the SchematicParser actually returns"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from kicad_mcp.utils.netlist_parser import SchematicParser

project_path = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005"
battery_charger_sch = project_path / "battery_charger.kicad_sch"

print(f"Parsing: {battery_charger_sch}")
print()

parser = SchematicParser(str(battery_charger_sch), is_hierarchical=False)
parsed_data = parser.parse()

print("\n" + "=" * 70)
print("PARSED DATA STRUCTURE:")
print("=" * 70)
print(f"Keys: {list(parsed_data.keys())}")
print(f"Component count: {parsed_data['component_count']}")
print(f"Net count: {parsed_data['net_count']}")
print()

# Check first component
components = parsed_data.get("components", {})
if components:
    first_comp_ref = list(components.keys())[0]
    first_comp = components[first_comp_ref]

    print(f"First component: {first_comp_ref}")
    print(f"Component data keys: {list(first_comp.keys())}")
    print()
    print("Component data:")
    for key, value in first_comp.items():
        if key == "pins" and isinstance(value, dict):
            print(f"  {key}: (dict with {len(value)} pins)")
            # Show first pin
            if value:
                first_pin_num = list(value.keys())[0]
                first_pin = value[first_pin_num]
                print(f"    Example pin {first_pin_num}: {first_pin}")
        else:
            print(f"  {key}: {value}")

    print()
    print(f"Total components: {len(components)}")
else:
    print("No components found!")

print()

# Check nets
nets = parsed_data.get("nets", {})
print(f"Nets: {len(nets)} nets found")
if nets:
    for net_name, members in list(nets.items())[:3]:
        print(f"  - {net_name}: {len(members)} connections")
