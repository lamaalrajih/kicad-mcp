"""
Example usage of the Altium JSON Adapter.

This script demonstrates how to use the AltiumJSONAdapter to parse Altium
schematic data and extract components and nets.
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from schematic_core.adapters.altium_json import AltiumJSONAdapter


def main():
    """
    Example: Load and parse Altium sample JSON data.
    """
    # Path to sample JSON file
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "altium_sample.json"
    )

    if not os.path.exists(sample_path):
        print(f"Error: Sample file not found at {sample_path}")
        return

    print("="*60)
    print("Altium JSON Adapter - Example Usage")
    print("="*60)
    print()

    # Read the JSON file
    print("1. Reading JSON file...")
    with open(sample_path, 'r') as f:
        json_data = f.read()

    # Create adapter and fetch data
    print("2. Initializing adapter...")
    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()

    # Extract components
    print("3. Extracting components...")
    components = adapter.get_components()
    print(f"   Found {len(components)} components")
    print()

    # Display component details
    print("Component Details:")
    print("-" * 60)
    for comp in components[:5]:  # Show first 5 components
        print(f"  {comp.refdes}: {comp.value}")
        print(f"    Type: {comp.derived_type()}")
        print(f"    MPN: {comp.mpn}")
        print(f"    Footprint: {comp.footprint}")
        print(f"    Page: {comp.page}")
        print(f"    Pins: {len(comp.pins)}")
        print(f"    Complex: {comp.is_complex()}")
        print(f"    Passive: {comp.is_passive()}")
        print()

    if len(components) > 5:
        print(f"  ... and {len(components) - 5} more components")
        print()

    # Extract nets
    print("4. Building nets...")
    nets = adapter.get_nets()
    print(f"   Found {len(nets)} nets")
    print()

    # Display net details
    print("Net Details:")
    print("-" * 60)

    # Show global nets
    print("Global Nets:")
    global_nets = [n for n in nets if n.is_global()]
    for net in global_nets[:5]:  # Show first 5 global nets
        print(f"  {net.name}:")
        print(f"    Members: {len(net.members)}")
        print(f"    Pages: {', '.join(sorted(net.pages))}")
        print(f"    Inter-page: {net.is_inter_page()}")
        print()

    # Show regular nets
    print("Regular Nets:")
    regular_nets = [n for n in nets if not n.is_global()]
    for net in regular_nets[:5]:  # Show first 5 regular nets
        print(f"  {net.name}:")
        print(f"    Members: {', '.join([f'{r}.{p}' for r, p in net.members[:3]])}")
        if len(net.members) > 3:
            print(f"             ... and {len(net.members) - 3} more")
        print(f"    Pages: {', '.join(sorted(net.pages))}")
        print(f"    Inter-page: {net.is_inter_page()}")
        print()

    # Statistics
    print("="*60)
    print("Statistics:")
    print("-" * 60)
    print(f"  Total components: {len(components)}")
    print(f"  Total nets: {len(nets)}")
    print(f"  Complex components: {len([c for c in components if c.is_complex()])}")
    print(f"  Passive components: {len([c for c in components if c.is_passive()])}")
    print(f"  Global nets: {len(global_nets)}")
    print(f"  Inter-page nets: {len([n for n in nets if n.is_inter_page()])}")

    # Component type breakdown
    print()
    print("Component Type Breakdown:")
    type_counts = {}
    for comp in components:
        comp_type = comp.derived_type()
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

    for comp_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comp_type}: {count}")

    # Page breakdown
    print()
    print("Components per Page:")
    page_counts = {}
    for comp in components:
        page_counts[comp.page] = page_counts.get(comp.page, 0) + 1

    for page, count in sorted(page_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {page}: {count} components")

    print()
    print("="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
