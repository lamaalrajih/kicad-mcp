#!/usr/bin/env python3
"""Debug PCB parser to see what pads are extracted"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kicad_mcp.utils.pcb_netlist_parser import PCBNetlistParser

# Parse the PCB file
project_path = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005"
pcb_files = list(project_path.glob("*.kicad_pcb"))

if not pcb_files:
    print(f"No PCB file found in {project_path}")
    sys.exit(1)

print(f"Parsing: {pcb_files[0]}")
parser = PCBNetlistParser(str(pcb_files[0]))
component_pads = parser.parse()

# Check specific transistors
test_components = ["Q200", "Q201", "Q203", "R213", "C218"]

print("\n" + "=" * 80)
print("PAD EXTRACTION DEBUG")
print("=" * 80)

for comp_ref in test_components:
    if comp_ref in component_pads:
        pads = component_pads[comp_ref]
        print(f"\n{comp_ref}: {len(pads)} pads")
        for pad_num, net_name in sorted(pads.items()):
            print(f"  {pad_num}: {net_name}")
    else:
        print(f"\n{comp_ref}: NOT FOUND")

# Statistics
print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print(f"Total components: {len(component_pads)}")

# Count pads per component
pad_counts = {}
for refdes, pads in component_pads.items():
    count = len(pads)
    pad_counts[count] = pad_counts.get(count, 0) + 1

print(f"\nComponents by pad count:")
for count in sorted(pad_counts.keys()):
    print(f"  {count} pad(s): {pad_counts[count]} components")

# Show components with only 1 pad (suspicious for transistors)
single_pad_comps = [ref for ref, pads in component_pads.items() if len(pads) == 1 and ref.startswith('Q')]
if single_pad_comps:
    print(f"\n[WARNING] Transistors with only 1 pad (SUSPICIOUS):")
    for ref in sorted(single_pad_comps)[:10]:  # First 10
        pad_num, net = list(component_pads[ref].items())[0]
        print(f"  {ref}: pad {pad_num} -> {net}")
