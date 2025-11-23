#!/usr/bin/env python3
"""Find all pad definitions in Q200 footprint"""
import re
from pathlib import Path

pcb_file = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005" / "Astro-DB_rev00005.kicad_pcb"

with open(pcb_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find Q200 footprint
start_idx = content.find('(property "Reference" "Q200"')
if start_idx == -1:
    print("Q200 not found")
    exit(1)

# Go back to find the start of the footprint
footprint_start = content.rfind('(footprint', 0, start_idx)

# Find the end (balanced parentheses)
depth = 0
footprint_end = footprint_start
for i in range(footprint_start, len(content)):
    if content[i] == '(':
        depth += 1
    elif content[i] == ')':
        depth -= 1
        if depth == 0:
            footprint_end = i + 1
            break

footprint_block = content[footprint_start:footprint_end]

print("Q200 Footprint length:", len(footprint_block))
print("\n" + "=" * 80)
print("Searching for (pad definitions:")
print("=" * 80)

# Find all pad starts
for match in re.finditer(r'\(pad\s+"([^"]+)"', footprint_block):
    pad_num = match.group(1)
    pad_start = match.start()

    # Extract just 200 chars after pad start
    snippet = footprint_block[pad_start:pad_start + 300]

    # Try to find net in this snippet
    net_match = re.search(r'\(net\s+(\d+)\s+"([^"]*)"\)', snippet)

    if net_match:
        net_id = net_match.group(1)
        net_name = net_match.group(2)
        print(f"\nPad {pad_num}: net {net_id} = '{net_name}'")
    else:
        print(f"\nPad {pad_num}: NO NET FOUND")
        print(f"Snippet: {snippet[:150]}")
