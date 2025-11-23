#!/usr/bin/env python3
"""Extract a sample footprint from PCB to understand format"""
import re
from pathlib import Path

pcb_file = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005" / "Astro-DB_rev00005.kicad_pcb"

with open(pcb_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find Q200 footprint using balanced parentheses
pattern = r'\(footprint\s+"[^"]*"'
for match in re.finditer(pattern, content):
    start_pos = match.start()

    # Extract balanced S-expression
    depth = 0
    end_pos = start_pos
    for i in range(start_pos, len(content)):
        if content[i] == '(':
            depth += 1
        elif content[i] == ')':
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break

    footprint_block = content[start_pos:end_pos]

    # Check if this is Q200
    if '(property "Reference" "Q200"' in footprint_block:
        print("=" * 80)
        print("Q200 FOOTPRINT (first 3000 characters):")
        print("=" * 80)
        print(footprint_block[:3000])

        print("\n" + "=" * 80)
        print("ALL PAD DEFINITIONS IN Q200:")
        print("=" * 80)

        # Extract all pad definitions
        pad_pattern = r'\(pad\s+"([^"]+)"[^)]*(?:\([^)]*\))*[^)]*\)'
        for pad_match in re.finditer(r'\(pad\s+"[^"]+', footprint_block):
            # Find the balanced pad S-expression
            pad_start = pad_match.start()
            pad_depth = 0
            pad_end = pad_start
            for i in range(pad_start, end_pos):
                if footprint_block[i - start_pos] == '(':
                    pad_depth += 1
                elif footprint_block[i - start_pos] == ')':
                    pad_depth -= 1
                    if pad_depth == 0:
                        pad_end = i + 1
                        break

            pad_block = footprint_block[pad_start - start_pos:pad_end - start_pos]
            print(f"\nPad block ({len(pad_block)} chars):")
            print(pad_block[:500])
            print("...")

        break
