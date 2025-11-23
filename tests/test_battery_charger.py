#!/usr/bin/env python3
"""
Focused test of battery_charger page DSL
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kicad_mcp.tools.schematic_dsl_tools import (
    get_schematic_index,
    get_schematic_page
)

async def test_battery_charger():
    project_path = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005"

    print("=" * 70)
    print("BATTERY CHARGER DSL TEST")
    print("=" * 70)
    print()

    # Get index first
    print("Getting schematic index...")
    index = await get_schematic_index(str(project_path))

    # Save index to file
    index_file = Path(__file__).parent / "schematic_index.txt"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index)
    print(f"Index saved to: {index_file}")
    print()
    print("=" * 70)
    print()

    # Get battery charger page
    print("Getting battery_charger page DSL...")
    print()
    page_dsl = await get_schematic_page(str(project_path), "battery_charger")

    # Save to file for inspection
    output_file = Path(__file__).parent / "battery_charger_dsl.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(page_dsl)

    print(f"DSL saved to: {output_file}")
    print(f"DSL length: {len(page_dsl)} characters")
    print()
    print("First 2000 characters (ASCII safe):")
    print("=" * 70)
    # Print ASCII-safe version
    ascii_safe = page_dsl[:2000].encode('ascii', 'replace').decode('ascii')
    print(ascii_safe)

if __name__ == "__main__":
    asyncio.run(test_battery_charger())
