#!/usr/bin/env python3
"""
Test KiCAD schematic DSL integration
"""
import asyncio
import sys
from pathlib import Path

# Add kicad_mcp to path
sys.path.insert(0, str(Path(__file__).parent))

from kicad_mcp.tools.schematic_dsl_tools import (
    get_schematic_index,
    get_schematic_page
)

async def test_kicad_dsl():
    """Test the KiCAD DSL tools on Rev0005 battery charger"""

    project_path = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005"

    print("=" * 70)
    print("TESTING KICAD SCHEMATIC DSL INTEGRATION")
    print("=" * 70)
    print()

    # Test 1: Get index
    print("[TEST 1] Getting schematic index...")
    print("-" * 70)
    try:
        index = await get_schematic_index(str(project_path))
        print(index[:1000])  # Show first 1000 chars
        print()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 2: Get battery charger page
    print("[TEST 2] Getting battery_charger page DSL...")
    print("-" * 70)
    try:
        page_dsl = await get_schematic_page(str(project_path), "battery_charger")
        print(page_dsl[:1500])  # Show first 1500 chars
        print()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_kicad_dsl())
