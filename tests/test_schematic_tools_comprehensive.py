#!/usr/bin/env python3
"""
Comprehensive test of KiCAD Schematic DSL tools
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kicad_mcp.tools.schematic_dsl_tools import (
    get_schematic_index,
    get_schematic_page,
    get_schematic_context
)

async def test_all_tools():
    project_path = Path(__file__).parent.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005"

    print("=" * 80)
    print("COMPREHENSIVE SCHEMATIC DSL TOOL TEST")
    print("=" * 80)
    print(f"Project: {project_path.name}")
    print()

    # ========================================================================
    # TEST 1: Index - Get overview of entire schematic
    # ========================================================================
    print("[TEST 1] get_schematic_index()")
    print("-" * 80)
    try:
        index = await get_schematic_index(str(project_path))

        # Save to file
        with open("test_output_index.txt", 'w', encoding='utf-8') as f:
            f.write(index)

        print("[OK] Index generated successfully")
        print(f"  Saved to: test_output_index.txt")

        # Parse and display summary
        lines = index.split('\n')
        pages = [l for l in lines if l.startswith('- ') and '(' in l]
        signals = [l for l in lines if l.startswith('- ') and ':' in l and '(' not in l]

        print(f"  Pages found: {len(pages)}")
        print(f"  Inter-page signals: {len(signals)}")
        print()

    except Exception as e:
        print(f"[ERR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

    # ========================================================================
    # TEST 2: Get specific page - battery_charger
    # ========================================================================
    print("[TEST 2] get_schematic_page('battery_charger')")
    print("-" * 80)
    try:
        page_dsl = await get_schematic_page(str(project_path), "battery_charger")

        # Save to file
        with open("test_output_battery_charger.txt", 'w', encoding='utf-8') as f:
            f.write(page_dsl)

        print("[OK] Battery charger page DSL generated")
        print(f"  Saved to: test_output_battery_charger.txt")
        print(f"  Size: {len(page_dsl)} characters, {len(page_dsl.split(chr(10)))} lines")

        # Count components and nets
        lines = page_dsl.split('\n')
        components = [l for l in lines if l.startswith('COMP ')]
        nets = [l for l in lines if l.startswith('NET ')]

        print(f"  Components: {len(components)}")
        print(f"  Nets: {len(nets)}")

        # Show first few transistors (Q components)
        transistors = [l for l in components if ' Q' in l and '(' in l][:5]
        if transistors:
            print(f"\n  First 5 transistors:")
            for t in transistors:
                # Extract Q number and part number
                parts = t.split('(')
                if len(parts) >= 2:
                    q_num = parts[0].split()[-1]
                    mpn = parts[1].rstrip(')')
                    print(f"    {q_num}: {mpn}")
        print()

    except Exception as e:
        print(f"[ERR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

    # ========================================================================
    # TEST 3: Get component context - Q1 (if exists)
    # ========================================================================
    print("[TEST 3] get_schematic_context(component_ref='Q1')")
    print("-" * 80)
    try:
        context = await get_schematic_context(str(project_path), component_ref="Q1")

        # Save to file
        with open("test_output_Q1_context.txt", 'w', encoding='utf-8') as f:
            f.write(context)

        print("[OK] Q1 context generated")
        print(f"  Saved to: test_output_Q1_context.txt")
        print(f"  Size: {len(context)} characters")

        # Show first 500 chars
        print(f"\n  First 500 chars:")
        print("  " + "-" * 76)
        preview = context[:500].replace('\n', '\n  ')
        print(f"  {preview}")
        print()

    except Exception as e:
        print(f"[ERR] ERROR: {e}")
        print()

    # ========================================================================
    # TEST 4: Get net context - GND
    # ========================================================================
    print("[TEST 4] get_schematic_context(net_name='GND')")
    print("-" * 80)
    try:
        context = await get_schematic_context(str(project_path), net_name="GND")

        # Save to file
        with open("test_output_GND_context.txt", 'w', encoding='utf-8') as f:
            f.write(context)

        print("[OK] GND net context generated")
        print(f"  Saved to: test_output_GND_context.txt")
        print(f"  Size: {len(context)} characters")

        # Count connections
        lines = context.split('\n')
        connections = [l for l in lines if 'CON:' in l or '.1' in l or '.2' in l]

        print(f"  Connection lines: {len(connections)}")
        print(f"\n  Sample connections (first 5):")
        for conn in connections[:5]:
            print(f"    {conn.strip()}")
        print()

    except Exception as e:
        print(f"[ERR] ERROR: {e}")
        print()

    # ========================================================================
    # TEST 5: Get different page - Power_Supplies
    # ========================================================================
    print("[TEST 5] get_schematic_page('Power_Supplies')")
    print("-" * 80)
    try:
        page_dsl = await get_schematic_page(str(project_path), "Power_Supplies")

        # Save to file
        with open("test_output_power_supplies.txt", 'w', encoding='utf-8') as f:
            f.write(page_dsl)

        print("[OK] Power_Supplies page DSL generated")
        print(f"  Saved to: test_output_power_supplies.txt")

        # Count components by type
        lines = page_dsl.split('\n')
        components = [l for l in lines if l.startswith('COMP ')]

        # Group by prefix
        prefixes = {}
        for comp in components:
            parts = comp.split()
            if len(parts) >= 2:
                ref = parts[1]
                prefix = ''.join([c for c in ref if c.isalpha()])
                prefixes[prefix] = prefixes.get(prefix, 0) + 1

        print(f"  Total components: {len(components)}")
        print(f"  Components by type:")
        for prefix, count in sorted(prefixes.items()):
            print(f"    {prefix}: {count}")
        print()

    except Exception as e:
        print(f"[ERR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

    # ========================================================================
    # TEST 6: Component context for U200 (battery charger IC)
    # ========================================================================
    print("[TEST 6] get_schematic_context(component_ref='U200') - Battery Charger IC")
    print("-" * 80)
    try:
        context = await get_schematic_context(str(project_path), component_ref="U200")

        # Save to file
        with open("test_output_U200_context.txt", 'w', encoding='utf-8') as f:
            f.write(context)

        print("[OK] U200 (Battery Charger IC) context generated")
        print(f"  Saved to: test_output_U200_context.txt")

        # Extract component definition
        lines = context.split('\n')
        comp_lines = []
        in_comp = False
        for line in lines:
            if line.startswith('COMP U200'):
                in_comp = True
            if in_comp:
                comp_lines.append(line)
                if line.startswith('COMP ') and 'U200' not in line:
                    break
                if line.startswith('# NETS'):
                    break

        print(f"\n  Component definition:")
        for line in comp_lines[:15]:  # First 15 lines
            print(f"    {line}")
        print()

    except Exception as e:
        print(f"[ERR] ERROR: {e}")
        print()

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("All test outputs saved to test_output_*.txt files")
    print()
    print("Files created:")
    print("  - test_output_index.txt              (schematic overview)")
    print("  - test_output_battery_charger.txt    (battery charger page)")
    print("  - test_output_Q1_context.txt         (Q1 transistor context)")
    print("  - test_output_GND_context.txt        (GND net context)")
    print("  - test_output_power_supplies.txt     (power supplies page)")
    print("  - test_output_U200_context.txt       (battery charger IC context)")
    print()

if __name__ == "__main__":
    asyncio.run(test_all_tools())
