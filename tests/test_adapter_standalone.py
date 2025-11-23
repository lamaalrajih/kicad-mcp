#!/usr/bin/env python3
"""
Standalone test of KiCAD adapter - imports modules directly to avoid package dependencies
"""
import sys
from pathlib import Path

# Add paths for direct module imports (bypassing __init__.py)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "kicad_mcp"))
sys.path.insert(0, str(project_root / "kicad_mcp" / "schematic_core"))
sys.path.insert(0, str(project_root / "kicad_mcp" / "schematic_core" / "adapters"))

# Direct imports of just the modules we need
import importlib.util

# Load adapter module directly
adapter_path = project_root / "kicad_mcp" / "schematic_core" / "adapters" / "kicad_sch.py"
spec = importlib.util.spec_from_file_location("kicad_sch", adapter_path)
kicad_sch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kicad_sch_module)
KiCADSchematicAdapter = kicad_sch_module.KiCADSchematicAdapter

# Load librarian module directly
librarian_path = project_root / "kicad_mcp" / "schematic_core" / "librarian.py"
spec = importlib.util.spec_from_file_location("librarian", librarian_path)
librarian_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(librarian_module)
Librarian = librarian_module.Librarian

def test_kicad_adapter():
    """Test the KiCAD adapter and librarian"""

    project_path = project_root.parent / "kicad-astro-daughterboard2" / "Astro-DB_rev00005"

    print("=" * 70)
    print("TESTING KICAD SCHEMATIC DSL - STANDALONE TEST")
    print("=" * 70)
    print()
    print(f"Project path: {project_path}")
    print()

    # Test 1: Create adapter
    print("[TEST 1] Creating KiCAD adapter...")
    print("-" * 70)
    try:
        adapter = KiCADSchematicAdapter(str(project_path))
        print("✓ Adapter created successfully")
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Fetch raw data
    print("[TEST 2] Fetching raw schematic data...")
    print("-" * 70)
    try:
        adapter.fetch_raw_data()
        print(f"✓ Parsed {len(adapter._parsed_sheets)} schematic files:")
        for sheet_name in adapter._parsed_sheets.keys():
            print(f"  - {sheet_name}")
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Get components
    print("[TEST 3] Extracting components...")
    print("-" * 70)
    try:
        components = adapter.get_components()
        print(f"✓ Extracted {len(components)} components")

        # Show first 5 components
        print("\nFirst 5 components:")
        for comp in components[:5]:
            print(f"  - {comp.refdes}: {comp.value} ({comp.footprint})")
            print(f"    Page: {comp.page}, Pins: {len(comp.pins)}")
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Get nets
    print("[TEST 4] Extracting nets...")
    print("-" * 70)
    try:
        nets = adapter.get_nets()
        print(f"✓ Extracted {len(nets)} nets")

        # Show first 5 nets
        print("\nFirst 5 nets:")
        for net in nets[:5]:
            print(f"  - {net.name}: {len(net.members)} connections on {len(net.pages)} page(s)")
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 5: Create librarian and get index
    print("[TEST 5] Creating librarian and getting index...")
    print("-" * 70)
    try:
        librarian = Librarian(adapter)
        index = librarian.get_index()

        print("✓ Index generated successfully")
        print("\nIndex (first 1000 chars):")
        print(index[:1000])
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 6: Get battery_charger page
    print("[TEST 6] Getting battery_charger page DSL...")
    print("-" * 70)
    try:
        page_dsl = librarian.get_page("battery_charger")

        print("✓ Page DSL generated successfully")
        print(f"\nPage DSL length: {len(page_dsl)} chars")
        print("\nPage DSL (first 1500 chars):")
        print(page_dsl[:1500])
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 7: Get context for a component
    print("[TEST 7] Getting context for component Q1...")
    print("-" * 70)
    try:
        context = librarian.get_context(["Q1"])

        print("✓ Context generated successfully")
        print(f"\nContext length: {len(context)} chars")
        print("\nContext (first 1000 chars):")
        print(context[:1000])
        print()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)

if __name__ == "__main__":
    test_kicad_adapter()
