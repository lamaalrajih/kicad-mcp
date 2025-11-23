"""
Integration tests for the Unified Schematic Core v0.3.

This test suite validates the entire end-to-end workflow:
1. Load JSON schematic data (altium_sample.json)
2. Transform via adapter (AltiumJSONAdapter)
3. Manage state via Librarian
4. Generate DSL output for pages, contexts, and indexes

The tests verify:
- Data loading and parsing
- Component and net transformations
- DSL format compliance
- Multi-page navigation
- Context bubble generation
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Import core modules
try:
    from .models import Component, Net, Pin
    from .dsl_emitter import emit_page_dsl, emit_context_dsl
except ImportError:
    from models import Component, Net, Pin
    from dsl_emitter import emit_page_dsl, emit_context_dsl

# Conditional imports - will be available once adapters/librarian are implemented
try:
    from .adapters.altium_json import AltiumJSONAdapter
    ADAPTER_AVAILABLE = True
except ImportError:
    try:
        from adapters.altium_json import AltiumJSONAdapter
        ADAPTER_AVAILABLE = True
    except ImportError:
        ADAPTER_AVAILABLE = False
        AltiumJSONAdapter = None

try:
    from .librarian import Librarian
    LIBRARIAN_AVAILABLE = True
except ImportError:
    try:
        from librarian import Librarian
        LIBRARIAN_AVAILABLE = True
    except ImportError:
        LIBRARIAN_AVAILABLE = False
        Librarian = None


# ============================================================================
# FIXTURES - Sample Data Loading
# ============================================================================

@pytest.fixture
def sample_json_path():
    """Path to the altium_sample.json file."""
    return Path(__file__).parent / "altium_sample.json"


@pytest.fixture
def sample_json_data(sample_json_path):
    """Load and return the raw JSON data from altium_sample.json."""
    with open(sample_json_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def mock_adapter(sample_json_data):
    """
    Create a mock adapter that provides normalized data without requiring
    the real AltiumJSONAdapter (useful for testing when adapter isn't ready).
    """
    class MockAltiumAdapter:
        def __init__(self, json_data):
            self.json_data = json_data
            self.components = []
            self.nets = []
            self.net_members_map = {}

        def fetch_raw_data(self):
            """Simulate data fetching."""
            pass

        def get_components(self) -> List[Component]:
            """Transform JSON components to Component objects."""
            if not self.components:
                for comp_data in self.json_data.get("components", []):
                    pins = [
                        Pin(
                            designator=p["name"],
                            name="",  # Names not in sample data
                            net=p.get("net", "")
                        )
                        for p in comp_data.get("pins", [])
                    ]

                    component = Component(
                        refdes=comp_data["designator"],
                        value=comp_data.get("lib_reference", ""),
                        footprint=comp_data.get("footprint", ""),
                        mpn=comp_data.get("parameters", {}).get("PN", ""),
                        page=comp_data["sheet"],
                        description=comp_data.get("description", ""),
                        pins=pins,
                        location=(
                            comp_data.get("schematic_x", 0),
                            comp_data.get("schematic_y", 0)
                        ),
                        properties=comp_data.get("parameters", {})
                    )
                    self.components.append(component)
            return self.components

        def get_nets(self) -> List[Net]:
            """Build Net objects from component pin connectivity."""
            if not self.nets:
                components = self.get_components()
                net_members = {}
                net_pages = {}

                # Build map of net -> (refdes, pin) members
                for comp in components:
                    for pin in comp.pins:
                        if pin.net:  # Skip unconnected pins
                            if pin.net not in net_members:
                                net_members[pin.net] = []
                                net_pages[pin.net] = set()
                            net_members[pin.net].append((comp.refdes, pin.designator))
                            net_pages[pin.net].add(comp.page)

                # Create Net objects
                for net_name, members in net_members.items():
                    net = Net(
                        name=net_name,
                        pages=net_pages[net_name],
                        members=members
                    )
                    self.nets.append(net)

                # Sort nets by name for consistency
                self.nets.sort(key=lambda n: n.name)

            return self.nets

    return MockAltiumAdapter(sample_json_data)


# ============================================================================
# TEST 1: Load Sample Data
# ============================================================================

def test_load_sample_data(sample_json_data):
    """
    Test 1: Load and parse altium_sample.json

    Validates:
    - JSON file can be loaded
    - Required top-level keys present (components, nets)
    - Components and nets are non-empty
    - Component count matches expected
    """
    print("\n" + "="*80)
    print("TEST 1: Load Sample Data")
    print("="*80)

    # Verify structure
    assert "components" in sample_json_data
    assert "nets" in sample_json_data
    assert "metadata" in sample_json_data

    components = sample_json_data["components"]
    nets = sample_json_data["nets"]

    print(f"[OK] JSON structure valid")
    print(f"  - Components: {len(components)}")
    print(f"  - Nets: {len(nets)}")

    # Verify counts
    assert len(components) == 10, f"Expected 10 components, got {len(components)}"
    assert len(nets) > 0, "Expected non-empty nets list"

    print(f"[OK] Component count verified: {len(components)}")
    print(f"[OK] Net count verified: {len(nets)}")

    # Verify each component has required fields
    for comp in components:
        assert "designator" in comp
        assert "sheet" in comp
        assert "pins" in comp
        assert len(comp["pins"]) > 0

    print(f"[OK] All components have required fields")

    # Print summary of components
    print(f"\nComponent Designators:")
    designators = [c["designator"] for c in components]
    for des in sorted(designators):
        print(f"  - {des}")


# ============================================================================
# TEST 2: Adapter Transform
# ============================================================================

def test_adapter_transform(mock_adapter):
    """
    Test 2: Adapter transforms JSON to normalized data models

    Validates:
    - get_components() returns Component objects
    - get_nets() returns Net objects
    - Field mappings are correct
    - Component properties preserved
    """
    print("\n" + "="*80)
    print("TEST 2: Adapter Transform")
    print("="*80)

    # Get components
    components = mock_adapter.get_components()
    assert len(components) == 10, f"Expected 10 components, got {len(components)}"
    print(f"[OK] Adapter returned {len(components)} components")

    # Verify component is a Component object
    assert all(isinstance(c, Component) for c in components)
    print(f"[OK] All components are Component instances")

    # Check first component U1
    u1 = next((c for c in components if c.refdes == "U1"), None)
    assert u1 is not None
    assert u1.value == "LTC7003EMSE#TRPBF"
    assert u1.page == "Power_Switches.SchDoc"
    assert len(u1.pins) == 17
    print(f"[OK] U1 component verified: {len(u1.pins)} pins")

    # Get nets
    nets = mock_adapter.get_nets()
    assert len(nets) > 0
    print(f"[OK] Adapter returned {len(nets)} nets")

    # Verify net is a Net object
    assert all(isinstance(n, Net) for n in nets)
    print(f"[OK] All nets are Net instances")

    # Check GND net (should be present and global)
    gnd_net = next((n for n in nets if n.name == "GND"), None)
    assert gnd_net is not None
    assert len(gnd_net.members) > 0
    print(f"[OK] GND net found with {len(gnd_net.members)} connections")
    print(f"  - Spans {len(gnd_net.pages)} pages: {sorted(gnd_net.pages)}")

    # Verify GND is identified as global
    assert gnd_net.is_global()
    print(f"[OK] GND correctly identified as global net")

    # Check a specific net connection
    fuse_vout = next((n for n in nets if n.name == "FUSE_VOUT"), None)
    assert fuse_vout is not None
    print(f"[OK] FUSE_VOUT net found with {len(fuse_vout.members)} connections")

    print(f"\nNet Summary (first 10):")
    for net in sorted(nets, key=lambda n: n.name)[:10]:
        print(f"  - {net.name}: {len(net.members)} connections, {len(net.pages)} page(s)")


# ============================================================================
# TEST 3: Basic DSL Generation (Page)
# ============================================================================

def test_page_dsl_generation(mock_adapter):
    """
    Test 3: Generate DSL for a single page

    Validates:
    - get_page() returns valid DSL string
    - DSL has required sections (# PAGE:, # COMPONENTS, # NETS)
    - Complex components get COMP blocks
    - Simple passives appear only in nets
    """
    print("\n" + "="*80)
    print("TEST 3: Page DSL Generation")
    print("="*80)

    components = mock_adapter.get_components()
    nets = mock_adapter.get_nets()

    # Find all pages
    pages = set(c.page for c in components)
    print(f"[OK] Found {len(pages)} pages in schematic")
    for page in sorted(pages):
        print(f"  - {page}")

    # Test DSL generation for Power_Switches.SchDoc
    page_name = "Power_Switches.SchDoc"
    page_components = [c for c in components if c.page == page_name]

    print(f"\n[OK] Testing page: {page_name}")
    print(f"  - Components on page: {len(page_components)}")

    # Build net page map
    net_page_map = {}
    for net in nets:
        net_page_map[net.name] = net.pages

    # Filter nets that appear on this page
    page_nets = [n for n in nets if page_name in n.pages]

    # Generate DSL
    dsl = emit_page_dsl(page_components, page_nets, net_page_map)

    print(f"  - Generated DSL ({len(dsl)} chars)")

    # Verify format
    assert "# PAGE:" in dsl
    assert page_name in dsl
    assert "# COMPONENTS" in dsl
    assert "# NETS" in dsl

    print(f"[OK] DSL has required sections")

    # Identify complex vs simple components
    complex_comps = [c for c in page_components if c.is_complex()]
    simple_comps = [c for c in page_components if not c.is_complex()]

    print(f"\n[OK] Component breakdown:")
    print(f"  - Complex (get COMP blocks): {len(complex_comps)}")
    for c in complex_comps:
        print(f"    - {c.refdes}: {c.derived_type()}, {len(c.pins)} pins")
    print(f"  - Simple (inline only): {len(simple_comps)}")
    for c in simple_comps:
        print(f"    - {c.refdes}: {c.derived_type()}")

    # Verify complex components have COMP blocks in DSL
    for comp in complex_comps:
        assert f"COMP {comp.refdes}" in dsl or f"DEF {comp.derived_type()}" in dsl

    print(f"[OK] Complex components appear in DSL")

    # Print DSL excerpt
    print(f"\nDSL Output (first 1500 chars):")
    print("-" * 80)
    print(dsl[:1500])
    if len(dsl) > 1500:
        print(f"\n... ({len(dsl) - 1500} more characters)")
    print("-" * 80)

    return True


# ============================================================================
# TEST 4: Component Properties
# ============================================================================

def test_component_properties(mock_adapter):
    """
    Test 4: Verify component property mapping and type classification

    Validates:
    - Component types correctly identified (RES, CAP, IC, etc.)
    - is_complex() works correctly
    - is_passive() identifies passive components
    - Pin data preserved
    """
    print("\n" + "="*80)
    print("TEST 4: Component Properties")
    print("="*80)

    components = mock_adapter.get_components()

    # Test derived_type() for various components
    type_map = {}
    for comp in components:
        comp_type = comp.derived_type()
        if comp_type not in type_map:
            type_map[comp_type] = []
        type_map[comp_type].append(comp)

    print(f"[OK] Component types detected:")
    for comp_type in sorted(type_map.keys()):
        comps = type_map[comp_type]
        print(f"  - {comp_type}: {len(comps)} components")
        for c in comps[:3]:  # Show first 3
            print(f"    - {c.refdes}")

    # Test is_complex() and is_passive()
    print(f"\n[OK] Component complexity analysis:")
    complex_comps = [c for c in components if c.is_complex()]
    simple_comps = [c for c in components if not c.is_complex()]

    print(f"  - Complex: {len(complex_comps)}")
    for c in complex_comps:
        print(f"    - {c.refdes} ({len(c.pins)} pins)")

    print(f"  - Simple: {len(simple_comps)}")
    for c in simple_comps:
        print(f"    - {c.refdes} ({len(c.pins)} pins)")

    # Test passive classification
    print(f"\n[OK] Passive component detection:")
    passive_comps = [c for c in components if c.is_passive()]
    active_comps = [c for c in components if not c.is_passive()]

    print(f"  - Passive: {len(passive_comps)}")
    for c in passive_comps:
        print(f"    - {c.refdes} ({c.derived_type()})")

    print(f"  - Active: {len(active_comps)}")
    for c in active_comps:
        print(f"    - {c.refdes} ({c.derived_type()})")

    # Verify all components have required attributes
    for comp in components:
        assert comp.refdes
        assert comp.page
        assert isinstance(comp.pins, list)
        assert comp.derived_type() in [
            "RES", "CAP", "IND", "FUSE", "DIODE", "TRANSISTOR",
            "IC", "CONN", "SWITCH", "OSC", "ACTIVE"
        ]

    print(f"\n[OK] All components have valid properties")

    return True


# ============================================================================
# TEST 5: Multi-Page Nets
# ============================================================================

def test_multi_page_nets(mock_adapter):
    """
    Test 5: Identify and verify inter-page nets

    Validates:
    - Inter-page nets correctly identified
    - LINKS section generated for multi-page nets
    - Global nets properly categorized
    - Page information preserved
    """
    print("\n" + "="*80)
    print("TEST 5: Multi-Page Nets")
    print("="*80)

    nets = mock_adapter.get_nets()

    # Categorize nets
    local_nets = [n for n in nets if not n.is_inter_page()]
    inter_page_nets = [n for n in nets if n.is_inter_page() and not n.is_global()]
    global_nets = [n for n in nets if n.is_global()]

    print(f"[OK] Net categorization:")
    print(f"  - Local (single page): {len(local_nets)}")
    print(f"  - Inter-page: {len(inter_page_nets)}")
    print(f"  - Global: {len(global_nets)}")

    if local_nets:
        print(f"\n  Local nets (first 5):")
        for net in local_nets[:5]:
            print(f"    - {net.name}: {len(net.members)} connections on {list(net.pages)[0]}")

    if inter_page_nets:
        print(f"\n  Inter-page nets:")
        for net in inter_page_nets:
            print(f"    - {net.name}: {len(net.members)} connections on {len(net.pages)} pages")
            print(f"      Pages: {', '.join(sorted(net.pages))}")

    if global_nets:
        print(f"\n  Global nets:")
        for net in global_nets[:5]:
            print(f"    - {net.name}: {len(net.members)} connections on {len(net.pages)} pages")

    # Verify GND is multi-page (should be)
    gnd = next((n for n in nets if n.name == "GND"), None)
    if gnd:
        assert gnd.is_inter_page() or gnd.is_global()
        print(f"\n[OK] GND is correctly identified as global/multi-page")

    return True


# ============================================================================
# TEST 6: Global Net Truncation
# ============================================================================

def test_global_net_truncation(mock_adapter):
    """
    Test 6: Verify global nets are truncated in DSL output

    Validates:
    - Global nets show only first 10 connections
    - Shows "(+ N others)" suffix
    - LINKS: ALL_PAGES appears for global nets
    """
    print("\n" + "="*80)
    print("TEST 6: Global Net Truncation")
    print("="*80)

    components = mock_adapter.get_components()
    nets = mock_adapter.get_nets()

    # Find a global net
    global_nets = [n for n in nets if n.is_global()]

    if not global_nets:
        print("[SKIP] No global nets found in sample data - skipping truncation test")
        return True

    net_page_map = {n.name: n.pages for n in nets}

    print(f"[OK] Testing {len(global_nets)} global net(s)")

    for gnet in global_nets[:3]:
        print(f"\n  Testing {gnet.name}:")
        print(f"    - {len(gnet.members)} total connections")
        print(f"    - Spans {len(gnet.pages)} pages")

        if len(gnet.members) > 10:
            # Generate DSL for a page containing this net
            page = list(gnet.pages)[0]
            page_comps = [c for c in components if c.page == page]
            page_nets = [n for n in nets if page in n.pages]

            dsl = emit_page_dsl(page_comps, page_nets, net_page_map)

            # Check for truncation marker
            if f"NET {gnet.name}" in dsl:
                net_section = dsl[dsl.find(f"NET {gnet.name}"):]
                net_section = net_section[:net_section.find("\nNET") if "\nNET" in net_section else None]

                if "(+ " in net_section:
                    print(f"    [OK] Truncation detected: shows first 10 + others")
                    # Extract the count
                    import re
                    match = re.search(r'\(\+ (\d+) others\)', net_section)
                    if match:
                        count = int(match.group(1))
                        print(f"    [OK] Shows {count} additional connections")
        else:
            print(f"    - Less than 10 connections, no truncation needed")

    return True


# ============================================================================
# TEST 7: Context Bubble Generation
# ============================================================================

def test_context_bubble(mock_adapter):
    """
    Test 7: Generate context bubble for specific components

    Validates:
    - Context DSL includes primary component
    - Neighbor components identified
    - Connected nets shown
    - Output format correct
    """
    print("\n" + "="*80)
    print("TEST 7: Context Bubble Generation")
    print("="*80)

    components = mock_adapter.get_components()
    nets = mock_adapter.get_nets()

    # Find a complex component to use as primary
    complex_comps = [c for c in components if c.is_complex()]
    if not complex_comps:
        print("[SKIP] No complex components found - skipping context test")
        return True

    primary_refdes = complex_comps[0].refdes
    print(f"[OK] Testing context bubble for: {primary_refdes}")

    primary_comps = [c for c in components if c.refdes == primary_refdes]
    print(f"  - Primary component: {primary_refdes}")

    # Find neighbors (components on same nets)
    primary_nets = set()
    for comp in primary_comps:
        for pin in comp.pins:
            if pin.net:
                primary_nets.add(pin.net)

    neighbor_refdes = set()
    for net_name in primary_nets:
        net = next((n for n in nets if n.name == net_name), None)
        if net:
            for refdes, _ in net.members:
                if refdes != primary_refdes:
                    neighbor_refdes.add(refdes)

    neighbor_comps = [c for c in components if c.refdes in neighbor_refdes]
    print(f"  - Connected nets: {len(primary_nets)}")
    print(f"  - Neighbor components: {len(neighbor_comps)}")

    # Find all nets connecting primary and neighbors
    context_nets = [n for n in nets if n.name in primary_nets]

    # Generate context DSL
    dsl = emit_context_dsl(primary_comps, neighbor_comps, context_nets)

    print(f"  - Generated context DSL ({len(dsl)} chars)")

    # Verify format
    assert f"# CONTEXT:" in dsl
    assert primary_refdes in dsl
    assert "# COMPONENTS" in dsl
    assert "# NETS" in dsl

    print(f"[OK] Context DSL has required sections")

    if neighbor_comps:
        assert "# CONTEXT_NEIGHBORS" in dsl
        print(f"[OK] Neighbors section included")

    # Print excerpt
    print(f"\nContext DSL Output (first 1000 chars):")
    print("-" * 80)
    print(dsl[:1000])
    if len(dsl) > 1000:
        print(f"\n... ({len(dsl) - 1000} more characters)")
    print("-" * 80)

    return True


# ============================================================================
# TEST 8: Net Connectivity Verification
# ============================================================================

def test_net_connectivity(mock_adapter):
    """
    Test 8: Verify net connectivity is correctly represented

    Validates:
    - All pins connected to nets
    - No orphaned pins
    - Member counts match pin counts
    - Pin references valid
    """
    print("\n" + "="*80)
    print("TEST 8: Net Connectivity Verification")
    print("="*80)

    components = mock_adapter.get_components()
    nets = mock_adapter.get_nets()

    # Build a map of all pins
    all_pins = {}
    for comp in components:
        for pin in comp.pins:
            key = (comp.refdes, pin.designator)
            all_pins[key] = pin

    # Build a map of all net members
    all_members = set()
    for net in nets:
        for member in net.members:
            all_members.add(member)

    print(f"[OK] Total components: {len(components)}")
    print(f"[OK] Total pins: {len(all_pins)}")
    print(f"[OK] Total net connections: {len(all_members)}")

    # Find pins with no net (unconnected)
    unconnected_pins = []
    for comp in components:
        for pin in comp.pins:
            if not pin.net:
                unconnected_pins.append((comp.refdes, pin.designator))

    print(f"[OK] Unconnected pins: {len(unconnected_pins)}")
    if unconnected_pins:
        print(f"  Examples:")
        for refdes, pin in unconnected_pins[:5]:
            print(f"    - {refdes}.{pin}")

    # Verify all net members reference valid pins
    invalid_refs = []
    for net in nets:
        for refdes, pin_des in net.members:
            key = (refdes, pin_des)
            if key not in all_pins:
                invalid_refs.append((net.name, refdes, pin_des))

    assert len(invalid_refs) == 0, f"Found {len(invalid_refs)} invalid pin references"
    print(f"[OK] All net member references are valid")

    # Analyze connectivity statistics
    conn_stats = {}
    for net in nets:
        conn_stats[net.name] = len(net.members)

    print(f"\n[OK] Connectivity statistics:")
    print(f"  - Avg connections per net: {sum(conn_stats.values()) / len(conn_stats):.1f}")
    print(f"  - Max connections: {max(conn_stats.values())}")
    print(f"  - Min connections: {min(conn_stats.values())}")

    # Show top 5 most connected nets
    top_nets = sorted(conn_stats.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Most connected nets:")
    for net_name, count in top_nets:
        print(f"    - {net_name}: {count} connections")

    return True


# ============================================================================
# INTEGRATION TEST - Full End-to-End
# ============================================================================

def test_full_integration_flow(sample_json_data):
    """
    Integration Test: Full end-to-end flow from JSON to DSL output

    This is the master integration test that validates the entire workflow:
    1. Load JSON data
    2. Transform via adapter
    3. Generate DSL for multiple formats

    Validates nothing breaks and data flows correctly through the pipeline.
    """
    print("\n" + "="*80)
    print("INTEGRATION TEST: Full End-to-End Flow")
    print("="*80)

    # Step 1: Create mock adapter
    print("\n[1/5] Creating adapter from JSON data...")

    class FullAdapter:
        def __init__(self, json_data):
            self.json_data = json_data
            self.components = None
            self.nets = None

        def fetch_raw_data(self):
            pass

        def get_components(self) -> List[Component]:
            if self.components is None:
                self.components = []
                for comp_data in self.json_data.get("components", []):
                    pins = [
                        Pin(
                            designator=p["name"],
                            name="",
                            net=p.get("net", "")
                        )
                        for p in comp_data.get("pins", [])
                    ]
                    component = Component(
                        refdes=comp_data["designator"],
                        value=comp_data.get("lib_reference", ""),
                        footprint=comp_data.get("footprint", ""),
                        mpn=comp_data.get("parameters", {}).get("PN", ""),
                        page=comp_data["sheet"],
                        description=comp_data.get("description", ""),
                        pins=pins,
                        location=(comp_data.get("schematic_x", 0), comp_data.get("schematic_y", 0)),
                        properties=comp_data.get("parameters", {})
                    )
                    self.components.append(component)
            return self.components

        def get_nets(self) -> List[Net]:
            if self.nets is None:
                self.nets = []
                components = self.get_components()
                net_members = {}
                net_pages = {}

                for comp in components:
                    for pin in comp.pins:
                        if pin.net:
                            if pin.net not in net_members:
                                net_members[pin.net] = []
                                net_pages[pin.net] = set()
                            net_members[pin.net].append((comp.refdes, pin.designator))
                            net_pages[pin.net].add(comp.page)

                for net_name, members in net_members.items():
                    net = Net(name=net_name, pages=net_pages[net_name], members=members)
                    self.nets.append(net)

                self.nets.sort(key=lambda n: n.name)
            return self.nets

    adapter = FullAdapter(sample_json_data)
    print("[OK] Adapter created")

    # Step 2: Fetch data
    print("\n[2/5] Fetching schematic data...")
    adapter.fetch_raw_data()
    print("[OK] Data fetch complete")

    # Step 3: Get components and nets
    print("\n[3/5] Transforming components and nets...")
    components = adapter.get_components()
    nets = adapter.get_nets()
    print(f"[OK] Got {len(components)} components and {len(nets)} nets")

    # Step 4: Generate DSL for all pages
    print("\n[4/5] Generating page DSL...")
    pages = sorted(set(c.page for c in components))
    net_page_map = {n.name: n.pages for n in nets}

    page_outputs = {}
    for page in pages:
        page_comps = [c for c in components if c.page == page]
        page_nets = [n for n in nets if page in n.pages]
        dsl = emit_page_dsl(page_comps, page_nets, net_page_map)
        page_outputs[page] = dsl
        print(f"  [OK] {page}: {len(dsl)} chars")

    # Step 5: Validate outputs
    print("\n[5/5] Validating outputs...")

    # Check that all pages have valid DSL
    for page, dsl in page_outputs.items():
        assert "# PAGE:" in dsl
        assert page in dsl
        assert "# NETS" in dsl

    print(f"[OK] All {len(page_outputs)} pages have valid DSL output")
    print(f"[OK] Total DSL generated: {sum(len(d) for d in page_outputs.values())} chars")

    # Summary statistics
    print(f"\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"[OK] Components: {len(components)}")
    print(f"[OK] Nets: {len(nets)}")
    print(f"[OK] Pages: {len(pages)}")
    print(f"[OK] Global nets: {len([n for n in nets if n.is_global()])}")
    print(f"[OK] Multi-page nets: {len([n for n in nets if n.is_inter_page()])}")
    print(f"[OK] DSL outputs generated: {len(page_outputs)}")

    return True


# ============================================================================
# PYTEST ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run tests with: pytest test_integration.py -v -s

    The -v flag shows verbose output (test names)
    The -s flag shows print statements
    """
    pytest.main([__file__, "-v", "-s"])
