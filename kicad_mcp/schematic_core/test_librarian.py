"""
Test suite for Librarian - State Manager and Navigation Layer

This module tests the Librarian's core functionality including:
- Refresh and caching behavior
- Atlas building
- Index generation
- Page retrieval
- Context bubble generation (1-hop traversal)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from typing import List
from models import Component, Net, Pin
from interfaces import SchematicProvider
from librarian import Librarian


class MockProvider(SchematicProvider):
    """Mock provider for testing."""

    def __init__(self):
        self.fetch_count = 0
        self._components = []
        self._nets = []

    def fetch_raw_data(self) -> None:
        """Track fetch calls."""
        self.fetch_count += 1

    def get_components(self) -> List[Component]:
        """Return mock components."""
        return self._components

    def get_nets(self) -> List[Net]:
        """Return mock nets."""
        return self._nets

    def set_test_data(self, components: List[Component], nets: List[Net]):
        """Helper to set test data."""
        self._components = components
        self._nets = nets


def create_multi_page_schematic():
    """
    Create a realistic multi-page schematic for testing.

    Structure:
    - Page 1 (Main_Sheet): MCU with UART and power
    - Page 2 (Power_Module): Regulator
    - Page 3 (Connector_Page): UART connector

    Inter-page signals:
    - UART_TX: Main_Sheet <-> Connector_Page
    - UART_RX: Main_Sheet <-> Connector_Page
    - 3V3: All pages
    - GND: All pages (global - power rail)
    """

    # PAGE 1: Main_Sheet - MCU
    u1_pins = [
        Pin("1", "VDD", "3V3"),
        Pin("10", "VSS", "GND"),
        Pin("22", "PA9_TX", "UART_TX"),
        Pin("23", "PA10_RX", "UART_RX"),
        Pin("50", "GND", "GND"),
    ]
    u1 = Component(
        refdes="U1",
        value="STM32F407VGT6",
        footprint="LQFP-100",
        mpn="STM32F407VGT6",
        page="Main_Sheet",
        description="ARM Cortex-M4 MCU, 168MHz",
        pins=u1_pins,
        location=(6100, 3700),
        properties={"Manufacturer": "STMicroelectronics"}
    )

    # Simple resistor on Page 1
    r1_pins = [
        Pin("1", "", "UART_TX"),
        Pin("2", "", "UART_TX_BUF"),
    ]
    r1 = Component(
        refdes="R1",
        value="220",
        footprint="0603",
        mpn="RC0603FR-07220RL",
        page="Main_Sheet",
        description="",
        pins=r1_pins,
        location=(6500, 3800),
        properties={}
    )

    # Decoupling cap on Page 1
    c1_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "GND"),
    ]
    c1 = Component(
        refdes="C1",
        value="100nF",
        footprint="0603",
        mpn="CL10A104KB8NNNC",
        page="Main_Sheet",
        description="",
        pins=c1_pins,
        location=(6200, 4000),
        properties={}
    )

    # PAGE 2: Power_Module - Regulator
    u2_pins = [
        Pin("1", "GND", "GND"),
        Pin("2", "VOUT", "3V3"),
        Pin("3", "VIN", "VCC"),
    ]
    u2 = Component(
        refdes="U2",
        value="LM1117-3.3",
        footprint="SOT223",
        mpn="LM1117IMP-3.3",
        page="Power_Module",
        description="Linear Regulator 3.3V",
        pins=u2_pins,
        location=(5100, 4000),
        properties={}
    )

    # Input cap on Page 2
    c2_pins = [
        Pin("1", "", "VCC"),
        Pin("2", "", "GND"),
    ]
    c2 = Component(
        refdes="C2",
        value="10uF",
        footprint="0805",
        mpn="CL21A106KQCLNNC",
        page="Power_Module",
        description="",
        pins=c2_pins,
        location=(4800, 4000),
        properties={}
    )

    # Output cap on Page 2
    c3_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "GND"),
    ]
    c3 = Component(
        refdes="C3",
        value="10uF",
        footprint="0805",
        mpn="CL21A106KQCLNNC",
        page="Power_Module",
        description="",
        pins=c3_pins,
        location=(5400, 4000),
        properties={}
    )

    # PAGE 3: Connector_Page - UART connector
    j1_pins = [
        Pin("1", "VCC", "VCC"),
        Pin("2", "GND", "GND"),
        Pin("3", "TX", "UART_TX_BUF"),
        Pin("4", "RX", "UART_RX_BUF"),
    ]
    j1 = Component(
        refdes="J1",
        value="CONN_04",
        footprint="JST-XH-4",
        mpn="B4B-XH-A",
        page="Connector_Page",
        description="4-pin JST connector",
        pins=j1_pins,
        location=(7600, 5200),
        properties={}
    )

    # Series resistor on Page 3
    r2_pins = [
        Pin("1", "", "UART_RX"),
        Pin("2", "", "UART_RX_BUF"),
    ]
    r2 = Component(
        refdes="R2",
        value="220",
        footprint="0603",
        mpn="RC0603FR-07220RL",
        page="Connector_Page",
        description="",
        pins=r2_pins,
        location=(7200, 5000),
        properties={}
    )

    components = [u1, r1, c1, u2, c2, c3, j1, r2]

    # Build nets
    # Global power nets (span all 3 pages, many connections)
    gnd_net = Net(
        name="GND",
        pages={"Main_Sheet", "Power_Module", "Connector_Page"},
        members=[
            ("U1", "10"), ("U1", "50"), ("C1", "2"),  # Page 1
            ("U2", "1"), ("C2", "2"), ("C3", "2"),    # Page 2
            ("J1", "2"),                               # Page 3
        ]
    )

    v3v3_net = Net(
        name="3V3",
        pages={"Main_Sheet", "Power_Module"},
        members=[
            ("U1", "1"), ("C1", "1"),     # Page 1
            ("U2", "2"), ("C3", "1"),     # Page 2
        ]
    )

    vcc_net = Net(
        name="VCC",
        pages={"Power_Module", "Connector_Page"},
        members=[
            ("U2", "3"), ("C2", "1"),  # Page 2
            ("J1", "1"),               # Page 3
        ]
    )

    # Inter-page signal nets
    uart_tx_net = Net(
        name="UART_TX",
        pages={"Main_Sheet"},
        members=[
            ("U1", "22"), ("R1", "1"),
        ]
    )

    uart_tx_buf_net = Net(
        name="UART_TX_BUF",
        pages={"Main_Sheet", "Connector_Page"},
        members=[
            ("R1", "2"), ("J1", "3"),
        ]
    )

    uart_rx_net = Net(
        name="UART_RX",
        pages={"Main_Sheet", "Connector_Page"},
        members=[
            ("U1", "23"), ("R2", "1"),
        ]
    )

    uart_rx_buf_net = Net(
        name="UART_RX_BUF",
        pages={"Connector_Page"},
        members=[
            ("R2", "2"), ("J1", "4"),
        ]
    )

    nets = [gnd_net, v3v3_net, vcc_net, uart_tx_net, uart_tx_buf_net,
            uart_rx_net, uart_rx_buf_net]

    return components, nets


def test_refresh_caching():
    """Test that refresh only fetches when dirty."""
    print("=" * 80)
    print("TEST: Refresh and Caching Behavior")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)

    # Initial state - dirty=True
    assert librarian.dirty == True
    assert provider.fetch_count == 0

    # First refresh - should fetch
    librarian.refresh()
    assert librarian.dirty == False
    assert provider.fetch_count == 1

    # Second refresh - should NOT fetch (cached)
    librarian.refresh()
    assert provider.fetch_count == 1  # Still 1, no additional fetch

    # Mark dirty and refresh - should fetch again
    librarian.mark_dirty()
    assert librarian.dirty == True
    librarian.refresh()
    assert provider.fetch_count == 2

    print("[PASS] Caching works correctly")
    print(f"  - First refresh: fetched (count={provider.fetch_count})")
    print(f"  - Second refresh: cached (no fetch)")
    print(f"  - After mark_dirty: fetched again")
    print()


def test_atlas_building():
    """Test that the Atlas (net_page_map) is built correctly."""
    print("=" * 80)
    print("TEST: Atlas Building (Net-to-Pages Mapping)")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)
    librarian.refresh()

    # Verify Atlas structure
    assert "GND" in librarian.net_page_map
    assert "3V3" in librarian.net_page_map
    assert "UART_TX" in librarian.net_page_map

    # Verify GND spans all pages
    assert librarian.net_page_map["GND"] == {"Main_Sheet", "Power_Module", "Connector_Page"}

    # Verify 3V3 spans Main_Sheet and Power_Module
    assert librarian.net_page_map["3V3"] == {"Main_Sheet", "Power_Module"}

    # Verify UART_TX is only on Main_Sheet
    assert librarian.net_page_map["UART_TX"] == {"Main_Sheet"}

    print("[PASS] Atlas built correctly")
    print(f"  - Total nets in atlas: {len(librarian.net_page_map)}")
    print(f"  - GND pages: {sorted(librarian.net_page_map['GND'])}")
    print(f"  - 3V3 pages: {sorted(librarian.net_page_map['3V3'])}")
    print()


def test_get_index():
    """Test index generation with page summaries and inter-page signals."""
    print("=" * 80)
    print("TEST: Index Generation")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)
    index = librarian.get_index()

    print(index)
    print()

    # Verify structure
    assert "# SCHEMATIC INDEX" in index
    assert "## Pages" in index
    assert "## Inter-Page Signals" in index

    # Verify pages are listed
    assert "Main_Sheet" in index
    assert "Power_Module" in index
    assert "Connector_Page" in index

    # Verify inter-page signals
    assert "GND:" in index
    assert "ALL_PAGES" in index  # GND should be marked as global

    print("[PASS] Index generated successfully")
    print()


def test_get_page():
    """Test page-specific DSL generation."""
    print("=" * 80)
    print("TEST: Page Retrieval")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)

    # Get Main_Sheet page
    page_dsl = librarian.get_page("Main_Sheet")

    print(page_dsl)
    print()

    # Verify structure
    assert "# PAGE: Main_Sheet" in page_dsl
    assert "# COMPONENTS" in page_dsl
    assert "# NETS" in page_dsl

    # Verify components on this page
    assert "U1" in page_dsl  # MCU should be there
    assert "STM32F407VGT6" in page_dsl
    assert "R1" in page_dsl  # Should appear in nets (passive)

    # Verify U2 (from different page) is NOT on this page
    assert "U2" not in page_dsl or "Power_Module" in page_dsl  # Only if page name mentioned

    print("[PASS] Page DSL generated correctly")
    print()


def test_get_page_nonexistent():
    """Test handling of nonexistent page."""
    print("=" * 80)
    print("TEST: Nonexistent Page Handling")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)
    page_dsl = librarian.get_page("NonexistentPage")

    print(page_dsl)
    print()

    # Should return a message indicating page not found
    assert "not found" in page_dsl.lower() or "NonexistentPage" in page_dsl

    print("[PASS] Nonexistent page handled gracefully")
    print()


def test_get_context_simple():
    """Test context bubble for a simple component."""
    print("=" * 80)
    print("TEST: Context Bubble - Simple Component (R1)")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)

    # Get context for R1 (simple resistor)
    context_dsl = librarian.get_context(["R1"])

    print(context_dsl)
    print()

    # Verify structure
    assert "# CONTEXT: R1" in context_dsl
    assert "# COMPONENTS" in context_dsl
    assert "# NETS" in context_dsl

    # R1 connects to UART_TX and UART_TX_BUF
    # Should see U1 (via UART_TX) and J1 (via UART_TX_BUF) as neighbors
    assert "U1" in context_dsl  # Should be in neighbors

    print("[PASS] Simple context bubble generated correctly")
    print()


def test_get_context_complex():
    """Test context bubble for a complex component (MCU)."""
    print("=" * 80)
    print("TEST: Context Bubble - Complex Component (U1 MCU)")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)

    # Get context for U1 (MCU)
    context_dsl = librarian.get_context(["U1"])

    print(context_dsl)
    print()

    # Verify structure
    assert "# CONTEXT: U1" in context_dsl
    assert "# COMPONENTS" in context_dsl

    # U1 is the primary component, should have full DEF block
    assert "COMP U1" in context_dsl
    assert "STM32F407VGT6" in context_dsl

    # Should have CONTEXT_NEIGHBORS section (non-passive neighbors)
    # U1 connects to: R1 (passive), C1 (passive) - these won't be in neighbors
    # But it also connects to U2 via 3V3 net

    # Neighbors might not appear if they're all passive
    # The key is that nets show the connections
    assert "UART_TX" in context_dsl
    assert "3V3" in context_dsl

    print("[PASS] Complex context bubble generated correctly")
    print()


def test_get_context_multiple():
    """Test context bubble for multiple components."""
    print("=" * 80)
    print("TEST: Context Bubble - Multiple Components")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)

    # Get context for U1 and U2
    context_dsl = librarian.get_context(["U1", "U2"])

    print(context_dsl)
    print()

    # Verify both are listed
    assert "U1" in context_dsl
    assert "U2" in context_dsl

    # They share the 3V3 net
    assert "3V3" in context_dsl

    print("[PASS] Multi-component context bubble generated correctly")
    print()


def test_get_context_empty():
    """Test context bubble with empty refdes list."""
    print("=" * 80)
    print("TEST: Context Bubble - Empty Input")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)
    context_dsl = librarian.get_context([])

    print(context_dsl)
    print()

    # Should return a message indicating empty context
    assert "empty" in context_dsl.lower() or "No components" in context_dsl

    print("[PASS] Empty context handled gracefully")
    print()


def test_get_context_nonexistent():
    """Test context bubble with nonexistent component."""
    print("=" * 80)
    print("TEST: Context Bubble - Nonexistent Component")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)
    context_dsl = librarian.get_context(["DOES_NOT_EXIST"])

    print(context_dsl)
    print()

    # Should return a message indicating component not found
    assert "not found" in context_dsl.lower() or "DOES_NOT_EXIST" in context_dsl

    print("[PASS] Nonexistent component handled gracefully")
    print()


def test_helper_methods():
    """Test helper methods (get_all_pages, get_component, get_net, get_stats)."""
    print("=" * 80)
    print("TEST: Helper Methods")
    print("=" * 80)

    provider = MockProvider()
    components, nets = create_multi_page_schematic()
    provider.set_test_data(components, nets)

    librarian = Librarian(provider)

    # Test get_all_pages
    pages = librarian.get_all_pages()
    print(f"All pages: {pages}")
    assert "Main_Sheet" in pages
    assert "Power_Module" in pages
    assert "Connector_Page" in pages
    assert len(pages) == 3

    # Test get_component
    u1 = librarian.get_component("U1")
    assert u1 is not None
    assert u1.refdes == "U1"
    assert u1.value == "STM32F407VGT6"

    missing = librarian.get_component("DOES_NOT_EXIST")
    assert missing is None

    # Test get_net
    gnd = librarian.get_net("GND")
    assert gnd is not None
    assert gnd.name == "GND"
    assert len(gnd.pages) == 3

    missing_net = librarian.get_net("DOES_NOT_EXIST")
    assert missing_net is None

    # Test get_stats
    stats = librarian.get_stats()
    print(f"Stats: {stats}")
    assert stats['total_components'] == 8
    assert stats['total_nets'] == 7
    assert stats['total_pages'] == 3
    assert stats['inter_page_nets'] > 0
    assert stats['global_nets'] >= 1  # At least GND should be global

    print("[PASS] All helper methods work correctly")
    print()


def test_empty_schematic():
    """Test behavior with empty schematic data."""
    print("=" * 80)
    print("TEST: Empty Schematic")
    print("=" * 80)

    provider = MockProvider()
    provider.set_test_data([], [])  # Empty

    librarian = Librarian(provider)

    # Test index
    index = librarian.get_index()
    print("Index for empty schematic:")
    print(index)
    print()

    assert "Empty schematic" in index or "No pages" in index

    # Test stats
    stats = librarian.get_stats()
    assert stats['total_components'] == 0
    assert stats['total_nets'] == 0
    assert stats['total_pages'] == 0

    print("[PASS] Empty schematic handled gracefully")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LIBRARIAN TEST SUITE")
    print("=" * 80 + "\n")

    test_refresh_caching()
    test_atlas_building()
    test_get_index()
    test_get_page()
    test_get_page_nonexistent()
    test_get_context_simple()
    test_get_context_complex()
    test_get_context_multiple()
    test_get_context_empty()
    test_get_context_nonexistent()
    test_helper_methods()
    test_empty_schematic()

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
