"""
Edge Case Tests for DSL Emitter

This module tests various edge cases and boundary conditions
to ensure robust behavior.
"""

from models import Component, Net, Pin
from dsl_emitter import emit_page_dsl, emit_context_dsl, _natural_sort_key


def test_empty_fields():
    """Test components with empty/missing fields."""
    print("=" * 80)
    print("TEST: Empty Fields Handling")
    print("=" * 80)

    # Component with empty MPN and footprint
    u1_pins = [
        Pin("1", "VCC", "3V3"),
        Pin("2", "GND", "GND"),
        Pin("3", "OUT", "OUTPUT"),
    ]
    u1 = Component(
        refdes="U1",
        value="UNKNOWN_IC",
        footprint="",  # Empty
        mpn="",  # Empty
        page="Test_Page",
        description="Mystery IC with no part number",
        pins=u1_pins,
        location=(0, 0),
        properties={}
    )

    # Component with empty description
    r1_pins = [
        Pin("1", "", "NET1"),
        Pin("2", "", "NET2"),
    ]
    r1 = Component(
        refdes="R1",
        value="10k",
        footprint="0603",
        mpn="",
        page="Test_Page",
        description="",  # Empty
        pins=r1_pins,
        location=(0, 0),
        properties={}
    )

    components = [u1, r1]

    net1 = Net(
        name="3V3",
        pages={"Test_Page"},
        members=[("U1", "1")]
    )

    net2 = Net(
        name="GND",
        pages={"Test_Page"},
        members=[("U1", "2")]
    )

    nets = [net1, net2]
    net_page_map = {"3V3": {"Test_Page"}, "GND": {"Test_Page"}}

    dsl = emit_page_dsl(components, nets, net_page_map)
    print(dsl)
    print()


def test_pin_sorting():
    """Test natural sorting of pin designators."""
    print("=" * 80)
    print("TEST: Pin Natural Sorting")
    print("=" * 80)

    # Component with various pin designator formats
    u1_pins = [
        Pin("100", "PIN100", "NET_A"),
        Pin("1", "PIN1", "NET_B"),
        Pin("2", "PIN2", "NET_C"),
        Pin("10", "PIN10", "NET_D"),
        Pin("A1", "PINA1", "NET_E"),
        Pin("20", "PIN20", "NET_F"),
        Pin("3", "PIN3", "NET_G"),
    ]
    u1 = Component(
        refdes="U1",
        value="TEST_IC",
        footprint="QFP-100",
        mpn="TEST123",
        page="Test_Page",
        description="Test IC with various pin formats",
        pins=u1_pins,
        location=(0, 0),
        properties={}
    )

    components = [u1]
    nets = []
    net_page_map = {}

    dsl = emit_page_dsl(components, nets, net_page_map)
    print(dsl)
    print()

    # Verify sort order
    print("Expected pin order: 1, 2, 3, 10, 20, 100, A1")
    print()


def test_multipart_components():
    """Test handling of multi-part components."""
    print("=" * 80)
    print("TEST: Multi-Part Components")
    print("=" * 80)

    # U1A - Part A of multi-part component
    u1a_pins = [
        Pin("1", "IN+", "SIGNAL_IN_P"),
        Pin("2", "IN-", "SIGNAL_IN_N"),
        Pin("3", "OUT", "SIGNAL_OUT"),
    ]
    u1a = Component(
        refdes="U1A",
        value="LM358",
        footprint="SOIC-8",
        mpn="LM358DR",
        page="Test_Page",
        description="Dual Op-Amp, Part A",
        pins=u1a_pins,
        location=(0, 0),
        properties={},
        multipart_parent="U1"
    )

    # U1B - Part B of multi-part component
    u1b_pins = [
        Pin("5", "IN+", "REF_IN_P"),
        Pin("6", "IN-", "REF_IN_N"),
        Pin("7", "OUT", "REF_OUT"),
    ]
    u1b = Component(
        refdes="U1B",
        value="LM358",
        footprint="SOIC-8",
        mpn="LM358DR",
        page="Test_Page",
        description="Dual Op-Amp, Part B",
        pins=u1b_pins,
        location=(0, 0),
        properties={},
        multipart_parent="U1"
    )

    components = [u1a, u1b]

    net1 = Net("SIGNAL_IN_P", {"Test_Page"}, [("U1A", "1")])
    net2 = Net("SIGNAL_OUT", {"Test_Page"}, [("U1A", "3")])
    net3 = Net("REF_OUT", {"Test_Page"}, [("U1B", "7")])

    nets = [net1, net2, net3]
    net_page_map = {
        "SIGNAL_IN_P": {"Test_Page"},
        "SIGNAL_OUT": {"Test_Page"},
        "REF_OUT": {"Test_Page"},
    }

    dsl = emit_page_dsl(components, nets, net_page_map)
    print(dsl)
    print()


def test_unusual_pin_names():
    """Test handling of unusual pin names and special characters."""
    print("=" * 80)
    print("TEST: Unusual Pin Names")
    print("=" * 80)

    # Component with complex pin names
    u1_pins = [
        Pin("1", "VCC_3.3V", "3V3"),
        Pin("2", "UART0_TX", "UART_TX"),
        Pin("3", "I2C_SDA/GPIO4", "I2C_SDA"),
        Pin("4", "~RESET", "RESET_N"),
        Pin("5", "EN/PWDN", "ENABLE"),
    ]
    u1 = Component(
        refdes="U1",
        value="ESP32",
        footprint="QFN-48",
        mpn="ESP32-WROOM-32",
        page="Test_Page",
        description="WiFi/BT SoC",
        pins=u1_pins,
        location=(0, 0),
        properties={}
    )

    components = [u1]

    net1 = Net("3V3", {"Test_Page"}, [("U1", "1")])
    net2 = Net("UART_TX", {"Test_Page"}, [("U1", "2")])

    nets = [net1, net2]
    net_page_map = {"3V3": {"Test_Page"}, "UART_TX": {"Test_Page"}}

    dsl = emit_page_dsl(components, nets, net_page_map)
    print(dsl)
    print()


def test_no_connect_pins():
    """Test handling of no-connect pins."""
    print("=" * 80)
    print("TEST: No-Connect Pins")
    print("=" * 80)

    # Component with NC pins
    u1_pins = [
        Pin("1", "VCC", "3V3"),
        Pin("2", "OUT1", "SIGNAL1"),
        Pin("3", "NC", ""),  # No connect
        Pin("4", "NC", ""),  # No connect
        Pin("5", "GND", "GND"),
    ]
    u1 = Component(
        refdes="U1",
        value="REGULATOR",
        footprint="SOT-23-5",
        mpn="LP5907MFX-3.3",
        page="Test_Page",
        description="LDO Regulator with NC pins",
        pins=u1_pins,
        location=(0, 0),
        properties={}
    )

    components = [u1]

    net1 = Net("3V3", {"Test_Page"}, [("U1", "1")])
    net2 = Net("SIGNAL1", {"Test_Page"}, [("U1", "2")])
    net3 = Net("GND", {"Test_Page"}, [("U1", "5")])

    nets = [net1, net2, net3]
    net_page_map = {"3V3": {"Test_Page"}, "SIGNAL1": {"Test_Page"}, "GND": {"Test_Page"}}

    dsl = emit_page_dsl(components, nets, net_page_map)
    print(dsl)
    print()


def test_special_component_types():
    """Test handling of special component types."""
    print("=" * 80)
    print("TEST: Special Component Types")
    print("=" * 80)

    # Fuse
    f1_pins = [Pin("1", "", "VCC_IN"), Pin("2", "", "VCC_OUT")]
    f1 = Component(
        refdes="F1",
        value="500mA",
        footprint="1206",
        mpn="",
        page="Test_Page",
        description="Resettable Fuse",
        pins=f1_pins,
        location=(0, 0),
        properties={}
    )

    # Ferrite bead (IND type)
    fb1_pins = [Pin("1", "", "SIGNAL_IN"), Pin("2", "", "SIGNAL_OUT")]
    fb1 = Component(
        refdes="FB1",
        value="600 ohm @ 100MHz",
        footprint="0603",
        mpn="",
        page="Test_Page",
        description="Ferrite Bead",
        pins=fb1_pins,
        location=(0, 0),
        properties={}
    )

    # Crystal
    y1_pins = [Pin("1", "", "OSC_IN"), Pin("2", "", "OSC_OUT")]
    y1 = Component(
        refdes="Y1",
        value="16MHz",
        footprint="HC49",
        mpn="",
        page="Test_Page",
        description="Crystal Oscillator",
        pins=y1_pins,
        location=(0, 0),
        properties={}
    )

    # Button/Switch
    btn1_pins = [Pin("1", "", "BUTTON_IN"), Pin("2", "", "GND")]
    btn1 = Component(
        refdes="BTN1",
        value="TACT_SWITCH",
        footprint="6x6mm",
        mpn="",
        page="Test_Page",
        description="Tactile Switch",
        pins=btn1_pins,
        location=(0, 0),
        properties={}
    )

    components = [f1, fb1, y1, btn1]

    net1 = Net("VCC_IN", {"Test_Page"}, [("F1", "1")])
    net2 = Net("VCC_OUT", {"Test_Page"}, [("F1", "2")])

    nets = [net1, net2]
    net_page_map = {"VCC_IN": {"Test_Page"}, "VCC_OUT": {"Test_Page"}}

    dsl = emit_page_dsl(components, nets, net_page_map)
    print(dsl)
    print()

    # Verify types
    print("Component Types:")
    print(f"F1: {f1.derived_type()} (is_passive: {f1.is_passive()})")
    print(f"FB1: {fb1.derived_type()} (is_passive: {fb1.is_passive()})")
    print(f"Y1: {y1.derived_type()} (is_passive: {y1.is_passive()})")
    print(f"BTN1: {btn1.derived_type()} (is_passive: {btn1.is_passive()})")
    print()


def test_context_with_simple_primary():
    """Test context DSL when primary component is simple (passive)."""
    print("=" * 80)
    print("TEST: Context with Simple Primary Component")
    print("=" * 80)

    # Primary is a simple resistor
    r1_pins = [Pin("1", "", "3V3"), Pin("2", "", "LED_ANODE")]
    r1 = Component(
        refdes="R1",
        value="1k",
        footprint="0603",
        mpn="",
        page="Test_Page",
        description="Current limit resistor",
        pins=r1_pins,
        location=(0, 0),
        properties={}
    )

    # Neighbor LED
    led1_pins = [Pin("A", "", "LED_ANODE"), Pin("K", "", "GND")]
    led1 = Component(
        refdes="LED1",
        value="RED",
        footprint="0805",
        mpn="",
        page="Test_Page",
        description="Status LED",
        pins=led1_pins,
        location=(0, 0),
        properties={}
    )

    primary = [r1]
    neighbors = [led1]

    net1 = Net("LED_ANODE", {"Test_Page"}, [("R1", "2"), ("LED1", "A")])

    nets = [net1]

    dsl = emit_context_dsl(primary, neighbors, nets)
    print(dsl)
    print()


def test_natural_sort_key():
    """Test the natural sort key function."""
    print("=" * 80)
    print("TEST: Natural Sort Key Function")
    print("=" * 80)

    test_strings = ["10", "2", "1", "20", "3", "100", "A1", "A10", "A2", "B1"]
    sorted_strings = sorted(test_strings, key=_natural_sort_key)

    print("Original:", test_strings)
    print("Sorted:  ", sorted_strings)
    print("Expected: ['1', '2', '3', '10', '20', '100', 'A1', 'A2', 'A10', 'B1']")
    print()


if __name__ == "__main__":
    test_empty_fields()
    test_pin_sorting()
    test_multipart_components()
    test_unusual_pin_names()
    test_no_connect_pins()
    test_special_component_types()
    test_context_with_simple_primary()
    test_natural_sort_key()

    print("=" * 80)
    print("All edge case tests completed!")
    print("=" * 80)
