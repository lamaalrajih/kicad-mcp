"""
Integration test for Altium JSON adapter with the unified data model.

This test verifies that the adapter produces Component and Net objects that
correctly implement all the derived properties and methods defined in the
unified data model (Component.derived_type(), Component.is_complex(),
Net.is_global(), etc.).
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from schematic_core.adapters.altium_json import AltiumJSONAdapter


def test_component_derived_properties():
    """Test that Component derived properties work correctly."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "R1",
                "sheet": "Main.SchDoc",
                "parameters": {"Comment": "10k"},
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "2", "net": "GND"}
                ]
            },
            {
                "designator": "U1",
                "sheet": "Main.SchDoc",
                "parameters": {"Comment": "STM32F407"},
                "pins": [
                    {"name": str(i), "net": f"NET{i}"} for i in range(1, 11)
                ]
            },
            {
                "designator": "Q1",
                "sheet": "Main.SchDoc",
                "parameters": {"Comment": "MOSFET"},
                "pins": [
                    {"name": "S", "net": "VOUT"},
                    {"name": "G", "net": "GATE"},
                    {"name": "D", "net": "VIN"}
                ]
            },
            {
                "designator": "C1",
                "sheet": "Main.SchDoc",
                "parameters": {"Comment": "10u"},
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "2", "net": "GND"}
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    # Test derived_type()
    r1 = next(c for c in components if c.refdes == "R1")
    assert r1.derived_type() == "RES"

    u1 = next(c for c in components if c.refdes == "U1")
    assert u1.derived_type() == "IC"

    q1 = next(c for c in components if c.refdes == "Q1")
    assert q1.derived_type() == "TRANSISTOR"

    c1 = next(c for c in components if c.refdes == "C1")
    assert c1.derived_type() == "CAP"

    # Test is_complex()
    assert not r1.is_complex()  # 2 pins, no semantic names
    assert u1.is_complex()      # >4 pins
    assert q1.is_complex()      # semantic pin names (S, G, D)
    assert not c1.is_complex()  # 2 pins, no semantic names

    # Test is_passive()
    assert r1.is_passive()      # RES
    assert not u1.is_passive()  # IC
    assert not q1.is_passive()  # TRANSISTOR
    assert c1.is_passive()      # CAP

    print("[PASS] Component derived properties test passed")


def test_net_derived_properties():
    """Test that Net derived properties work correctly."""
    json_data = json.dumps({
        "components": [
            # GND on many components (should be global)
            {
                "designator": f"U{i}",
                "sheet": f"Page{i % 3}.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "GND"},
                    {"name": "2", "net": "VCC"}
                ]
            } for i in range(1, 21)  # 20 components
        ] + [
            # Simple signal net
            {
                "designator": "U100",
                "sheet": "Main.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "UART_TX"},
                    {"name": "2", "net": "UART_RX"}
                ]
            },
            {
                "designator": "U101",
                "sheet": "IO.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "UART_TX"},
                    {"name": "2", "net": "UART_RX"}
                ]
            },
            # 3V3 power net
            {
                "designator": "U102",
                "sheet": "Power1.SchDoc",
                "parameters": {},
                "pins": [{"name": "1", "net": "3V3"}]
            },
            {
                "designator": "U103",
                "sheet": "Power2.SchDoc",
                "parameters": {},
                "pins": [{"name": "1", "net": "3V3"}]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    nets = adapter.get_nets()

    # Test GND (should be global - many connections)
    gnd_net = next(n for n in nets if n.name == "GND")
    assert gnd_net.is_global()  # >15 members
    assert gnd_net.is_inter_page()  # Spans Page0, Page1, Page2

    # Test VCC (should be global - many connections)
    vcc_net = next(n for n in nets if n.name == "VCC")
    assert vcc_net.is_global()  # >15 members

    # Test UART_TX (should not be global)
    uart_net = next(n for n in nets if n.name == "UART_TX")
    assert not uart_net.is_global()  # Only 2 connections
    assert uart_net.is_inter_page()  # Spans Main.SchDoc and IO.SchDoc

    # Test 3V3 (should be global by name pattern)
    v3_net = next(n for n in nets if n.name == "3V3")
    assert v3_net.is_global()  # Matches voltage pattern
    assert v3_net.is_inter_page()  # Spans 2 pages

    print("[PASS] Net derived properties test passed")


def test_pin_data_integrity():
    """Test that Pin objects preserve all data correctly."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "U1",
                "sheet": "Main.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "VCC", "net": "VCC"},
                    {"name": "TX", "net": "UART_TX"},
                    {"name": "", "net": ""},  # Empty pin name and net
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    u1 = components[0]
    pins = u1.pins

    # Pin 1 (numeric)
    pin1 = next(p for p in pins if p.designator == "1")
    assert pin1.designator == "1"
    assert pin1.name == ""  # Not semantic
    assert pin1.net == "VCC"

    # Pin VCC (semantic)
    pin_vcc = next(p for p in pins if p.designator == "VCC")
    assert pin_vcc.designator == "VCC"
    assert pin_vcc.name == "VCC"  # Semantic
    assert pin_vcc.net == "VCC"

    # Pin TX (semantic)
    pin_tx = next(p for p in pins if p.designator == "TX")
    assert pin_tx.designator == "TX"
    assert pin_tx.name == "TX"  # Semantic
    assert pin_tx.net == "UART_TX"

    # Empty pin (no-connect)
    pin_empty = next(p for p in pins if p.designator == "")
    assert pin_empty.designator == ""
    assert pin_empty.name == ""
    assert pin_empty.net == "NC"  # Empty net becomes NC

    print("[PASS] Pin data integrity test passed")


def test_properties_extraction():
    """Test that component properties are correctly extracted."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "R1",
                "sheet": "Main.SchDoc",
                "parameters": {
                    "PN": "RC0805FR-0710KL",
                    "Comment": "10k",
                    "MFG": "Yageo",
                    "Tolerance": "1%",
                    "Power": "1/8W",
                    "Voltage": "150V"
                },
                "pins": []
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    r1 = components[0]

    # Check mapped fields
    assert r1.mpn == "RC0805FR-0710KL"  # From PN
    assert r1.value == "10k"  # From Comment

    # Check properties dict (should not include PN or Comment)
    assert "PN" not in r1.properties
    assert "Comment" not in r1.properties
    assert r1.properties["MFG"] == "Yageo"
    assert r1.properties["Tolerance"] == "1%"
    assert r1.properties["Power"] == "1/8W"
    assert r1.properties["Voltage"] == "150V"

    print("[PASS] Properties extraction test passed")


def test_location_data():
    """Test that location data is captured correctly."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "U1",
                "sheet": "Main.SchDoc",
                "schematic_x": 1000,
                "schematic_y": 2000,
                "parameters": {},
                "pins": []
            },
            {
                "designator": "U2",
                "sheet": "Main.SchDoc",
                # Missing location
                "parameters": {},
                "pins": []
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    u1 = next(c for c in components if c.refdes == "U1")
    assert u1.location == (1000, 2000)

    u2 = next(c for c in components if c.refdes == "U2")
    assert u2.location == (0, 0)  # Default

    print("[PASS] Location data test passed")


def test_multi_page_design():
    """Test handling of design spanning multiple pages."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "U1",
                "sheet": "C:\\Projects\\Main.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "SIGNAL_A"},
                    {"name": "2", "net": "GND"}
                ]
            },
            {
                "designator": "U2",
                "sheet": "/home/user/project/Power.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "SIGNAL_A"},
                    {"name": "2", "net": "GND"}
                ]
            },
            {
                "designator": "U3",
                "sheet": "IO.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "SIGNAL_B"},
                    {"name": "2", "net": "GND"}
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()
    nets = adapter.get_nets()

    # Check page names extracted correctly
    u1 = next(c for c in components if c.refdes == "U1")
    u2 = next(c for c in components if c.refdes == "U2")
    u3 = next(c for c in components if c.refdes == "U3")

    assert u1.page == "Main.SchDoc"
    assert u2.page == "Power.SchDoc"
    assert u3.page == "IO.SchDoc"

    # Check net page tracking
    signal_a = next(n for n in nets if n.name == "SIGNAL_A")
    assert len(signal_a.pages) == 2
    assert "Main.SchDoc" in signal_a.pages
    assert "Power.SchDoc" in signal_a.pages

    gnd_net = next(n for n in nets if n.name == "GND")
    assert len(gnd_net.pages) == 3
    assert "Main.SchDoc" in gnd_net.pages
    assert "Power.SchDoc" in gnd_net.pages
    assert "IO.SchDoc" in gnd_net.pages

    signal_b = next(n for n in nets if n.name == "SIGNAL_B")
    assert len(signal_b.pages) == 1
    assert "IO.SchDoc" in signal_b.pages

    print("[PASS] Multi-page design test passed")


if __name__ == "__main__":
    print("Running Altium JSON Adapter Integration Tests\n")

    test_component_derived_properties()
    test_net_derived_properties()
    test_pin_data_integrity()
    test_properties_extraction()
    test_location_data()
    test_multi_page_design()

    print("\n" + "="*50)
    print("All integration tests passed!")
    print("="*50)
