"""
Test suite for the Altium JSON adapter.

Tests the adapter's ability to transform Altium JSON format into the unified
data model, including edge cases like no-connect pins, missing fields, and
multi-pin components.
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from schematic_core.adapters.altium_json import AltiumJSONAdapter
from schematic_core.models import Component, Pin, Net


def test_basic_component_parsing():
    """Test basic component transformation from Altium JSON."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "R1",
                "description": "RES 10K 0805",
                "footprint": "RESC1608X55N",
                "sheet": "C:\\Users\\test\\Main.SchDoc",
                "schematic_x": 1000,
                "schematic_y": 2000,
                "parameters": {
                    "PN": "RC0805FR-0710KL",
                    "MFG": "Yageo",
                    "Comment": "10k"
                },
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "2", "net": "NetR1_2"}
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    assert len(components) == 1
    comp = components[0]

    assert comp.refdes == "R1"
    assert comp.value == "10k"  # From Comment
    assert comp.mpn == "RC0805FR-0710KL"
    assert comp.footprint == "RESC1608X55N"
    assert comp.page == "Main.SchDoc"  # Filename only
    assert comp.location == (1000, 2000)
    assert len(comp.pins) == 2
    assert comp.properties["MFG"] == "Yageo"

    print("[PASS] Basic component parsing test passed")


def test_pin_transformation():
    """Test pin data transformation including semantic vs numeric names."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "Q1",
                "description": "MOSFET N-CH",
                "footprint": "SOIC-8",
                "sheet": "Power.SchDoc",
                "parameters": {
                    "Comment": "SI4459BDY"
                },
                "pins": [
                    {"name": "S", "net": "VOUT"},     # Semantic
                    {"name": "G", "net": "GATE"},     # Semantic
                    {"name": "D", "net": "VIN"},      # Semantic
                    {"name": "1", "net": "TEST"}      # Numeric
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    pins = components[0].pins
    assert len(pins) == 4

    # Semantic pins should have name populated
    s_pin = next(p for p in pins if p.designator == "S")
    assert s_pin.name == "S"
    assert s_pin.net == "VOUT"

    g_pin = next(p for p in pins if p.designator == "G")
    assert g_pin.name == "G"
    assert g_pin.net == "GATE"

    # Numeric pin should have empty name
    pin_1 = next(p for p in pins if p.designator == "1")
    assert pin_1.name == ""
    assert pin_1.net == "TEST"

    print("[PASS] Pin transformation test passed")


def test_no_connect_pins():
    """Test handling of no-connect pins (empty net string)."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "LED1",
                "description": "LED RGB",
                "footprint": "LED-3528",
                "sheet": "Display.SchDoc",
                "parameters": {
                    "Comment": "RGB LED"
                },
                "pins": [
                    {"name": "1", "net": ""},          # No connect
                    {"name": "2", "net": "VCC"},
                    {"name": "3", "net": ""},          # No connect
                    {"name": "4", "net": "LED_DATA"}
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    pins = components[0].pins
    pin_1 = next(p for p in pins if p.designator == "1")
    pin_3 = next(p for p in pins if p.designator == "3")

    # Empty net should become "NC"
    assert pin_1.net == "NC"
    assert pin_3.net == "NC"

    print("[PASS] No-connect pins test passed")


def test_net_building():
    """Test building nets from component pin connectivity."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "U1",
                "sheet": "Main.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "10", "net": "GND"}
                ]
            },
            {
                "designator": "C1",
                "sheet": "Main.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "2", "net": "GND"}
                ]
            },
            {
                "designator": "R1",
                "sheet": "Power.SchDoc",
                "parameters": {},
                "pins": [
                    {"name": "1", "net": "VCC"},
                    {"name": "2", "net": "OUTPUT"}
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    nets = adapter.get_nets()

    # Should have 3 nets: VCC, GND, OUTPUT
    assert len(nets) == 3

    # Find VCC net
    vcc_net = next(n for n in nets if n.name == "VCC")
    assert len(vcc_net.members) == 3  # U1.1, C1.1, R1.1
    assert ("U1", "1") in vcc_net.members
    assert ("C1", "1") in vcc_net.members
    assert ("R1", "1") in vcc_net.members

    # VCC spans 2 pages
    assert len(vcc_net.pages) == 2
    assert "Main.SchDoc" in vcc_net.pages
    assert "Power.SchDoc" in vcc_net.pages

    # GND only on Main.SchDoc
    gnd_net = next(n for n in nets if n.name == "GND")
    assert len(gnd_net.pages) == 1
    assert "Main.SchDoc" in gnd_net.pages

    print("[PASS] Net building test passed")


def test_missing_fields():
    """Test graceful handling of missing optional fields."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "U1",
                "sheet": "Main.SchDoc",
                # Missing: description, footprint, parameters, location
                "pins": []
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    comp = components[0]
    assert comp.refdes == "U1"
    assert comp.value == ""  # No Comment or PN
    assert comp.footprint == ""
    assert comp.mpn == ""
    assert comp.description == ""
    assert comp.location == (0, 0)  # Default location
    assert len(comp.pins) == 0
    assert len(comp.properties) == 0

    print("[PASS] Missing fields test passed")


def test_value_fallback_logic():
    """Test component value extraction with fallback from Comment to PN."""
    # Test 1: Comment exists
    json_data_1 = json.dumps({
        "components": [{
            "designator": "R1",
            "sheet": "Main.SchDoc",
            "parameters": {
                "Comment": "10k",
                "PN": "RC0805FR-0710KL"
            },
            "pins": []
        }]
    })
    adapter1 = AltiumJSONAdapter(json_data_1)
    adapter1.fetch_raw_data()
    assert adapter1.get_components()[0].value == "10k"

    # Test 2: No Comment, use PN
    json_data_2 = json.dumps({
        "components": [{
            "designator": "R1",
            "sheet": "Main.SchDoc",
            "parameters": {
                "PN": "RC0805FR-0710KL"
            },
            "pins": []
        }]
    })
    adapter2 = AltiumJSONAdapter(json_data_2)
    adapter2.fetch_raw_data()
    assert adapter2.get_components()[0].value == "RC0805FR-0710KL"

    # Test 3: No Comment or PN
    json_data_3 = json.dumps({
        "components": [{
            "designator": "R1",
            "sheet": "Main.SchDoc",
            "parameters": {},
            "pins": []
        }]
    })
    adapter3 = AltiumJSONAdapter(json_data_3)
    adapter3.fetch_raw_data()
    assert adapter3.get_components()[0].value == ""

    print("[PASS] Value fallback logic test passed")


def test_multipin_same_net():
    """Test components with multiple pins on the same net (e.g., MOSFET with 4 source pins)."""
    json_data = json.dumps({
        "components": [
            {
                "designator": "Q1",
                "sheet": "Power.SchDoc",
                "parameters": {"Comment": "MOSFET"},
                "pins": [
                    {"name": "S", "net": "VOUT"},
                    {"name": "S", "net": "VOUT"},
                    {"name": "S", "net": "VOUT"},
                    {"name": "G", "net": "GATE"},
                    {"name": "D", "net": "VIN"},
                    {"name": "D", "net": "VIN"}
                ]
            }
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()

    # All pins should be preserved
    components = adapter.get_components()
    assert len(components[0].pins) == 6

    # Net should have all 6 connections
    nets = adapter.get_nets()
    vout_net = next(n for n in nets if n.name == "VOUT")
    vin_net = next(n for n in nets if n.name == "VIN")
    gate_net = next(n for n in nets if n.name == "GATE")

    # VOUT has 3 S pins
    assert len([m for m in vout_net.members if m[0] == "Q1"]) == 3
    # VIN has 2 D pins
    assert len([m for m in vin_net.members if m[0] == "Q1"]) == 2
    # GATE has 1 G pin
    assert len([m for m in gate_net.members if m[0] == "Q1"]) == 1

    print("[PASS] Multi-pin same net test passed")


def test_filename_extraction():
    """Test path extraction for various path formats."""
    json_data = json.dumps({
        "components": [
            {"designator": "U1", "sheet": "C:\\Users\\test\\project\\Main.SchDoc", "parameters": {}, "pins": []},
            {"designator": "U2", "sheet": "/home/user/project/Power.SchDoc", "parameters": {}, "pins": []},
            {"designator": "U3", "sheet": "Simple.SchDoc", "parameters": {}, "pins": []},
            {"designator": "U4", "sheet": "", "parameters": {}, "pins": []}
        ]
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    assert components[0].page == "Main.SchDoc"
    assert components[1].page == "Power.SchDoc"
    assert components[2].page == "Simple.SchDoc"
    assert components[3].page == ""

    print("[PASS] Filename extraction test passed")


def test_empty_design():
    """Test handling of empty component list."""
    json_data = json.dumps({
        "components": []
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()
    nets = adapter.get_nets()

    assert len(components) == 0
    assert len(nets) == 0

    print("[PASS] Empty design test passed")


def test_malformed_json():
    """Test error handling for malformed JSON."""
    json_data = "{ this is not valid json"

    adapter = AltiumJSONAdapter(json_data)
    try:
        adapter.fetch_raw_data()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid JSON" in str(e)

    print("[PASS] Malformed JSON test passed")


def test_missing_components_key():
    """Test handling of JSON without components key."""
    json_data = json.dumps({
        "metadata": {"project": "test"}
    })

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()

    # Should default to empty list
    assert len(components) == 0

    print("[PASS] Missing components key test passed")


def test_sample_json_file():
    """Test with the actual sample JSON file."""
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "altium_sample.json"
    )

    if not os.path.exists(sample_path):
        print("[SKIP] Sample JSON file not found, skipping")
        return

    with open(sample_path, 'r') as f:
        json_data = f.read()

    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()
    nets = adapter.get_nets()

    # Basic sanity checks
    assert len(components) > 0, "Sample should have components"
    assert len(nets) > 0, "Sample should have nets"

    # Check a specific component (U1 from sample)
    u1 = next((c for c in components if c.refdes == "U1"), None)
    if u1:
        assert u1.description == "IC GATE DRVR N-CH MOSFET 16 MSOP"
        assert u1.mpn == "LTC7003EMSE#TRPBF"
        assert len(u1.pins) == 17

    # Check net building worked
    gnd_net = next((n for n in nets if n.name == "GND"), None)
    if gnd_net:
        assert len(gnd_net.members) > 0

    print("[PASS] Sample JSON file test passed")


if __name__ == "__main__":
    print("Running Altium JSON Adapter Tests\n")

    test_basic_component_parsing()
    test_pin_transformation()
    test_no_connect_pins()
    test_net_building()
    test_missing_fields()
    test_value_fallback_logic()
    test_multipin_same_net()
    test_filename_extraction()
    test_empty_design()
    test_malformed_json()
    test_missing_components_key()
    test_sample_json_file()

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
