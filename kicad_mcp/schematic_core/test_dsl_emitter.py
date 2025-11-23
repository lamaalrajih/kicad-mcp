"""
Test suite for DSL Emitter v0.3

This module tests the DSL emitter with realistic data to verify
correct formatting according to the specification.
"""

from models import Component, Net, Pin
from dsl_emitter import emit_page_dsl, emit_context_dsl


def create_test_data():
    """Create realistic test data representing a typical schematic page."""

    # Complex IC - STM32 MCU
    u1_pins = [
        Pin("1", "VDD", "3V3"),
        Pin("2", "PA0", "ADC_IN"),
        Pin("10", "GND", "GND"),
        Pin("22", "PA9_TX", "UART_TX"),
        Pin("23", "PA10_RX", "UART_RX"),
        Pin("50", "VSS", "GND"),
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
        properties={}
    )

    # Simple regulator - still complex due to named pins
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
        page="Main_Sheet",
        description="Linear Regulator 3.3V",
        pins=u2_pins,
        location=(5100, 4000),
        properties={}
    )

    # Connector - complex due to many pins
    j1_pins = [
        Pin("1", "VCC", "VCC"),
        Pin("2", "GND", "GND"),
        Pin("3", "TX", "UART_TX"),
        Pin("4", "RX", "UART_RX"),
        Pin("5", "GPIO1", "GPIO1"),
    ]
    j1 = Component(
        refdes="J1",
        value="CONN_10",
        footprint="HDR-1x10",
        mpn="",
        page="Main_Sheet",
        description="10-pin header",
        pins=j1_pins,
        location=(7600, 5200),
        properties={}
    )

    # Simple resistor - will NOT get COMP block
    r1_pins = [
        Pin("1", "", "UART_TX"),
        Pin("2", "", "UART_TX_BUF"),
    ]
    r1 = Component(
        refdes="R1",
        value="10k",
        footprint="0603",
        mpn="RC0603FR-0710KL",
        page="Main_Sheet",
        description="RES SMD 10K OHM 1% 1/10W 0603",
        pins=r1_pins,
        location=(6500, 3800),
        properties={}
    )

    # Simple resistor
    r2_pins = [
        Pin("1", "", "UART_RX"),
        Pin("2", "", "UART_RX_BUF"),
    ]
    r2 = Component(
        refdes="R2",
        value="10k",
        footprint="0603",
        mpn="RC0603FR-0710KL",
        page="Main_Sheet",
        description="RES SMD 10K OHM 1% 1/10W 0603",
        pins=r2_pins,
        location=(6500, 4000),
        properties={}
    )

    # Simple resistor
    r3_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "ADC_IN"),
    ]
    r3 = Component(
        refdes="R3",
        value="100k",
        footprint="0603",
        mpn="RC0603FR-07100KL",
        page="Main_Sheet",
        description="RES SMD 100K OHM 1% 1/10W 0603",
        pins=r3_pins,
        location=(5800, 3700),
        properties={}
    )

    # Simple capacitors
    c1_pins = [
        Pin("1", "", "VCC"),
        Pin("2", "", "GND"),
    ]
    c1 = Component(
        refdes="C1",
        value="10u",
        footprint="0805",
        mpn="CL21A106KAYNNNE",
        page="Main_Sheet",
        description="CAP CER 10UF 25V X5R 0805",
        pins=c1_pins,
        location=(4800, 4200),
        properties={}
    )

    c2_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "GND"),
    ]
    c2 = Component(
        refdes="C2",
        value="10u",
        footprint="0805",
        mpn="CL21A106KAYNNNE",
        page="Main_Sheet",
        description="CAP CER 10UF 25V X5R 0805",
        pins=c2_pins,
        location=(5400, 4200),
        properties={}
    )

    c3_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "GND"),
    ]
    c3 = Component(
        refdes="C3",
        value="100n",
        footprint="0603",
        mpn="CL10B104KB8NNNC",
        page="Main_Sheet",
        description="CAP CER 0.1UF 50V X7R 0603",
        pins=c3_pins,
        location=(6200, 3500),
        properties={}
    )

    components = [u1, u2, j1, r1, r2, r3, c1, c2, c3]

    # Create nets
    # VCC - global net (power pattern + many connections)
    vcc_members = [
        ("U2", "3"),
        ("J1", "1"),
        ("C1", "1"),
        # Add more to make it exceed threshold
        ("U100", "1"), ("U101", "1"), ("U102", "1"), ("U103", "1"),
        ("U104", "1"), ("U105", "1"), ("U106", "1"), ("U107", "1"),
        ("U108", "1"), ("U109", "1"), ("U110", "1"), ("U111", "1"),
    ]
    vcc = Net(
        name="VCC",
        pages={"Main_Sheet", "Power_Sheet", "Connector_Page"},
        members=vcc_members
    )

    # 3V3 - local power net
    v3v3_members = [
        ("U2", "2"),
        ("U1", "1"),
        ("C2", "1"),
        ("C3", "1"),
        ("R3", "1"),
    ]
    v3v3 = Net(
        name="3V3",
        pages={"Main_Sheet"},
        members=v3v3_members
    )

    # GND - global net
    gnd_members = [
        ("U1", "10"),
        ("U1", "50"),
        ("U2", "1"),
        ("J1", "2"),
        ("C1", "2"),
        ("C2", "2"),
        ("C3", "2"),
        # Add more
        ("U100", "GND"), ("U101", "GND"), ("U102", "GND"), ("U103", "GND"),
        ("U104", "GND"), ("U105", "GND"), ("U106", "GND"), ("U107", "GND"),
        ("U108", "GND"), ("U109", "GND"), ("U110", "GND"), ("U111", "GND"),
        ("U112", "GND"), ("U113", "GND"), ("U114", "GND"), ("U115", "GND"),
        ("U116", "GND"), ("U117", "GND"), ("U118", "GND"), ("U119", "GND"),
    ]
    gnd = Net(
        name="GND",
        pages={"Main_Sheet", "Power_Sheet", "Connector_Page", "Sensor_Page"},
        members=gnd_members
    )

    # UART_TX - inter-page net
    uart_tx_members = [
        ("U1", "22"),
        ("R1", "1"),
        ("J1", "3"),
    ]
    uart_tx = Net(
        name="UART_TX",
        pages={"Main_Sheet", "Connector_Page"},
        members=uart_tx_members
    )

    # UART_RX - inter-page net
    uart_rx_members = [
        ("U1", "23"),
        ("R2", "1"),
        ("J1", "4"),
    ]
    uart_rx = Net(
        name="UART_RX",
        pages={"Main_Sheet", "Connector_Page"},
        members=uart_rx_members
    )

    # ADC_IN - local net
    adc_in_members = [
        ("U1", "2"),
        ("R3", "2"),
    ]
    adc_in = Net(
        name="ADC_IN",
        pages={"Main_Sheet"},
        members=adc_in_members
    )

    nets = [vcc, v3v3, gnd, uart_tx, uart_rx, adc_in]

    # Net page map
    net_page_map = {
        "VCC": {"Main_Sheet", "Power_Sheet", "Connector_Page"},
        "3V3": {"Main_Sheet"},
        "GND": {"Main_Sheet", "Power_Sheet", "Connector_Page", "Sensor_Page"},
        "UART_TX": {"Main_Sheet", "Connector_Page"},
        "UART_RX": {"Main_Sheet", "Connector_Page"},
        "ADC_IN": {"Main_Sheet"},
    }

    return components, nets, net_page_map


def test_page_dsl():
    """Test emit_page_dsl function."""
    print("=" * 80)
    print("TEST: Page DSL Output")
    print("=" * 80)

    components, nets, net_page_map = create_test_data()

    # Generate DSL
    dsl = emit_page_dsl(components, nets, net_page_map)

    print(dsl)
    print("\n")


def test_context_dsl():
    """Test emit_context_dsl function."""
    print("=" * 80)
    print("TEST: Context DSL Output")
    print("=" * 80)

    components, nets, net_page_map = create_test_data()

    # Create context around U1
    primary = [c for c in components if c.refdes == "U1"]

    # Find neighbors (components connected via nets to U1)
    u1_nets = set()
    for pin in primary[0].pins:
        u1_nets.add(pin.net)

    neighbor_refdes = set()
    for net in nets:
        for refdes, pin_designator in net.members:
            if net.name in u1_nets and refdes != "U1":
                neighbor_refdes.add(refdes)

    neighbors = [c for c in components if c.refdes in neighbor_refdes]

    # Filter nets to those involving U1
    context_nets = [n for n in nets if n.name in u1_nets]

    # Generate DSL
    dsl = emit_context_dsl(primary, neighbors, context_nets)

    print(dsl)
    print("\n")


def test_component_classification():
    """Test component classification logic."""
    print("=" * 80)
    print("TEST: Component Classification")
    print("=" * 80)

    components, _, _ = create_test_data()

    for comp in components:
        comp_type = comp.derived_type()
        is_complex = comp.is_complex()
        is_passive = comp.is_passive()

        print(f"{comp.refdes:6} | Type: {comp_type:10} | Complex: {is_complex:5} | Passive: {is_passive:5}")

    print("\n")


def test_net_classification():
    """Test net classification logic."""
    print("=" * 80)
    print("TEST: Net Classification")
    print("=" * 80)

    _, nets, _ = create_test_data()

    for net in nets:
        is_global = net.is_global()
        is_inter_page = net.is_inter_page()
        connection_count = len(net.members)
        page_count = len(net.pages)

        print(f"{net.name:15} | Global: {is_global:5} | Inter-page: {is_inter_page:5} | "
              f"Connections: {connection_count:3} | Pages: {page_count}")

    print("\n")


if __name__ == "__main__":
    test_component_classification()
    test_net_classification()
    test_page_dsl()
    test_context_dsl()
