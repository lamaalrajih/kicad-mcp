"""
Example DSL Output Generator

This script generates example DSL output that matches the format
specified in Appendix A of SCHEMATIC_CORE_SPEC.md
"""

from models import Component, Net, Pin
from dsl_emitter import emit_page_dsl


def create_spec_example_data():
    """
    Create data matching the Appendix A example from the specification.
    This demonstrates the exact format expected by the v0.3 DSL.
    """

    # U1 - STM32 MCU (Complex IC)
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

    # U2 - LDO Regulator (Complex IC)
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

    # J1 - Connector (Complex due to >4 pins)
    j1_pins = [
        Pin("1", "VCC", "VCC"),
        Pin("2", "GND", "GND"),
        Pin("3", "TX", "UART_TX"),
        Pin("4", "RX", "UART_RX"),
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

    # R1, R2, R3 - Simple resistors (NO COMP blocks)
    r1_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "UART_TX"),
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

    r2_pins = [
        Pin("1", "", "3V3"),
        Pin("2", "", "UART_RX"),
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

    # C1, C2, C3 - Simple capacitors (NO COMP blocks)
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

    # Create nets - matching Appendix A example
    # VCC - Global net (appears on multiple pages + power pattern)
    vcc_members = [
        ("U2", "3"),
        ("J1", "1"),
        ("C1", "1"),
        # Add dummy components to exceed threshold
        ("U100", "1"), ("U101", "1"), ("U102", "1"), ("U103", "1"),
        ("U104", "1"), ("U105", "1"), ("U106", "1"), ("U107", "1"),
        ("U108", "1"), ("U109", "1"), ("U110", "1"), ("U111", "1"),
    ]
    vcc = Net(
        name="VCC",
        pages={"Main_Sheet", "Power_Sheet", "Connector_Page", "Sensor_Page"},
        members=vcc_members
    )

    # 3V3 - Power net (identified by name pattern)
    v3v3_members = [
        ("U2", "2"),
        ("U1", "1"),
        ("U1", "50"),
        ("C2", "1"),
        ("C3", "1"),
        ("R1", "1"),
    ]
    v3v3 = Net(
        name="3V3",
        pages={"Main_Sheet"},
        members=v3v3_members
    )

    # GND - Global net
    gnd_members = [
        ("U1", "10"),
        ("U2", "1"),
        ("J1", "2"),
        ("C1", "2"),
        ("C2", "2"),
        ("C3", "2"),
        # Add many more connections
        ("U100", "GND"), ("U101", "GND"), ("U102", "GND"), ("U103", "GND"),
        ("U104", "GND"), ("U105", "GND"), ("U106", "GND"), ("U107", "GND"),
        ("U108", "GND"), ("U109", "GND"), ("U110", "GND"), ("U111", "GND"),
        ("U112", "GND"), ("U113", "GND"), ("U114", "GND"), ("U115", "GND"),
        ("U116", "GND"), ("U117", "GND"), ("U118", "GND"), ("U119", "GND"),
        ("U120", "GND"), ("U121", "GND"), ("U122", "GND"), ("U123", "GND"),
        ("U124", "GND"), ("U125", "GND"), ("U126", "GND"),
    ]
    gnd = Net(
        name="GND",
        pages={"Main_Sheet", "Power_Sheet", "Connector_Page", "Sensor_Page"},
        members=gnd_members
    )

    # UART_TX - Inter-page net
    uart_tx_members = [
        ("U1", "22"),
        ("R1", "2"),
        ("J1", "3"),
    ]
    uart_tx = Net(
        name="UART_TX",
        pages={"Main_Sheet", "Connector_Page"},
        members=uart_tx_members
    )

    # UART_RX - Inter-page net
    uart_rx_members = [
        ("U1", "23"),
        ("R2", "2"),
        ("J1", "4"),
    ]
    uart_rx = Net(
        name="UART_RX",
        pages={"Main_Sheet", "Connector_Page"},
        members=uart_rx_members
    )

    # ADC_IN - Local net
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
        "VCC": {"Main_Sheet", "Power_Sheet", "Connector_Page", "Sensor_Page"},
        "3V3": {"Main_Sheet"},
        "GND": {"Main_Sheet", "Power_Sheet", "Connector_Page", "Sensor_Page"},
        "UART_TX": {"Main_Sheet", "Connector_Page"},
        "UART_RX": {"Main_Sheet", "Connector_Page"},
        "ADC_IN": {"Main_Sheet"},
    }

    return components, nets, net_page_map


if __name__ == "__main__":
    print("=" * 80)
    print("DSL v0.3 Output - Matching Appendix A Format")
    print("=" * 80)
    print()

    components, nets, net_page_map = create_spec_example_data()
    dsl = emit_page_dsl(components, nets, net_page_map)

    print(dsl)
    print()
    print("=" * 80)
    print("Key Features Demonstrated:")
    print("=" * 80)
    print("1. Complex components (U1, U2, J1) have full DEF/COMP blocks")
    print("2. Simple passives (R1-R3, C1-C3) have NO COMP blocks")
    print("3. Pin references use parentheses for named pins: U1.22(PA9_TX)")
    print("4. Pin references without names: R1.1, C1.2")
    print("5. Global nets show 'LINKS: ALL_PAGES'")
    print("6. Inter-page nets show 'LINKS: Main_Sheet, Connector_Page'")
    print("7. Global nets truncate to first 10 connections + '(+ N others)'")
    print("8. All sections sorted alphabetically (refdes, nets, pins)")
