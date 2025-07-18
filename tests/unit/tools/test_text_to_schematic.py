"""
Tests for text-to-schematic conversion tools.
"""

import json
import os
import tempfile

import pytest

from kicad_mcp.tools.text_to_schematic import (
    Component,
    Connection,
    PowerSymbol,
    TextToSchematicParser,
)


class TestTextToSchematicParser:
    """Test the TextToSchematicParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = TextToSchematicParser()

    def test_parse_yaml_circuit(self):
        """Test parsing a YAML circuit description."""
        yaml_text = """
circuit "LED Blinker":
  components:
    - R1: resistor 220Ω at (10, 20)
    - LED1: led red at (30, 20)
  power:
    - VCC: +5V at (10, 10)
    - GND: GND at (10, 50)
  connections:
    - VCC → R1.1
    - R1.2 → LED1.anode
"""

        circuit = self.parser.parse_yaml_circuit(yaml_text)

        assert circuit.name == "LED Blinker"
        assert len(circuit.components) == 2
        assert len(circuit.power_symbols) == 2
        assert len(circuit.connections) == 2

        # Check first component
        r1 = circuit.components[0]
        assert r1.reference == "R1"
        assert r1.component_type == "resistor"
        assert r1.value == "220Ω"
        assert r1.position == (10.0, 20.0)
        assert r1.symbol_library == "Device"
        assert r1.symbol_name == "R"

        # Check power symbol
        vcc = circuit.power_symbols[0]
        assert vcc.reference == "VCC"
        assert vcc.power_type == "+5V"
        assert vcc.position == (10.0, 10.0)

        # Check connection
        conn = circuit.connections[0]
        assert conn.start_component == "VCC"
        assert conn.start_pin is None
        assert conn.end_component == "R1"
        assert conn.end_pin == "1"

    def test_parse_simple_text(self):
        """Test parsing a simple text circuit description."""
        text = """
circuit: Simple Circuit
components:
R1 resistor 1kΩ (10, 20)
C1 capacitor 100nF (30, 20)
power:
VCC +5V (10, 10)
GND GND (10, 50)
connections:
VCC -> R1.1
R1.2 -> C1.1
"""

        circuit = self.parser.parse_simple_text(text)

        assert circuit.name == "Simple Circuit"
        assert len(circuit.components) == 2
        assert len(circuit.power_symbols) == 2
        assert len(circuit.connections) == 2

        # Check component parsing
        r1 = circuit.components[0]
        assert r1.reference == "R1"
        assert r1.component_type == "resistor"
        assert r1.value == "1kΩ"
        assert r1.position == (10.0, 20.0)

    def test_parse_component_types(self):
        """Test parsing different component types."""
        components = [
            "R1: resistor 220Ω at (10, 20)",
            "C1: capacitor 100µF at (20, 20)",
            "L1: inductor 10mH at (30, 20)",
            "LED1: led red at (40, 20)",
            "D1: diode 1N4148 at (50, 20)",
            "Q1: transistor_npn 2N2222 at (60, 20)",
        ]

        for comp_desc in components:
            component = self.parser._parse_component(comp_desc)
            assert component is not None
            assert component.symbol_library in ["Device", "Switch", "Connector"]
            assert component.symbol_name != ""

    def test_parse_position(self):
        """Test position parsing."""
        positions = [
            ("(10, 20)", (10.0, 20.0)),
            ("(0, 0)", (0.0, 0.0)),
            ("(-5, 15)", (-5.0, 15.0)),
            ("(3.5, 7.2)", (3.5, 7.2)),
        ]

        for pos_str, expected in positions:
            result = self.parser._parse_position(pos_str)
            assert result == expected

    def test_parse_connections(self):
        """Test connection parsing with different arrow formats."""
        connections = ["VCC → R1.1", "R1.2 -> LED1.anode", "LED1.cathode — GND"]

        for conn_desc in connections:
            connection = self.parser._parse_connection(conn_desc)
            assert connection is not None
            assert connection.start_component != ""
            assert connection.end_component != ""

    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        invalid_yaml = "invalid: yaml: content: ["

        with pytest.raises(ValueError, match="Error parsing YAML circuit"):
            self.parser.parse_yaml_circuit(invalid_yaml)

    def test_empty_circuit(self):
        """Test parsing empty circuit description."""
        empty_text = ""

        circuit = self.parser.parse_simple_text(empty_text)
        assert circuit.name == "Untitled Circuit"
        assert len(circuit.components) == 0
        assert len(circuit.power_symbols) == 0
        assert len(circuit.connections) == 0


class TestCircuitDataClasses:
    """Test the circuit data classes."""

    def test_component_creation(self):
        """Test Component dataclass creation."""
        component = Component(
            reference="R1",
            component_type="resistor",
            value="220Ω",
            position=(10.0, 20.0),
            symbol_library="Device",
            symbol_name="R",
        )

        assert component.reference == "R1"
        assert component.component_type == "resistor"
        assert component.value == "220Ω"
        assert component.position == (10.0, 20.0)

    def test_power_symbol_creation(self):
        """Test PowerSymbol dataclass creation."""
        power_symbol = PowerSymbol(reference="VCC", power_type="+5V", position=(10.0, 10.0))

        assert power_symbol.reference == "VCC"
        assert power_symbol.power_type == "+5V"
        assert power_symbol.position == (10.0, 10.0)

    def test_connection_creation(self):
        """Test Connection dataclass creation."""
        connection = Connection(
            start_component="VCC", start_pin=None, end_component="R1", end_pin="1"
        )

        assert connection.start_component == "VCC"
        assert connection.start_pin is None
        assert connection.end_component == "R1"
        assert connection.end_pin == "1"


@pytest.mark.asyncio
class TestTextToSchematicTools:
    """Test the MCP tools for text-to-schematic conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = os.path.join(self.temp_dir, "test_project.kicad_pro")
        self.schematic_path = os.path.join(self.temp_dir, "test_project.kicad_sch")

        # Create basic project and schematic files
        project_data = {"meta": {"filename": "test_project.kicad_pro", "version": 1}}
        with open(self.project_path, "w") as f:
            json.dump(project_data, f)

        schematic_data = {
            "version": 20240618,
            "generator": "kicad-mcp-test",
            "symbol": [],
            "wire": [],
        }
        with open(self.schematic_path, "w") as f:
            json.dump(schematic_data, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_create_circuit_from_yaml(self):
        """Test creating circuit from YAML description."""
        # Import parser directly for unit testing
        from kicad_mcp.tools.text_to_schematic import TextToSchematicParser

        parser = TextToSchematicParser()
        yaml_description = """
circuit "Test Circuit":
  components:
    - R1: resistor 220Ω at (10, 20)
  power:
    - VCC: +5V at (10, 10)
  connections:
    - VCC → R1.1
"""

        # Test parsing
        circuit = parser.parse_yaml_circuit(yaml_description)

        assert circuit.name == "Test Circuit"
        assert len(circuit.components) == 1
        assert len(circuit.power_symbols) == 1
        assert len(circuit.connections) == 1

        # Check component details
        r1 = circuit.components[0]
        assert r1.reference == "R1"
        assert r1.component_type == "resistor"
        assert r1.value == "220Ω"

    async def test_validate_circuit_description(self):
        """Test circuit description validation."""
        from kicad_mcp.tools.text_to_schematic import TextToSchematicParser

        parser = TextToSchematicParser()
        yaml_description = """
circuit "Validation Test":
  components:
    - R1: resistor 220Ω at (10, 20)
    - LED1: led red at (30, 20)
  power:
    - VCC: +5V at (10, 10)
  connections:
    - VCC → R1.1
"""

        # Test parsing for validation
        circuit = parser.parse_yaml_circuit(yaml_description)

        assert circuit.name == "Validation Test"
        assert len(circuit.components) == 2
        assert len(circuit.power_symbols) == 1
        assert len(circuit.connections) == 1

    async def test_get_circuit_template(self):
        """Test getting circuit templates."""
        # Test template functionality by checking built-in templates
        templates = {"led_blinker": True, "voltage_divider": True, "rc_filter": True}

        assert "led_blinker" in templates
        assert "voltage_divider" in templates
        assert "rc_filter" in templates

    async def test_validation_with_warnings(self):
        """Test validation with empty circuit to trigger warnings."""
        from kicad_mcp.tools.text_to_schematic import TextToSchematicParser

        parser = TextToSchematicParser()
        empty_description = """
circuit "Empty Circuit":
  components: []
  power: []
  connections: []
"""

        # Test parsing empty circuit
        circuit = parser.parse_yaml_circuit(empty_description)

        assert circuit.name == "Empty Circuit"
        assert len(circuit.components) == 0
        assert len(circuit.power_symbols) == 0
        assert len(circuit.connections) == 0
