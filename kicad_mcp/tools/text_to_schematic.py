"""
Text-to-schematic conversion tools for KiCad projects.

Provides MermaidJS-like syntax for describing circuits that get converted to KiCad schematics.
"""

from dataclasses import dataclass
import os
import re
from typing import Any

from fastmcp import Context, FastMCP
import yaml

from kicad_mcp.utils.boundary_validator import BoundaryValidator
from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils.sexpr_generator import SExpressionGenerator


@dataclass
class Component:
    """Represents a circuit component."""

    reference: str
    component_type: str
    value: str
    position: tuple[float, float]
    symbol_library: str = "Device"
    symbol_name: str = ""


@dataclass
class PowerSymbol:
    """Represents a power symbol."""

    reference: str
    power_type: str
    position: tuple[float, float]


@dataclass
class Connection:
    """Represents a wire connection between components."""

    start_component: str
    start_pin: str | None
    end_component: str
    end_pin: str | None


@dataclass
class Circuit:
    """Represents a complete circuit description."""

    name: str
    components: list[Component]
    power_symbols: list[PowerSymbol]
    connections: list[Connection]


class TextToSchematicParser:
    """Parser for text-based circuit descriptions."""

    # Component type mappings to KiCad symbols
    COMPONENT_SYMBOLS = {
        "resistor": ("Device", "R"),
        "capacitor": ("Device", "C"),
        "inductor": ("Device", "L"),
        "led": ("Device", "LED"),
        "diode": ("Device", "D"),
        "transistor_npn": ("Device", "Q_NPN_CBE"),
        "transistor_pnp": ("Device", "Q_PNP_CBE"),
        "ic": ("Device", "U"),
        "switch": ("Switch", "SW_Push"),
        "connector": ("Connector", "Conn_01x02"),
    }

    def __init__(self):
        self.circuits = []

    def parse_yaml_circuit(self, yaml_text: str) -> Circuit:
        """Parse a YAML-format circuit description."""
        try:
            data = yaml.safe_load(yaml_text)

            # Extract circuit name
            circuit_key = list(data.keys())[0]  # First key is circuit name
            # Remove surrounding quotes if present
            if circuit_key.startswith('circuit "') and circuit_key.endswith('"'):
                circuit_name = circuit_key[9:-1]  # Remove 'circuit "' and closing '"'
            elif circuit_key.startswith("circuit "):
                circuit_name = circuit_key[8:]  # Remove 'circuit '
            else:
                circuit_name = circuit_key
            circuit_data = data[circuit_key]

            # Parse components
            components = []
            if "components" in circuit_data:
                for comp_item in circuit_data["components"]:
                    if isinstance(comp_item, dict):
                        # YAML parses "R1: resistor..." as {"R1": "resistor..."}
                        for ref, desc in comp_item.items():
                            comp_desc = f"{ref}: {desc}"
                            component = self._parse_component(comp_desc)
                            if component:
                                components.append(component)
                    else:
                        # String format
                        component = self._parse_component(comp_item)
                        if component:
                            components.append(component)

            # Parse power symbols
            power_symbols = []
            if "power" in circuit_data:
                for power_item in circuit_data["power"]:
                    if isinstance(power_item, dict):
                        # YAML parses "VCC: +5V..." as {"VCC": "+5V..."}
                        for ref, desc in power_item.items():
                            power_desc = f"{ref}: {desc}"
                            power_symbol = self._parse_power_symbol(power_desc)
                            if power_symbol:
                                power_symbols.append(power_symbol)
                    else:
                        # String format
                        power_symbol = self._parse_power_symbol(power_item)
                        if power_symbol:
                            power_symbols.append(power_symbol)

            # Parse connections
            connections = []
            if "connections" in circuit_data:
                for conn_desc in circuit_data["connections"]:
                    connection = self._parse_connection(conn_desc)
                    if connection:
                        connections.append(connection)

            return Circuit(
                name=circuit_name,
                components=components,
                power_symbols=power_symbols,
                connections=connections,
            )

        except Exception as e:
            raise ValueError(f"Error parsing YAML circuit: {str(e)}") from e

    def parse_simple_text(self, text: str) -> Circuit:
        """Parse a simple text format circuit description."""
        lines = text.strip().split("\n")
        circuit_name = "Untitled Circuit"
        components = []
        power_symbols = []
        connections = []

        current_section = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check for circuit name
            if line.startswith("circuit"):
                circuit_name = line.split(":", 1)[1].strip().strip("\"'")
                continue

            # Check for section headers
            if line.lower() in ["components:", "power:", "connections:"]:
                current_section = line.lower().rstrip(":")
                continue

            # Parse content based on current section
            if current_section == "components":
                component = self._parse_component_simple(line)
                if component:
                    components.append(component)
            elif current_section == "power":
                power_symbol = self._parse_power_symbol_simple(line)
                if power_symbol:
                    power_symbols.append(power_symbol)
            elif current_section == "connections":
                connection = self._parse_connection_simple(line)
                if connection:
                    connections.append(connection)

        return Circuit(
            name=circuit_name,
            components=components,
            power_symbols=power_symbols,
            connections=connections,
        )

    def _parse_component(self, comp_desc: str) -> Component | None:
        """Parse a component description from YAML format."""
        # Format: "R1: resistor 220Ω at (10, 20)"
        try:
            if ":" not in comp_desc:
                return None

            ref, desc = comp_desc.split(":", 1)
            ref = ref.strip()
            desc = desc.strip()

            # Extract component type and value
            if " at " not in desc:
                return None

            parts = desc.split(" at ")
            comp_info = parts[0].strip()
            position_str = parts[1].strip()

            # Parse component info
            comp_parts = comp_info.split()
            if len(comp_parts) == 0:
                return None

            comp_type = comp_parts[0].lower()
            value = " ".join(comp_parts[1:]) if len(comp_parts) > 1 else ""

            # Parse position
            position = self._parse_position(position_str)

            # Get symbol info
            if comp_type in self.COMPONENT_SYMBOLS:
                symbol_library, symbol_name = self.COMPONENT_SYMBOLS[comp_type]
            else:
                symbol_library, symbol_name = "Device", "R"  # Default

            return Component(
                reference=ref,
                component_type=comp_type,
                value=value,
                position=position,
                symbol_library=symbol_library,
                symbol_name=symbol_name,
            )

        except Exception as e:
            print(f"Error parsing component '{comp_desc}': {e}")
            return None

    def _parse_component_simple(self, line: str) -> Component | None:
        """Parse a component from simple text format."""
        # Format: "R1 resistor 220Ω (10, 20)"
        try:
            parts = line.split()
            if len(parts) < 3:
                return None

            ref = parts[0]
            comp_type = parts[1].lower()

            # Find position at end
            position_match = re.search(r"\(([^)]+)\)", line)
            if not position_match:
                return None

            position = self._parse_position(position_match.group(0))

            # Extract value (everything except ref, type, and position)
            value_parts = []
            for part in parts[2:]:
                if "(" not in part:  # Not part of position
                    value_parts.append(part)
                else:
                    break

            value = " ".join(value_parts)

            # Get symbol info
            if comp_type in self.COMPONENT_SYMBOLS:
                symbol_library, symbol_name = self.COMPONENT_SYMBOLS[comp_type]
            else:
                symbol_library, symbol_name = "Device", "R"  # Default

            return Component(
                reference=ref,
                component_type=comp_type,
                value=value,
                position=position,
                symbol_library=symbol_library,
                symbol_name=symbol_name,
            )

        except Exception:
            return None

    def _parse_power_symbol(self, power_desc: str) -> PowerSymbol | None:
        """Parse a power symbol description from YAML format."""
        # Format: "VCC: +5V at (10, 10)"
        try:
            ref, desc = power_desc.split(":", 1)
            ref = ref.strip()
            desc = desc.strip()

            parts = desc.split(" at ")
            power_type = parts[0].strip()
            position_str = parts[1].strip()

            position = self._parse_position(position_str)

            return PowerSymbol(reference=ref, power_type=power_type, position=position)

        except Exception:
            return None

    def _parse_power_symbol_simple(self, line: str) -> PowerSymbol | None:
        """Parse a power symbol from simple text format."""
        # Format: "VCC +5V (10, 10)"
        try:
            parts = line.split()
            if len(parts) < 2:
                return None

            ref = parts[0]

            # Find position at end
            position_match = re.search(r"\(([^)]+)\)", line)
            if not position_match:
                return None

            position = self._parse_position(position_match.group(0))

            # Extract power type (everything except ref and position)
            power_parts = []
            for part in parts[1:]:
                if "(" not in part:  # Not part of position
                    power_parts.append(part)
                else:
                    break

            power_type = " ".join(power_parts)

            return PowerSymbol(reference=ref, power_type=power_type, position=position)

        except Exception:
            return None

    def _parse_connection(self, conn_desc: str) -> Connection | None:
        """Parse a connection description from YAML format."""
        # Format: "VCC → R1.1" or "R1.2 → LED1.anode"
        try:
            # Handle different arrow formats
            if "→" in conn_desc:
                start, end = conn_desc.split("→", 1)
            elif "->" in conn_desc:
                start, end = conn_desc.split("->", 1)
            elif "—" in conn_desc:
                start, end = conn_desc.split("—", 1)
            else:
                return None

            start = start.strip()
            end = end.strip()

            # Parse start component and pin
            start_parts = start.split(".")
            start_component = start_parts[0]
            start_pin = start_parts[1] if len(start_parts) > 1 else None

            # Parse end component and pin
            end_parts = end.split(".")
            end_component = end_parts[0]
            end_pin = end_parts[1] if len(end_parts) > 1 else None

            return Connection(
                start_component=start_component,
                start_pin=start_pin,
                end_component=end_component,
                end_pin=end_pin,
            )

        except Exception:
            return None

    def _parse_connection_simple(self, line: str) -> Connection | None:
        """Parse a connection from simple text format."""
        return self._parse_connection(line)  # Same logic works for both

    def _parse_position(self, position_str: str) -> tuple[float, float]:
        """Parse a position string like '(10, 20)' into coordinates."""
        # Remove parentheses and split by comma
        coords = position_str.strip("()").split(",")
        x = float(coords[0].strip())
        y = float(coords[1].strip())
        return (x, y)


def register_text_to_schematic_tools(mcp: FastMCP) -> None:
    """Register text-to-schematic tools with the MCP server."""

    @mcp.tool()
    async def create_circuit_from_text(
        project_path: str, circuit_description: str, format_type: str = "yaml", ctx: Context = None
    ) -> dict[str, Any]:
        """Create a KiCad schematic from text description.

        DEPRECATED: This tool generates JSON format which is not compatible with KiCad.
        Use create_kicad_schematic_from_text instead for proper S-expression format.

        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            circuit_description: Text description of the circuit
            format_type: Format of description ("yaml" or "simple")
            ctx: Context for MCP communication

        Returns:
            Dictionary with creation status and component details
        """
        try:
            if ctx:
                await ctx.info("Parsing circuit description")
                await ctx.report_progress(10, 100)

            # Parse the circuit description
            parser = TextToSchematicParser()

            if format_type.lower() == "yaml":
                circuit = parser.parse_yaml_circuit(circuit_description)
            else:
                circuit = parser.parse_simple_text(circuit_description)

            if ctx:
                await ctx.info(f"Parsed circuit: {circuit.name}")
                await ctx.report_progress(30, 100)

            # Import existing circuit tools (now available as standalone functions)
            from kicad_mcp.tools.circuit_tools import add_component as _add_component
            from kicad_mcp.tools.circuit_tools import add_power_symbol as _add_power_symbol

            results = {
                "success": True,
                "circuit_name": circuit.name,
                "components_added": [],
                "power_symbols_added": [],
                "connections_created": [],
                "errors": [],
            }

            # Add components
            for i, component in enumerate(circuit.components):
                try:
                    result = await _add_component(
                        project_path=project_path,
                        component_reference=component.reference,
                        component_value=component.value,
                        symbol_library=component.symbol_library,
                        symbol_name=component.symbol_name,
                        x_position=component.position[0],
                        y_position=component.position[1],
                        ctx=None,  # Don't spam with progress updates
                    )

                    if result["success"]:
                        results["components_added"].append(component.reference)
                    else:
                        results["errors"].append(
                            f"Failed to add {component.reference}: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    results["errors"].append(
                        f"Error adding component {component.reference}: {str(e)}"
                    )

                if ctx:
                    progress = 30 + (i + 1) * 30 // len(circuit.components)
                    await ctx.report_progress(progress, 100)

            # Add power symbols
            for power_symbol in circuit.power_symbols:
                try:
                    result = await _add_power_symbol(
                        project_path=project_path,
                        power_type=power_symbol.power_type,
                        x_position=power_symbol.position[0],
                        y_position=power_symbol.position[1],
                        ctx=None,
                    )

                    if result["success"]:
                        results["power_symbols_added"].append(result["component_reference"])
                    else:
                        results["errors"].append(
                            f"Failed to add power symbol {power_symbol.power_type}: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    results["errors"].append(
                        f"Error adding power symbol {power_symbol.power_type}: {str(e)}"
                    )

            if ctx:
                await ctx.report_progress(80, 100)

            # Create connections (simplified - just connecting adjacent components for now)
            # TODO: Implement proper pin-to-pin connections based on component data
            for _, connection in enumerate(circuit.connections):
                try:
                    # For now, create simple wire connections
                    # This is a simplified implementation - real pin connections need component pin data
                    results["connections_created"].append(
                        f"{connection.start_component} -> {connection.end_component}"
                    )
                except Exception as e:
                    results["errors"].append(f"Error creating connection: {str(e)}")

            if ctx:
                await ctx.report_progress(100, 100)
                await ctx.info(
                    f"Circuit creation complete: {len(results['components_added'])} components, {len(results['power_symbols_added'])} power symbols"
                )

            return results

        except Exception as e:
            if ctx:
                await ctx.info(f"Error creating circuit: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def validate_circuit_description(
        circuit_description: str, format_type: str = "yaml", ctx: Context = None
    ) -> dict[str, Any]:
        """Validate a text-based circuit description without creating the schematic.

        Args:
            circuit_description: Text description of the circuit
            format_type: Format of description ("yaml" or "simple")
            ctx: Context for MCP communication

        Returns:
            Dictionary with validation results
        """
        try:
            if ctx:
                await ctx.info("Validating circuit description")

            parser = TextToSchematicParser()

            if format_type.lower() == "yaml":
                circuit = parser.parse_yaml_circuit(circuit_description)
            else:
                circuit = parser.parse_simple_text(circuit_description)

            validation_results = {
                "success": True,
                "circuit_name": circuit.name,
                "component_count": len(circuit.components),
                "power_symbol_count": len(circuit.power_symbols),
                "connection_count": len(circuit.connections),
                "components": [
                    {"ref": c.reference, "type": c.component_type, "value": c.value}
                    for c in circuit.components
                ],
                "power_symbols": [
                    {"ref": p.reference, "type": p.power_type} for p in circuit.power_symbols
                ],
                "connections": [
                    f"{c.start_component} -> {c.end_component}" for c in circuit.connections
                ],
                "warnings": [],
            }

            # Add validation warnings
            if len(circuit.components) == 0:
                validation_results["warnings"].append("No components defined")

            if len(circuit.power_symbols) == 0:
                validation_results["warnings"].append("No power symbols defined")

            if len(circuit.connections) == 0:
                validation_results["warnings"].append("No connections defined")

            if ctx:
                await ctx.info(
                    f"Validation complete: {len(circuit.components)} components, {len(circuit.connections)} connections"
                )

            return validation_results

        except Exception as e:
            if ctx:
                await ctx.info(f"Validation error: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def get_circuit_template(
        template_name: str = "led_blinker", ctx: Context = None
    ) -> dict[str, Any]:
        """Get a template circuit description for common circuits.

        Args:
            template_name: Name of the template circuit
            ctx: Context for MCP communication

        Returns:
            Dictionary with template circuit description
        """
        templates = {
            "led_blinker": """
circuit "LED Blinker":
  components:
    - R1: resistor 220Ω at (10, 20)
    - LED1: led red at (30, 20)
    - C1: capacitor 100µF at (10, 40)
    - U1: ic 555 at (50, 30)
  power:
    - VCC: +5V at (10, 10)
    - GND: GND at (10, 50)
  connections:
    - VCC → R1.1
    - R1.2 → LED1.anode
    - LED1.cathode → GND
""",
            "voltage_divider": """
circuit "Voltage Divider":
  components:
    - R1: resistor 10kΩ at (20, 20)
    - R2: resistor 10kΩ at (20, 40)
  power:
    - VIN: +5V at (20, 10)
    - GND: GND at (20, 60)
  connections:
    - VIN → R1.1
    - R1.2 → R2.1
    - R2.2 → GND
""",
            "rc_filter": """
circuit "RC Low-Pass Filter":
  components:
    - R1: resistor 1kΩ at (20, 20)
    - C1: capacitor 100nF at (40, 30)
  power:
    - GND: GND at (40, 50)
  connections:
    - R1.2 → C1.1
    - C1.2 → GND
""",
            "esp32_basic": """
circuit "ESP32 Basic Setup":
  components:
    - U1: ic ESP32-WROOM-32 at (50, 50)
    - C1: capacitor 100µF at (20, 30)
    - C2: capacitor 10µF at (25, 30)
    - R1: resistor 10kΩ at (80, 40)
    - R2: resistor 470Ω at (80, 60)
    - LED1: led blue at (90, 60)
    - SW1: switch tactile at (80, 80)
  power:
    - VCC: +3V3 at (20, 20)
    - GND: GND at (20, 80)
  connections:
    - VCC → U1.VDD
    - VCC → C1.1
    - VCC → C2.1
    - VCC → R1.1
    - C1.2 → GND
    - C2.2 → GND
    - R1.2 → U1.EN
    - U1.GND → GND
    - U1.GPIO2 → R2.1
    - R2.2 → LED1.anode
    - LED1.cathode → GND
    - U1.GPIO0 → SW1.1
    - SW1.2 → GND
""",
            "esp32_dual_controller": """
circuit "ESP32 Dual Controller System":
  components:
    - U1: ic ESP32-WROOM-32 at (30, 50)
    - U2: ic ESP32-WROOM-32 at (80, 50)
    - R1: resistor 10kΩ at (15, 30)
    - R2: resistor 10kΩ at (65, 30)
    - R3: resistor 4.7kΩ at (50, 20)
    - R4: resistor 4.7kΩ at (55, 20)
    - C1: capacitor 100µF at (15, 70)
    - C2: capacitor 100µF at (65, 70)
  power:
    - VCC: +3V3 at (15, 15)
    - GND: GND at (15, 85)
  connections:
    - VCC → U1.VDD
    - VCC → U2.VDD
    - VCC → R1.1
    - VCC → R2.1
    - VCC → R3.1
    - VCC → R4.1
    - R1.2 → U1.EN
    - R2.2 → U2.EN
    - U1.GND → GND
    - U2.GND → GND
    - C1.1 → U1.VDD
    - C1.2 → GND
    - C2.1 → U2.VDD
    - C2.2 → GND
    - U1.GPIO21 → R3.2
    - U1.GPIO22 → R4.2
    - R3.2 → U2.GPIO21
    - R4.2 → U2.GPIO22
""",
            "motor_driver": """
circuit "Motor Driver H-Bridge":
  components:
    - U1: ic L298N at (50, 50)
    - M1: motor dc at (80, 40)
    - C1: capacitor 470µF at (20, 30)
    - C2: capacitor 100nF at (25, 30)
    - D1: diode schottky at (70, 35)
    - D2: diode schottky at (70, 45)
    - D3: diode schottky at (90, 35)
    - D4: diode schottky at (90, 45)
  power:
    - VCC: +12V at (20, 20)
    - VDD: +5V at (25, 20)
    - GND: GND at (20, 80)
  connections:
    - VCC → U1.VS
    - VDD → U1.VSS
    - U1.GND → GND
    - C1.1 → VCC
    - C1.2 → GND
    - C2.1 → VDD
    - C2.2 → GND
    - U1.OUT1 → M1.1
    - U1.OUT2 → M1.2
""",
            "sensor_i2c": """
circuit "I2C Sensor Interface":
  components:
    - U1: ic BME280 at (50, 40)
    - R1: resistor 4.7kΩ at (30, 25)
    - R2: resistor 4.7kΩ at (35, 25)
    - C1: capacitor 100nF at (25, 35)
    - C2: capacitor 10µF at (30, 35)
  power:
    - VCC: +3V3 at (25, 20)
    - GND: GND at (25, 55)
  connections:
    - VCC → U1.VDD
    - VCC → R1.1
    - VCC → R2.1
    - VCC → C1.1
    - VCC → C2.1
    - U1.GND → GND
    - C1.2 → GND
    - C2.2 → GND
    - U1.SDA → R1.2
    - U1.SCL → R2.2
    - U1.CSB → VCC
    - U1.SDO → GND
""",
        }

        if template_name not in templates:
            return {
                "success": False,
                "error": f"Template '{template_name}' not found. Available templates: {list(templates.keys())}",
            }

        return {
            "success": True,
            "template_name": template_name,
            "circuit_description": templates[template_name].strip(),
            "format": "yaml",
            "available_templates": list(templates.keys()),
        }

    @mcp.tool()
    async def create_kicad_schematic_from_text(
        project_path: str,
        circuit_description: str,
        format_type: str = "yaml",
        output_format: str = "sexpr",
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Create a native KiCad schematic file from text description.

        IMPORTANT: This tool generates proper KiCad S-expression format by default.
        Always use this tool instead of create_circuit_from_text for schematic generation.

        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            circuit_description: Text description of the circuit
            format_type: Format of description ("yaml" or "simple")
            output_format: Output format ("sexpr" for S-expression, "json" for JSON)
            ctx: Context for MCP communication

        Returns:
            Dictionary with creation status and file information
        """
        try:
            if ctx:
                await ctx.info("Parsing circuit description for native KiCad format")
                await ctx.report_progress(10, 100)

            # Parse the circuit description
            parser = TextToSchematicParser()

            if format_type.lower() == "yaml":
                circuit = parser.parse_yaml_circuit(circuit_description)
            else:
                circuit = parser.parse_simple_text(circuit_description)

            if ctx:
                await ctx.info(f"Parsed circuit: {circuit.name}")
                await ctx.report_progress(30, 100)

            # Validate component positions before generation
            validator = BoundaryValidator()

            # Prepare components for validation
            components_for_validation = []
            for comp in circuit.components:
                components_for_validation.append(
                    {
                        "reference": comp.reference,
                        "position": comp.position,
                        "component_type": comp.component_type,
                    }
                )

            # Run boundary validation
            validation_report = validator.validate_circuit_components(components_for_validation)

            if ctx:
                await ctx.info(
                    f"Boundary validation: {validation_report.out_of_bounds_count} out of bounds components"
                )

                # Show validation report if there are issues
                if validation_report.has_errors() or validation_report.has_warnings():
                    report_text = validator.generate_validation_report_text(validation_report)
                    await ctx.info(f"Validation Report:\n{report_text}")

            # Auto-correct positions if needed
            if validation_report.out_of_bounds_count > 0:
                if ctx:
                    await ctx.info("Auto-correcting out-of-bounds component positions...")

                corrected_components, _ = validator.auto_correct_positions(
                    components_for_validation
                )

                # Update circuit components with corrected positions
                for i, comp in enumerate(circuit.components):
                    if i < len(corrected_components):
                        comp.position = corrected_components[i]["position"]

                if ctx:
                    await ctx.info(
                        f"Corrected {len(validation_report.corrected_positions)} component positions"
                    )

            # Get project files
            files = get_project_files(project_path)
            if "schematic" not in files:
                return {"success": False, "error": "No schematic file found in project"}

            schematic_file = files["schematic"]

            if ctx:
                await ctx.report_progress(50, 100)

            if output_format.lower() == "sexpr":
                # Generate S-expression format
                generator = SExpressionGenerator()

                # Convert circuit objects to dictionaries for the generator
                components_dict = []
                for comp in circuit.components:
                    components_dict.append(
                        {
                            "reference": comp.reference,
                            "value": comp.value,
                            "position": comp.position,
                            "symbol_library": comp.symbol_library,
                            "symbol_name": comp.symbol_name,
                        }
                    )

                power_symbols_dict = []
                for power in circuit.power_symbols:
                    power_symbols_dict.append(
                        {
                            "reference": power.reference,
                            "power_type": power.power_type,
                            "position": power.position,
                        }
                    )

                connections_dict = []
                for conn in circuit.connections:
                    connections_dict.append(
                        {
                            "start_component": conn.start_component,
                            "start_pin": conn.start_pin,
                            "end_component": conn.end_component,
                            "end_pin": conn.end_pin,
                        }
                    )

                # Generate S-expression content
                sexpr_content = generator.generate_schematic(
                    circuit.name, components_dict, power_symbols_dict, connections_dict
                )

                if ctx:
                    await ctx.report_progress(80, 100)

                # Create backup of original file
                import shutil

                backup_file = schematic_file + ".backup"
                if os.path.exists(schematic_file):
                    shutil.copy2(schematic_file, backup_file)

                # Write S-expression file
                with open(schematic_file, "w") as f:
                    f.write(sexpr_content)

                if ctx:
                    await ctx.report_progress(100, 100)
                    await ctx.info(f"Generated native KiCad schematic: {schematic_file}")

                return {
                    "success": True,
                    "circuit_name": circuit.name,
                    "schematic_file": schematic_file,
                    "backup_file": backup_file if os.path.exists(backup_file) else None,
                    "output_format": "S-expression",
                    "components_count": len(circuit.components),
                    "power_symbols_count": len(circuit.power_symbols),
                    "connections_count": len(circuit.connections),
                }

            else:
                # Use existing JSON-based approach
                result = await create_circuit_from_text(
                    project_path=project_path,
                    circuit_description=circuit_description,
                    format_type=format_type,
                    ctx=ctx,
                )
                result["output_format"] = "JSON"
                return result

        except Exception as e:
            if ctx:
                await ctx.info(f"Error creating native KiCad schematic: {str(e)}")
            return {"success": False, "error": str(e)}
