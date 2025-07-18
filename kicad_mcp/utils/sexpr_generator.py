"""
S-expression generator for KiCad schematic files.

Converts circuit descriptions to proper KiCad S-expression format.
"""

import uuid

from kicad_mcp.utils.component_layout import ComponentLayoutManager
from kicad_mcp.utils.coordinate_converter import layout_to_kicad
from kicad_mcp.utils.pin_mapper import ComponentPinMapper


class SExpressionGenerator:
    """Generator for KiCad S-expression format schematics."""

    def __init__(self):
        self.symbol_libraries = {}
        self.component_uuid_map = {}
        self.layout_manager = ComponentLayoutManager()
        self.pin_mapper = ComponentPinMapper()

    def generate_schematic(
        self,
        circuit_name: str,
        components: list[dict],
        power_symbols: list[dict],
        connections: list[dict],
    ) -> str:
        """Generate a complete KiCad schematic in S-expression format.

        Args:
            circuit_name: Name of the circuit
            components: List of component dictionaries
            power_symbols: List of power symbol dictionaries
            connections: List of connection dictionaries

        Returns:
            S-expression formatted schematic as string
        """
        # Clear previous layout and pin mappings
        self.layout_manager.clear_layout()
        self.pin_mapper.clear_mappings()

        # Validate and fix component positions using layout manager
        validated_components = self._validate_component_positions(components)
        validated_power_symbols = self._validate_power_positions(power_symbols)

        # Add components to pin mapper for accurate pin tracking
        self._map_component_pins(validated_components, validated_power_symbols)

        # Generate main schematic UUID
        main_uuid = str(uuid.uuid4())

        # Start building the S-expression
        sexpr_lines = [
            "(kicad_sch",
            "  (version 20240618)",
            "  (generator kicad-mcp)",
            f'  (uuid "{main_uuid}")',
            '  (paper "A4")',
            "",
            "  (title_block",
            f'    (title "{circuit_name}")',
            '    (date "")',
            '    (rev "")',
            '    (company "")',
            "  )",
            "",
        ]

        # Add symbol libraries
        lib_symbols = self._generate_lib_symbols(validated_components, validated_power_symbols)
        if lib_symbols:
            sexpr_lines.extend(lib_symbols)
            sexpr_lines.append("")

        # Add components (symbols)
        for component in validated_components:
            symbol_lines = self._generate_component_symbol(component)
            sexpr_lines.extend(symbol_lines)
            sexpr_lines.append("")

        # Add power symbols
        for power_symbol in validated_power_symbols:
            symbol_lines = self._generate_power_symbol(power_symbol)
            sexpr_lines.extend(symbol_lines)
            sexpr_lines.append("")

        # Add wires for connections
        for connection in connections:
            wire_lines = self._generate_wire(connection)
            sexpr_lines.extend(wire_lines)

        if connections:
            sexpr_lines.append("")

        # Add sheet instances (required)
        sexpr_lines.extend(["  (sheet_instances", '    (path "/" (page "1"))', "  )", ")"])

        return "\n".join(sexpr_lines)

    def _generate_lib_symbols(self, components: list[dict], power_symbols: list[dict]) -> list[str]:
        """Generate lib_symbols section."""
        lines = ["  (lib_symbols"]

        # Collect unique symbol libraries
        symbols_needed = set()

        for component in components:
            lib_id = (
                f"{component.get('symbol_library', 'Device')}:{component.get('symbol_name', 'R')}"
            )
            symbols_needed.add(lib_id)

        for power_symbol in power_symbols:
            power_type = power_symbol.get("power_type", "VCC")
            lib_id = f"power:{power_type}"
            symbols_needed.add(lib_id)

        # Generate basic symbol definitions
        for lib_id in sorted(symbols_needed):
            library, symbol = lib_id.split(":")
            symbol_def = self._generate_symbol_definition(library, symbol)
            lines.extend([f"    {line}" for line in symbol_def])

        lines.append("  )")
        return lines

    def _generate_symbol_definition(self, library: str, symbol: str) -> list[str]:
        """Generate a basic symbol definition."""
        if library == "Device":
            if symbol == "R":
                return self._generate_resistor_symbol()
            elif symbol == "C":
                return self._generate_capacitor_symbol()
            elif symbol == "L":
                return self._generate_inductor_symbol()
            elif symbol == "LED":
                return self._generate_led_symbol()
            elif symbol == "D":
                return self._generate_diode_symbol()
        elif library == "power":
            return self._generate_power_symbol_definition(symbol)

        # Default symbol (resistor-like)
        return self._generate_resistor_symbol()

    def _generate_resistor_symbol(self) -> list[str]:
        """Generate resistor symbol definition."""
        return [
            '(symbol "Device:R"',
            "  (pin_numbers hide)",
            "  (pin_names (offset 0))",
            "  (exclude_from_sim no)",
            "  (in_bom yes)",
            "  (on_board yes)",
            '  (property "Reference" "R" (at 2.032 0 90))',
            '  (property "Value" "R" (at 0 0 90))',
            '  (property "Footprint" "" (at -1.778 0 90))',
            '  (property "Datasheet" "~" (at 0 0 0))',
            '  (symbol "R_0_1"',
            "    (rectangle (start -1.016 -2.54) (end 1.016 2.54))",
            "  )",
            '  (symbol "R_1_1"',
            "    (pin passive line (at 0 3.81 270) (length 1.27)",
            '      (name "~" (effects (font (size 1.27 1.27))))',
            '      (number "1" (effects (font (size 1.27 1.27))))',
            "    )",
            "    (pin passive line (at 0 -3.81 90) (length 1.27)",
            '      (name "~" (effects (font (size 1.27 1.27))))',
            '      (number "2" (effects (font (size 1.27 1.27))))',
            "    )",
            "  )",
            ")",
        ]

    def _generate_capacitor_symbol(self) -> list[str]:
        """Generate capacitor symbol definition."""
        return [
            '(symbol "Device:C"',
            "  (pin_numbers hide)",
            "  (pin_names (offset 0.254))",
            "  (exclude_from_sim no)",
            "  (in_bom yes)",
            "  (on_board yes)",
            '  (property "Reference" "C" (at 0.635 2.54 0))',
            '  (property "Value" "C" (at 0.635 -2.54 0))',
            '  (property "Footprint" "" (at 0.9652 -3.81 0))',
            '  (property "Datasheet" "~" (at 0 0 0))',
            '  (symbol "C_0_1"',
            "    (polyline",
            "      (pts (xy -2.032 -0.762) (xy 2.032 -0.762))",
            "    )",
            "    (polyline",
            "      (pts (xy -2.032 0.762) (xy 2.032 0.762))",
            "    )",
            "  )",
            '  (symbol "C_1_1"',
            "    (pin passive line (at 0 3.81 270) (length 2.794)",
            '      (name "~" (effects (font (size 1.27 1.27))))',
            '      (number "1" (effects (font (size 1.27 1.27))))',
            "    )",
            "    (pin passive line (at 0 -3.81 90) (length 2.794)",
            '      (name "~" (effects (font (size 1.27 1.27))))',
            '      (number "2" (effects (font (size 1.27 1.27))))',
            "    )",
            "  )",
            ")",
        ]

    def _generate_inductor_symbol(self) -> list[str]:
        """Generate inductor symbol definition."""
        return [
            '(symbol "Device:L"',
            "  (pin_numbers hide)",
            "  (pin_names (offset 1.016) hide)",
            "  (exclude_from_sim no)",
            "  (in_bom yes)",
            "  (on_board yes)",
            '  (property "Reference" "L" (at -1.27 0 90))',
            '  (property "Value" "L" (at 1.905 0 90))',
            '  (property "Footprint" "" (at 0 0 0))',
            '  (property "Datasheet" "~" (at 0 0 0))',
            '  (symbol "L_0_1"',
            "    (arc (start 0 -2.54) (mid 0.6323 -1.905) (end 0 -1.27))",
            "    (arc (start 0 -1.27) (mid 0.6323 -0.635) (end 0 0))",
            "    (arc (start 0 0) (mid 0.6323 0.635) (end 0 1.27))",
            "    (arc (start 0 1.27) (mid 0.6323 1.905) (end 0 2.54))",
            "  )",
            '  (symbol "L_1_1"',
            "    (pin passive line (at 0 3.81 270) (length 1.27)",
            '      (name "1" (effects (font (size 1.27 1.27))))',
            '      (number "1" (effects (font (size 1.27 1.27))))',
            "    )",
            "    (pin passive line (at 0 -3.81 90) (length 1.27)",
            '      (name "2" (effects (font (size 1.27 1.27))))',
            '      (number "2" (effects (font (size 1.27 1.27))))',
            "    )",
            "  )",
            ")",
        ]

    def _generate_led_symbol(self) -> list[str]:
        """Generate LED symbol definition."""
        return [
            '(symbol "Device:LED"',
            "  (pin_numbers hide)",
            "  (pin_names (offset 1.016) hide)",
            "  (exclude_from_sim no)",
            "  (in_bom yes)",
            "  (on_board yes)",
            '  (property "Reference" "D" (at 0 2.54 0))',
            '  (property "Value" "LED" (at 0 -2.54 0))',
            '  (property "Footprint" "" (at 0 0 0))',
            '  (property "Datasheet" "~" (at 0 0 0))',
            '  (symbol "LED_0_1"',
            "    (polyline",
            "      (pts (xy -1.27 -1.27) (xy -1.27 1.27))",
            "    )",
            "    (polyline",
            "      (pts (xy -1.27 0) (xy 1.27 0))",
            "    )",
            "    (polyline",
            "      (pts (xy 1.27 -1.27) (xy 1.27 1.27) (xy -1.27 0) (xy 1.27 -1.27))",
            "    )",
            "  )",
            '  (symbol "LED_1_1"',
            "    (pin passive line (at -3.81 0 0) (length 2.54)",
            '      (name "K" (effects (font (size 1.27 1.27))))',
            '      (number "1" (effects (font (size 1.27 1.27))))',
            "    )",
            "    (pin passive line (at 3.81 0 180) (length 2.54)",
            '      (name "A" (effects (font (size 1.27 1.27))))',
            '      (number "2" (effects (font (size 1.27 1.27))))',
            "    )",
            "  )",
            ")",
        ]

    def _generate_diode_symbol(self) -> list[str]:
        """Generate diode symbol definition."""
        return [
            '(symbol "Device:D"',
            "  (pin_numbers hide)",
            "  (pin_names (offset 1.016) hide)",
            "  (exclude_from_sim no)",
            "  (in_bom yes)",
            "  (on_board yes)",
            '  (property "Reference" "D" (at 0 2.54 0))',
            '  (property "Value" "D" (at 0 -2.54 0))',
            '  (property "Footprint" "" (at 0 0 0))',
            '  (property "Datasheet" "~" (at 0 0 0))',
            '  (symbol "D_0_1"',
            "    (polyline",
            "      (pts (xy -1.27 -1.27) (xy -1.27 1.27))",
            "    )",
            "    (polyline",
            "      (pts (xy -1.27 0) (xy 1.27 0))",
            "    )",
            "    (polyline",
            "      (pts (xy 1.27 -1.27) (xy 1.27 1.27) (xy -1.27 0) (xy 1.27 -1.27))",
            "    )",
            "  )",
            '  (symbol "D_1_1"',
            "    (pin passive line (at -3.81 0 0) (length 2.54)",
            '      (name "K" (effects (font (size 1.27 1.27))))',
            '      (number "1" (effects (font (size 1.27 1.27))))',
            "    )",
            "    (pin passive line (at 3.81 0 180) (length 2.54)",
            '      (name "A" (effects (font (size 1.27 1.27))))',
            '      (number "2" (effects (font (size 1.27 1.27))))',
            "    )",
            "  )",
            ")",
        ]

    def _generate_power_symbol_definition(self, power_type: str) -> list[str]:
        """Generate power symbol definition."""
        return [
            f'(symbol "power:{power_type}"',
            "  (power)",
            "  (pin_names (offset 0) hide)",
            "  (exclude_from_sim no)",
            "  (in_bom yes)",
            "  (on_board yes)",
            '  (property "Reference" "#PWR" (at 0 -3.81 0))',
            f'  (property "Value" "{power_type}" (at 0 3.556 0))',
            '  (property "Footprint" "" (at 0 0 0))',
            '  (property "Datasheet" "" (at 0 0 0))',
            f'  (symbol "{power_type}_0_1"',
            "    (polyline",
            "      (pts (xy -0.762 1.27) (xy 0 2.54))",
            "    )",
            "    (polyline",
            "      (pts (xy 0 0) (xy 0 2.54))",
            "    )",
            "    (polyline",
            "      (pts (xy 0 2.54) (xy 0.762 1.27))",
            "    )",
            "  )",
            f'  (symbol "{power_type}_1_1"',
            "    (pin power_in line (at 0 0 90) (length 0) hide",
            '      (name "1" (effects (font (size 1.27 1.27))))',
            '      (number "1" (effects (font (size 1.27 1.27))))',
            "    )",
            "  )",
            ")",
        ]

    def _validate_component_positions(self, components: list[dict]) -> list[dict]:
        """Validate and fix component positions using the layout manager."""
        validated_components = []

        for component in components:
            # Get component type for sizing
            component_type = self._get_component_type(component)

            # Check if position is provided
            if "position" in component and component["position"]:
                x, y = component["position"]
                # Validate position is within bounds
                if self.layout_manager.validate_position(x, y, component_type):
                    # Position is valid, place component at exact location
                    final_x, final_y = self.layout_manager.place_component(
                        component["reference"], component_type, x, y
                    )
                else:
                    # Position is invalid, find a valid one
                    final_x, final_y = self.layout_manager.place_component(
                        component["reference"], component_type
                    )
            else:
                # No position provided, auto-place
                final_x, final_y = self.layout_manager.place_component(
                    component["reference"], component_type
                )

            # Update component with validated position
            validated_component = component.copy()
            validated_component["position"] = (final_x, final_y)
            validated_components.append(validated_component)

        return validated_components

    def _validate_power_positions(self, power_symbols: list[dict]) -> list[dict]:
        """Validate and fix power symbol positions using the layout manager."""
        validated_power_symbols = []

        for power_symbol in power_symbols:
            # Power symbols use 'power' component type
            component_type = "power"

            # Check if position is provided
            if "position" in power_symbol and power_symbol["position"]:
                x, y = power_symbol["position"]
                # Validate position is within bounds
                if self.layout_manager.validate_position(x, y, component_type):
                    # Position is valid, place power symbol at exact location
                    final_x, final_y = self.layout_manager.place_component(
                        power_symbol["reference"], component_type, x, y
                    )
                else:
                    # Position is invalid, find a valid one
                    final_x, final_y = self.layout_manager.place_component(
                        power_symbol["reference"], component_type
                    )
            else:
                # No position provided, auto-place
                final_x, final_y = self.layout_manager.place_component(
                    power_symbol["reference"], component_type
                )

            # Update power symbol with validated position
            validated_power_symbol = power_symbol.copy()
            validated_power_symbol["position"] = (final_x, final_y)
            validated_power_symbols.append(validated_power_symbol)

        return validated_power_symbols

    def _get_component_type(self, component: dict) -> str:
        """Determine component type from component dictionary."""
        # Check if component_type is explicitly provided
        if "component_type" in component:
            return component["component_type"]

        # Infer from symbol information
        symbol_name = component.get("symbol_name", "").lower()
        symbol_library = component.get("symbol_library", "").lower()

        # Map symbol names to component types
        if symbol_name in ["r", "resistor"]:
            return "resistor"
        elif symbol_name in ["c", "capacitor"]:
            return "capacitor"
        elif symbol_name in ["l", "inductor"]:
            return "inductor"
        elif symbol_name in ["led"]:
            return "led"
        elif symbol_name in ["d", "diode"]:
            return "diode"
        elif "transistor" in symbol_name:
            return "transistor"
        elif symbol_library == "switch":
            return "switch"
        elif symbol_library == "connector":
            return "connector"
        elif "ic" in symbol_name or "mcu" in symbol_name:
            return "ic"
        else:
            return "default"

    def _map_component_pins(self, components: list[dict], power_symbols: list[dict]):
        """Map all components and power symbols to the pin mapper."""
        # Map regular components
        for component in components:
            component_type = self._get_component_type(component)
            self.pin_mapper.add_component(
                component_ref=component["reference"],
                component_type=component_type,
                position=component["position"],
                angle=0.0,  # Default angle, could be extended later
            )

        # Map power symbols
        for power_symbol in power_symbols:
            self.pin_mapper.add_component(
                component_ref=power_symbol["reference"],
                component_type="power",
                position=power_symbol["position"],
                angle=0.0,
            )

    def _generate_component_symbol(self, component: dict) -> list[str]:
        """Generate component symbol instance."""
        comp_uuid = str(uuid.uuid4())
        self.component_uuid_map[component["reference"]] = comp_uuid

        # Convert position from ComponentLayoutManager coordinates to KiCad coordinates
        x_pos, y_pos = layout_to_kicad(component["position"][0], component["position"][1])

        lib_id = f"{component.get('symbol_library', 'Device')}:{component.get('symbol_name', 'R')}"

        lines = [
            f'  (symbol (lib_id "{lib_id}") (at {x_pos} {y_pos} 0) (unit 1)',
            "    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)",
            f'    (uuid "{comp_uuid}")',
            f'    (property "Reference" "{component["reference"]}" (at {x_pos + 25.4} {y_pos - 12.7} 0))',
            f'    (property "Value" "{component["value"]}" (at {x_pos + 25.4} {y_pos + 12.7} 0))',
            f'    (property "Footprint" "" (at {x_pos} {y_pos} 0))',
            f'    (property "Datasheet" "~" (at {x_pos} {y_pos} 0))',
        ]

        # Add pin UUIDs (basic 2-pin component)
        lines.extend(
            [
                f'    (pin "1" (uuid "{str(uuid.uuid4())}"))',
                f'    (pin "2" (uuid "{str(uuid.uuid4())}"))',
                "  )",
            ]
        )

        return lines

    def _generate_power_symbol(self, power_symbol: dict) -> list[str]:
        """Generate power symbol instance."""
        power_uuid = str(uuid.uuid4())
        ref = power_symbol.get("reference", f"#PWR0{len(self.component_uuid_map) + 1:03d}")
        self.component_uuid_map[ref] = power_uuid

        # Convert position from ComponentLayoutManager coordinates to KiCad coordinates
        x_pos, y_pos = layout_to_kicad(power_symbol["position"][0], power_symbol["position"][1])

        power_type = power_symbol["power_type"]
        lib_id = f"power:{power_type}"

        lines = [
            f'  (symbol (lib_id "{lib_id}") (at {x_pos} {y_pos} 0) (unit 1)',
            "    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)",
            f'    (uuid "{power_uuid}")',
            f'    (property "Reference" "{ref}" (at {x_pos} {y_pos - 25.4} 0))',
            f'    (property "Value" "{power_type}" (at {x_pos} {y_pos + 35.56} 0))',
            f'    (property "Footprint" "" (at {x_pos} {y_pos} 0))',
            f'    (property "Datasheet" "" (at {x_pos} {y_pos} 0))',
            f'    (pin "1" (uuid "{str(uuid.uuid4())}"))',
            "  )",
        ]

        return lines

    def _generate_wire(self, connection: dict) -> list[str]:
        """Generate wire connection using pin-level routing."""
        lines = []

        # Check if connection specifies components and pins
        if "start_component" in connection and "end_component" in connection:
            # Pin-level connection
            start_component = connection["start_component"]
            start_pin = connection.get("start_pin", "1")
            end_component = connection["end_component"]
            end_pin = connection.get("end_pin", "1")

            # Get pin connection points
            start_point = self.pin_mapper.get_pin_connection_point(start_component, start_pin)
            end_point = self.pin_mapper.get_pin_connection_point(end_component, end_pin)

            if start_point and end_point:
                # Get the pins for routing calculation
                start_pin_obj = self.pin_mapper.get_pin(start_component, start_pin)
                end_pin_obj = self.pin_mapper.get_pin(end_component, end_pin)

                if start_pin_obj and end_pin_obj:
                    # Calculate wire route using pin mapper
                    route_points = self.pin_mapper.calculate_wire_route(start_pin_obj, end_pin_obj)

                    # Generate wire segments for the route
                    for i in range(len(route_points) - 1):
                        wire_uuid = str(uuid.uuid4())
                        start_x = int(route_points[i][0] * 10)
                        start_y = int(route_points[i][1] * 10)
                        end_x = int(route_points[i + 1][0] * 10)
                        end_y = int(route_points[i + 1][1] * 10)

                        lines.extend(
                            [
                                f"  (wire (pts (xy {start_x} {start_y}) (xy {end_x} {end_y})) (stroke (width 0) (type default))",
                                f'    (uuid "{wire_uuid}")',
                                "  )",
                            ]
                        )

                    # Add connection tracking
                    self.pin_mapper.add_connection(
                        start_component, start_pin, end_component, end_pin
                    )
        else:
            # Legacy coordinate-based connection
            wire_uuid = str(uuid.uuid4())
            start_x = connection.get("start_x", 100) * 10
            start_y = connection.get("start_y", 100) * 10
            end_x = connection.get("end_x", 200) * 10
            end_y = connection.get("end_y", 100) * 10

            lines = [
                f"  (wire (pts (xy {start_x} {start_y}) (xy {end_x} {end_y})) (stroke (width 0) (type default))",
                f'    (uuid "{wire_uuid}")',
                "  )",
            ]

        return lines

    def generate_advanced_wire_routing(self, net_connections: list[dict]) -> list[str]:
        """
        Generate advanced wire routing for complex nets.

        Args:
            net_connections: List of net connection dictionaries with multiple pins

        Returns:
            List of S-expression lines for all wire segments
        """
        lines = []

        for net in net_connections:
            net.get("name", "unnamed_net")
            net_pins = net.get("pins", [])

            if len(net_pins) < 2:
                continue

            # Get ComponentPin objects for all pins in the net
            component_pins = []
            for pin_ref in net_pins:
                if "." in pin_ref:
                    component_ref, pin_number = pin_ref.split(".", 1)
                    pin_obj = self.pin_mapper.get_pin(component_ref, pin_number)
                    if pin_obj:
                        component_pins.append(pin_obj)

            if len(component_pins) < 2:
                continue

            # Use bus routing for nets with multiple pins
            if len(component_pins) > 2:
                bus_routes = self.pin_mapper.calculate_bus_route(component_pins)

                for route in bus_routes:
                    for i in range(len(route) - 1):
                        wire_uuid = str(uuid.uuid4())
                        start_x = int(route[i][0] * 10)
                        start_y = int(route[i][1] * 10)
                        end_x = int(route[i + 1][0] * 10)
                        end_y = int(route[i + 1][1] * 10)

                        lines.extend(
                            [
                                f"  (wire (pts (xy {start_x} {start_y}) (xy {end_x} {end_y})) (stroke (width 0) (type default))",
                                f'    (uuid "{wire_uuid}")',
                                "  )",
                            ]
                        )
            else:
                # Point-to-point routing for two pins
                route_points = self.pin_mapper.calculate_wire_route(
                    component_pins[0], component_pins[1]
                )

                for i in range(len(route_points) - 1):
                    wire_uuid = str(uuid.uuid4())
                    start_x = int(route_points[i][0] * 10)
                    start_y = int(route_points[i][1] * 10)
                    end_x = int(route_points[i + 1][0] * 10)
                    end_y = int(route_points[i + 1][1] * 10)

                    lines.extend(
                        [
                            f"  (wire (pts (xy {start_x} {start_y}) (xy {end_x} {end_y})) (stroke (width 0) (type default))",
                            f'    (uuid "{wire_uuid}")',
                            "  )",
                        ]
                    )

        return lines
