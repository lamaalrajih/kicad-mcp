"""
Pin mapping and connectivity management for KiCad components.

Provides pin-level tracking, position calculation, and connection validation
for accurate wire routing in KiCad schematics.
"""

from dataclasses import dataclass
from enum import Enum
import math


class PinDirection(Enum):
    """Pin electrical directions."""

    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    PASSIVE = "passive"
    POWER_IN = "power_in"
    POWER_OUT = "power_out"
    OPEN_COLLECTOR = "open_collector"
    OPEN_EMITTER = "open_emitter"
    NO_CONNECT = "no_connect"


class PinType(Enum):
    """Pin electrical types."""

    ELECTRICAL = "electrical"
    POWER = "power"
    GROUND = "ground"
    SIGNAL = "signal"


@dataclass
class PinInfo:
    """Information about a component pin."""

    number: str
    name: str
    direction: PinDirection
    pin_type: PinType
    position: tuple[float, float]  # Relative to component center
    length: float = 2.54  # Default pin length in mm
    angle: float = 0.0  # Pin angle in degrees (0 = right, 90 = up, etc.)

    def get_connection_point(
        self, component_x: float, component_y: float, component_angle: float = 0.0
    ) -> tuple[float, float]:
        """Calculate the wire connection point for this pin."""
        # Apply component rotation to pin position
        rad = math.radians(component_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Rotate pin position relative to component
        rotated_x = self.position[0] * cos_a - self.position[1] * sin_a
        rotated_y = self.position[0] * sin_a + self.position[1] * cos_a

        # Add component position
        pin_x = component_x + rotated_x
        pin_y = component_y + rotated_y

        # Calculate connection point at pin tip
        pin_angle_rad = math.radians(self.angle + component_angle)
        connection_x = pin_x + self.length * math.cos(pin_angle_rad)
        connection_y = pin_y + self.length * math.sin(pin_angle_rad)

        return (connection_x, connection_y)


@dataclass
class ComponentPin:
    """Component pin with position information."""

    component_ref: str
    pin_info: PinInfo
    component_position: tuple[float, float]
    component_angle: float = 0.0

    @property
    def connection_point(self) -> tuple[float, float]:
        """Get the wire connection point for this pin."""
        return self.pin_info.get_connection_point(
            self.component_position[0], self.component_position[1], self.component_angle
        )


class ComponentPinMapper:
    """
    Maps component pins and tracks their positions for wire routing.

    Features:
    - Pin position calculation based on component placement
    - Connection point determination for wire routing
    - Pin compatibility checking for connections
    - Standard component pin layouts
    """

    # Standard pin layouts for common components
    STANDARD_PIN_LAYOUTS = {
        "resistor": [
            PinInfo("1", "~", PinDirection.PASSIVE, PinType.ELECTRICAL, (-2.54, 0), 2.54, 180),
            PinInfo("2", "~", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), 2.54, 0),
        ],
        "capacitor": [
            PinInfo("1", "~", PinDirection.PASSIVE, PinType.ELECTRICAL, (-2.54, 0), 2.54, 180),
            PinInfo("2", "~", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), 2.54, 0),
        ],
        "inductor": [
            PinInfo("1", "1", PinDirection.PASSIVE, PinType.ELECTRICAL, (-2.54, 0), 2.54, 180),
            PinInfo("2", "2", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), 2.54, 0),
        ],
        "led": [
            PinInfo(
                "1", "K", PinDirection.PASSIVE, PinType.ELECTRICAL, (-2.54, 0), 2.54, 180
            ),  # Cathode
            PinInfo(
                "2", "A", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), 2.54, 0
            ),  # Anode
        ],
        "diode": [
            PinInfo(
                "1", "K", PinDirection.PASSIVE, PinType.ELECTRICAL, (-2.54, 0), 2.54, 180
            ),  # Cathode
            PinInfo(
                "2", "A", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), 2.54, 0
            ),  # Anode
        ],
        "transistor_npn": [
            PinInfo("1", "B", PinDirection.INPUT, PinType.SIGNAL, (-5.08, 0), 2.54, 180),  # Base
            PinInfo(
                "2", "C", PinDirection.PASSIVE, PinType.ELECTRICAL, (0, 2.54), 2.54, 90
            ),  # Collector
            PinInfo(
                "3", "E", PinDirection.PASSIVE, PinType.ELECTRICAL, (0, -2.54), 2.54, 270
            ),  # Emitter
        ],
        "transistor_pnp": [
            PinInfo("1", "B", PinDirection.INPUT, PinType.SIGNAL, (-5.08, 0), 2.54, 180),  # Base
            PinInfo(
                "2", "C", PinDirection.PASSIVE, PinType.ELECTRICAL, (0, 2.54), 2.54, 90
            ),  # Collector
            PinInfo(
                "3", "E", PinDirection.PASSIVE, PinType.ELECTRICAL, (0, -2.54), 2.54, 270
            ),  # Emitter
        ],
        "power": [
            PinInfo(
                "1", "1", PinDirection.POWER_IN, PinType.POWER, (0, 0), 0, 0
            )  # Power connection point
        ],
        "ic": [
            # Generic IC with at least 4 pins (VCC, GND, and 2 I/O)
            PinInfo(
                "1", "Pin1", PinDirection.BIDIRECTIONAL, PinType.SIGNAL, (-7.62, -2.54), 2.54, 180
            ),
            PinInfo(
                "2", "Pin2", PinDirection.BIDIRECTIONAL, PinType.SIGNAL, (-7.62, 2.54), 2.54, 180
            ),
            PinInfo("3", "VCC", PinDirection.POWER_IN, PinType.POWER, (7.62, 2.54), 2.54, 0),
            PinInfo("4", "GND", PinDirection.POWER_IN, PinType.GROUND, (7.62, -2.54), 2.54, 0),
        ],
    }

    def __init__(self):
        """Initialize the pin mapper."""
        self.component_pins: dict[str, list[ComponentPin]] = {}
        self.pin_connections: dict[str, set[str]] = {}  # Track which pins are connected

    def add_component(
        self,
        component_ref: str,
        component_type: str,
        position: tuple[float, float],
        angle: float = 0.0,
        custom_pins: list[PinInfo] | None = None,
    ) -> list[ComponentPin]:
        """
        Add a component and map its pins.

        Args:
            component_ref: Component reference (e.g., 'R1')
            component_type: Type of component for pin layout
            position: Component center position (x, y) in mm
            angle: Component rotation angle in degrees
            custom_pins: Custom pin layout (overrides standard layout)

        Returns:
            List of ComponentPin objects for this component
        """
        # Get pin layout
        pin_layout = custom_pins or self.STANDARD_PIN_LAYOUTS.get(component_type, [])

        # Create ComponentPin objects
        component_pins = []
        for pin_info in pin_layout:
            component_pin = ComponentPin(
                component_ref=component_ref,
                pin_info=pin_info,
                component_position=position,
                component_angle=angle,
            )
            component_pins.append(component_pin)

        # Store pins
        self.component_pins[component_ref] = component_pins

        return component_pins

    def get_component_pins(self, component_ref: str) -> list[ComponentPin]:
        """Get all pins for a component."""
        return self.component_pins.get(component_ref, [])

    def get_pin(self, component_ref: str, pin_number: str) -> ComponentPin | None:
        """Get a specific pin from a component."""
        pins = self.get_component_pins(component_ref)
        for pin in pins:
            if pin.pin_info.number == pin_number:
                return pin
        return None

    def get_pin_connection_point(
        self, component_ref: str, pin_number: str
    ) -> tuple[float, float] | None:
        """Get the connection point for a specific pin."""
        pin = self.get_pin(component_ref, pin_number)
        return pin.connection_point if pin else None

    def can_connect_pins(self, pin1: ComponentPin, pin2: ComponentPin) -> bool:
        """Check if two pins can be electrically connected."""
        # Power pins should connect to compatible pins
        if pin1.pin_info.pin_type == PinType.POWER:
            return pin2.pin_info.direction in [PinDirection.POWER_IN, PinDirection.PASSIVE]

        if pin2.pin_info.pin_type == PinType.POWER:
            return pin1.pin_info.direction in [PinDirection.POWER_IN, PinDirection.PASSIVE]

        # Ground pins
        if pin1.pin_info.pin_type == PinType.GROUND or pin2.pin_info.pin_type == PinType.GROUND:
            return True

        # Signal connections
        if pin1.pin_info.direction == PinDirection.OUTPUT:
            return pin2.pin_info.direction in [
                PinDirection.INPUT,
                PinDirection.BIDIRECTIONAL,
                PinDirection.PASSIVE,
            ]

        if pin2.pin_info.direction == PinDirection.OUTPUT:
            return pin1.pin_info.direction in [
                PinDirection.INPUT,
                PinDirection.BIDIRECTIONAL,
                PinDirection.PASSIVE,
            ]

        # Passive components can connect to anything
        if (
            pin1.pin_info.direction == PinDirection.PASSIVE
            or pin2.pin_info.direction == PinDirection.PASSIVE
        ):
            return True

        # Bidirectional can connect to anything
        return bool(
            pin1.pin_info.direction == PinDirection.BIDIRECTIONAL
            or pin2.pin_info.direction == PinDirection.BIDIRECTIONAL
        )

    def add_connection(
        self, component_ref1: str, pin_number1: str, component_ref2: str, pin_number2: str
    ) -> bool:
        """
        Add a connection between two pins.

        Returns:
            True if connection is valid and added, False otherwise
        """
        pin1 = self.get_pin(component_ref1, pin_number1)
        pin2 = self.get_pin(component_ref2, pin_number2)

        if not pin1 or not pin2:
            return False

        if not self.can_connect_pins(pin1, pin2):
            return False

        # Track the connection
        pin1_id = f"{component_ref1}.{pin_number1}"
        pin2_id = f"{component_ref2}.{pin_number2}"

        if pin1_id not in self.pin_connections:
            self.pin_connections[pin1_id] = set()
        if pin2_id not in self.pin_connections:
            self.pin_connections[pin2_id] = set()

        self.pin_connections[pin1_id].add(pin2_id)
        self.pin_connections[pin2_id].add(pin1_id)

        return True

    def get_connected_pins(self, component_ref: str, pin_number: str) -> list[str]:
        """Get all pins connected to the specified pin."""
        pin_id = f"{component_ref}.{pin_number}"
        return list(self.pin_connections.get(pin_id, set()))

    def calculate_wire_route(
        self, start_pin: ComponentPin, end_pin: ComponentPin, avoid_components: bool = True
    ) -> list[tuple[float, float]]:
        """
        Calculate a wire route between two pins with collision avoidance.

        Args:
            start_pin: Starting pin
            end_pin: Ending pin
            avoid_components: Whether to avoid routing through components

        Returns:
            List of waypoints for the wire route
        """
        start_point = start_pin.connection_point
        end_point = end_pin.connection_point

        # Direct connection for simple cases
        if (
            abs(start_point[0] - end_point[0]) < 1.0 or abs(start_point[1] - end_point[1]) < 1.0
        ):  # Vertically aligned
            return [start_point, end_point]

        # Choose routing strategy based on pin directions and positions
        return self._calculate_orthogonal_route(start_pin, end_pin, avoid_components)

    def _calculate_orthogonal_route(
        self, start_pin: ComponentPin, end_pin: ComponentPin, avoid_components: bool
    ) -> list[tuple[float, float]]:
        """Calculate an orthogonal (L-shaped or stepped) route between pins."""
        start_point = start_pin.connection_point
        end_point = end_pin.connection_point

        # Determine routing preference based on pin angles
        start_angle = start_pin.pin_info.angle + start_pin.component_angle
        end_angle = end_pin.pin_info.angle + end_pin.component_angle

        # Normalize angles to 0-360 range
        start_angle = start_angle % 360
        end_angle = end_angle % 360

        # Choose routing direction based on pin orientations
        if self._should_route_horizontally_first(start_angle, end_angle, start_point, end_point):
            return self._route_horizontal_then_vertical(start_point, end_point)
        else:
            return self._route_vertical_then_horizontal(start_point, end_point)

    def _should_route_horizontally_first(
        self,
        start_angle: float,
        end_angle: float,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
    ) -> bool:
        """Determine if horizontal routing should be preferred."""
        # If start pin points horizontally (0° or 180°), route horizontally first
        if abs(start_angle) < 45 or abs(start_angle - 180) < 45:
            return True

        # If end pin points horizontally, route vertically first to approach horizontally
        if abs(end_angle) < 45 or abs(end_angle - 180) < 45:
            return False

        # For vertical pins, choose based on relative positions
        return abs(start_point[0] - end_point[0]) > abs(start_point[1] - end_point[1])

    def _route_horizontal_then_vertical(
        self, start_point: tuple[float, float], end_point: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Route horizontally first, then vertically."""
        mid_point = (end_point[0], start_point[1])
        return [start_point, mid_point, end_point]

    def _route_vertical_then_horizontal(
        self, start_point: tuple[float, float], end_point: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Route vertically first, then horizontally."""
        mid_point = (start_point[0], end_point[1])
        return [start_point, mid_point, end_point]

    def calculate_bus_route(self, pins: list[ComponentPin]) -> list[list[tuple[float, float]]]:
        """
        Calculate routing for multiple pins connected to a bus.

        Args:
            pins: List of pins to connect to the bus

        Returns:
            List of wire routes, one for each pin connection
        """
        if len(pins) < 2:
            return []

        # Calculate bus line position (average of all pin positions)
        total_x = sum(pin.connection_point[0] for pin in pins)
        total_y = sum(pin.connection_point[1] for pin in pins)
        center_x = total_x / len(pins)
        center_y = total_y / len(pins)

        # Determine bus orientation (horizontal or vertical)
        x_spread = max(pin.connection_point[0] for pin in pins) - min(
            pin.connection_point[0] for pin in pins
        )
        y_spread = max(pin.connection_point[1] for pin in pins) - min(
            pin.connection_point[1] for pin in pins
        )

        routes = []

        if x_spread > y_spread:
            # Horizontal bus
            bus_y = center_y
            for pin in pins:
                pin_point = pin.connection_point
                bus_point = (pin_point[0], bus_y)
                routes.append([pin_point, bus_point])
        else:
            # Vertical bus
            bus_x = center_x
            for pin in pins:
                pin_point = pin.connection_point
                bus_point = (bus_x, pin_point[1])
                routes.append([pin_point, bus_point])

        return routes

    def get_component_statistics(self) -> dict[str, int]:
        """Get statistics about mapped components and pins."""
        total_components = len(self.component_pins)
        total_pins = sum(len(pins) for pins in self.component_pins.values())
        total_connections = len(self.pin_connections)

        return {
            "total_components": total_components,
            "total_pins": total_pins,
            "total_connections": total_connections,
        }

    def clear_mappings(self):
        """Clear all component and pin mappings."""
        self.component_pins.clear()
        self.pin_connections.clear()
