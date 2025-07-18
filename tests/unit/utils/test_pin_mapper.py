"""
Tests for ComponentPinMapper functionality.
"""

from kicad_mcp.utils.pin_mapper import (
    ComponentPin,
    ComponentPinMapper,
    PinDirection,
    PinInfo,
    PinType,
)


class TestPinInfo:
    """Test PinInfo class."""

    def test_pin_info_creation(self):
        """Test PinInfo creation."""
        pin = PinInfo(
            number="1",
            name="Anode",
            direction=PinDirection.PASSIVE,
            pin_type=PinType.ELECTRICAL,
            position=(2.54, 0),
            length=2.54,
            angle=0.0,
        )

        assert pin.number == "1"
        assert pin.name == "Anode"
        assert pin.direction == PinDirection.PASSIVE
        assert pin.pin_type == PinType.ELECTRICAL
        assert pin.position == (2.54, 0)
        assert pin.length == 2.54
        assert pin.angle == 0.0

    def test_connection_point_calculation(self):
        """Test connection point calculation."""
        # Pin pointing right (0 degrees)
        pin = PinInfo(
            "1", "test", PinDirection.PASSIVE, PinType.ELECTRICAL, (0, 0), length=2.54, angle=0.0
        )

        # Component at (10, 10), no rotation
        connection_point = pin.get_connection_point(10.0, 10.0, 0.0)
        expected_x = 10.0 + 0.0 + 2.54  # component_x + pin_x + length * cos(0)
        expected_y = 10.0 + 0.0 + 0.0  # component_y + pin_y + length * sin(0)

        assert abs(connection_point[0] - expected_x) < 0.01
        assert abs(connection_point[1] - expected_y) < 0.01

    def test_connection_point_with_rotation(self):
        """Test connection point with component rotation."""
        # Pin pointing right, component rotated 90 degrees
        pin = PinInfo(
            "1", "test", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), length=2.54, angle=0.0
        )

        connection_point = pin.get_connection_point(10.0, 10.0, 90.0)

        # With 90-degree rotation, the pin should point up
        # Original pin at (2.54, 0) becomes (0, 2.54) after rotation
        # Connection point should be at (10, 10 + 2.54 + 2.54)
        assert abs(connection_point[0] - 10.0) < 0.01
        assert abs(connection_point[1] - (10.0 + 2.54 + 2.54)) < 0.01


class TestComponentPin:
    """Test ComponentPin class."""

    def test_component_pin_creation(self):
        """Test ComponentPin creation."""
        pin_info = PinInfo("1", "test", PinDirection.PASSIVE, PinType.ELECTRICAL, (0, 0))
        component_pin = ComponentPin("R1", pin_info, (50.0, 50.0), 0.0)

        assert component_pin.component_ref == "R1"
        assert component_pin.pin_info == pin_info
        assert component_pin.component_position == (50.0, 50.0)
        assert component_pin.component_angle == 0.0

    def test_connection_point_property(self):
        """Test connection point property."""
        pin_info = PinInfo(
            "1", "test", PinDirection.PASSIVE, PinType.ELECTRICAL, (2.54, 0), length=2.54, angle=0.0
        )
        component_pin = ComponentPin("R1", pin_info, (50.0, 50.0), 0.0)

        connection_point = component_pin.connection_point
        expected_x = 50.0 + 2.54 + 2.54  # component + pin position + length

        assert abs(connection_point[0] - expected_x) < 0.01
        assert abs(connection_point[1] - 50.0) < 0.01


class TestComponentPinMapper:
    """Test ComponentPinMapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = ComponentPinMapper()

    def test_initialization(self):
        """Test mapper initialization."""
        assert len(self.mapper.component_pins) == 0
        assert len(self.mapper.pin_connections) == 0

    def test_standard_pin_layouts(self):
        """Test standard pin layouts."""
        # Check resistor layout
        resistor_pins = self.mapper.STANDARD_PIN_LAYOUTS["resistor"]
        assert len(resistor_pins) == 2
        assert resistor_pins[0].number == "1"
        assert resistor_pins[1].number == "2"

        # Check LED layout
        led_pins = self.mapper.STANDARD_PIN_LAYOUTS["led"]
        assert len(led_pins) == 2
        assert led_pins[0].name == "K"  # Cathode
        assert led_pins[1].name == "A"  # Anode

    def test_add_component(self):
        """Test adding components."""
        # Add a resistor
        pins = self.mapper.add_component("R1", "resistor", (50.0, 50.0))

        assert len(pins) == 2
        assert pins[0].component_ref == "R1"
        assert pins[0].component_position == (50.0, 50.0)

        # Check it's stored
        assert "R1" in self.mapper.component_pins
        assert len(self.mapper.component_pins["R1"]) == 2

    def test_get_component_pins(self):
        """Test getting component pins."""
        # Add component
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))

        # Get pins
        pins = self.mapper.get_component_pins("R1")
        assert len(pins) == 2

        # Non-existent component
        pins = self.mapper.get_component_pins("R999")
        assert len(pins) == 0

    def test_get_specific_pin(self):
        """Test getting specific pin."""
        # Add component
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))

        # Get specific pin
        pin = self.mapper.get_pin("R1", "1")
        assert pin is not None
        assert pin.pin_info.number == "1"

        # Non-existent pin
        pin = self.mapper.get_pin("R1", "999")
        assert pin is None

    def test_pin_connection_point(self):
        """Test getting pin connection point."""
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))

        point = self.mapper.get_pin_connection_point("R1", "1")
        assert point is not None
        assert len(point) == 2

        # Non-existent pin
        point = self.mapper.get_pin_connection_point("R1", "999")
        assert point is None

    def test_pin_compatibility(self):
        """Test pin compatibility checking."""
        # Add components
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))
        self.mapper.add_component("LED1", "led", (100.0, 50.0))
        self.mapper.add_component("VCC", "power", (30.0, 30.0))

        # Get pins
        r1_pin = self.mapper.get_pin("R1", "1")
        led_pin = self.mapper.get_pin("LED1", "1")
        vcc_pin = self.mapper.get_pin("VCC", "1")

        # Passive to passive should work
        assert self.mapper.can_connect_pins(r1_pin, led_pin)

        # Power to passive should work
        assert self.mapper.can_connect_pins(vcc_pin, r1_pin)

    def test_add_connection(self):
        """Test adding connections."""
        # Add components
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))
        self.mapper.add_component("LED1", "led", (100.0, 50.0))

        # Add connection
        result = self.mapper.add_connection("R1", "2", "LED1", "2")
        assert result

        # Check connection is tracked
        connected = self.mapper.get_connected_pins("R1", "2")
        assert "LED1.2" in connected

        # Try invalid connection
        result = self.mapper.add_connection("R1", "999", "LED1", "1")
        assert not result

    def test_wire_routing(self):
        """Test wire routing."""
        # Add components
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))
        self.mapper.add_component("R2", "resistor", (100.0, 50.0))

        # Get pins
        pin1 = self.mapper.get_pin("R1", "2")
        pin2 = self.mapper.get_pin("R2", "1")

        # Calculate route
        route = self.mapper.calculate_wire_route(pin1, pin2)

        assert len(route) >= 2  # At least start and end points
        assert route[0] == pin1.connection_point
        assert route[-1] == pin2.connection_point

    def test_bus_routing(self):
        """Test bus routing for multiple pins."""
        # Add multiple components
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))
        self.mapper.add_component("R2", "resistor", (100.0, 50.0))
        self.mapper.add_component("R3", "resistor", (150.0, 50.0))

        # Get pins
        pins = [
            self.mapper.get_pin("R1", "1"),
            self.mapper.get_pin("R2", "1"),
            self.mapper.get_pin("R3", "1"),
        ]

        # Calculate bus route
        routes = self.mapper.calculate_bus_route(pins)

        assert len(routes) == 3  # One route per pin
        for route in routes:
            assert len(route) >= 2  # Each route has at least 2 points

    def test_statistics(self):
        """Test component statistics."""
        # Initially empty
        stats = self.mapper.get_component_statistics()
        assert stats["total_components"] == 0
        assert stats["total_pins"] == 0
        assert stats["total_connections"] == 0

        # Add components
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))
        self.mapper.add_component("LED1", "led", (100.0, 50.0))

        # Add connection
        self.mapper.add_connection("R1", "2", "LED1", "2")

        stats = self.mapper.get_component_statistics()
        assert stats["total_components"] == 2
        assert stats["total_pins"] == 4  # 2 pins per component
        assert stats["total_connections"] > 0

    def test_clear_mappings(self):
        """Test clearing all mappings."""
        # Add some data
        self.mapper.add_component("R1", "resistor", (50.0, 50.0))
        self.mapper.add_connection("R1", "1", "R1", "2")  # Self-connection for testing

        assert len(self.mapper.component_pins) > 0
        assert len(self.mapper.pin_connections) > 0

        # Clear
        self.mapper.clear_mappings()

        assert len(self.mapper.component_pins) == 0
        assert len(self.mapper.pin_connections) == 0
