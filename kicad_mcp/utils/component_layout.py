"""
Component layout manager for KiCad schematics.

Provides intelligent positioning, boundary validation, and automatic layout
capabilities for components in KiCad schematics.
"""

from dataclasses import dataclass
from enum import Enum
import math

from kicad_mcp.config import CIRCUIT_DEFAULTS


class LayoutStrategy(Enum):
    """Layout strategies for automatic component placement."""

    GRID = "grid"
    ROW = "row"
    COLUMN = "column"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"


@dataclass
class ComponentBounds:
    """Component bounding box information."""

    reference: str
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def top(self) -> float:
        return self.y - self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height / 2

    def overlaps_with(self, other: "ComponentBounds") -> bool:
        """Check if this component overlaps with another."""
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )


@dataclass
class SchematicBounds:
    """Schematic sheet boundaries."""

    width: float = 297.0  # A4 width in mm
    height: float = 210.0  # A4 height in mm
    margin: float = 20.0  # Margin from edges in mm

    @property
    def usable_width(self) -> float:
        return self.width - 2 * self.margin

    @property
    def usable_height(self) -> float:
        return self.height - 2 * self.margin

    @property
    def min_x(self) -> float:
        return self.margin

    @property
    def max_x(self) -> float:
        return self.width - self.margin

    @property
    def min_y(self) -> float:
        return self.margin

    @property
    def max_y(self) -> float:
        return self.height - self.margin


class ComponentLayoutManager:
    """
    Manages component layout and positioning for KiCad schematics.

    Features:
    - Boundary validation for component positions
    - Automatic layout generation when positions not specified
    - Grid-based positioning with configurable spacing
    - Collision detection and avoidance
    - Support for different component types and sizes
    """

    # Default component sizes (width, height) in mm
    COMPONENT_SIZES = {
        "resistor": (10.0, 5.0),
        "capacitor": (8.0, 6.0),
        "inductor": (12.0, 8.0),
        "led": (6.0, 8.0),
        "diode": (8.0, 6.0),
        "ic": (20.0, 15.0),
        "transistor": (10.0, 12.0),
        "switch": (12.0, 8.0),
        "connector": (15.0, 10.0),
        "power": (5.0, 5.0),
        "default": (10.0, 8.0),
    }

    def __init__(self, bounds: SchematicBounds | None = None):
        """
        Initialize the layout manager.

        Args:
            bounds: Schematic boundaries to use (defaults to A4)
        """
        self.bounds = bounds or SchematicBounds()
        self.grid_spacing = CIRCUIT_DEFAULTS["grid_spacing"]
        self.component_spacing = CIRCUIT_DEFAULTS["component_spacing"]
        self.placed_components: list[ComponentBounds] = []

    def validate_position(self, x: float, y: float, component_type: str = "default") -> bool:
        """
        Validate that a component position is within schematic boundaries.

        Args:
            x: X coordinate in mm
            y: Y coordinate in mm
            component_type: Type of component for size calculation

        Returns:
            True if position is valid, False otherwise
        """
        width, height = self.COMPONENT_SIZES.get(component_type, self.COMPONENT_SIZES["default"])

        # Check boundaries including component size
        if x - width / 2 < self.bounds.min_x:
            return False
        if x + width / 2 > self.bounds.max_x:
            return False
        if y - height / 2 < self.bounds.min_y:
            return False
        return not y + height / 2 > self.bounds.max_y

    def snap_to_grid(self, x: float, y: float) -> tuple[float, float]:
        """
        Snap coordinates to the nearest grid point.

        Args:
            x: X coordinate in mm
            y: Y coordinate in mm

        Returns:
            Tuple of (snapped_x, snapped_y)
        """
        snapped_x = round(x / self.grid_spacing) * self.grid_spacing
        snapped_y = round(y / self.grid_spacing) * self.grid_spacing
        return snapped_x, snapped_y

    def find_valid_position(
        self,
        component_ref: str,
        component_type: str = "default",
        preferred_x: float | None = None,
        preferred_y: float | None = None,
    ) -> tuple[float, float]:
        """
        Find a valid position for a component, avoiding collisions.

        Args:
            component_ref: Component reference (e.g., 'R1')
            component_type: Type of component
            preferred_x: Preferred X coordinate (optional)
            preferred_y: Preferred Y coordinate (optional)

        Returns:
            Tuple of (x, y) coordinates in mm
        """
        width, height = self.COMPONENT_SIZES.get(component_type, self.COMPONENT_SIZES["default"])

        # If preferred position is provided and valid, try to use it
        if preferred_x is not None and preferred_y is not None:
            x, y = self.snap_to_grid(preferred_x, preferred_y)
            if self.validate_position(x, y, component_type):
                candidate = ComponentBounds(component_ref, x, y, width, height)
                if not self._has_collision(candidate):
                    return x, y

        # Find next available position using grid search
        return self._find_next_grid_position(component_ref, component_type)

    def _find_next_grid_position(
        self, component_ref: str, component_type: str
    ) -> tuple[float, float]:
        """Find the next available grid position."""
        width, height = self.COMPONENT_SIZES.get(component_type, self.COMPONENT_SIZES["default"])

        # Start from top-left of usable area
        start_x = self.bounds.min_x + width / 2
        start_y = self.bounds.min_y + height / 2

        # Search in rows
        current_y = start_y
        while current_y + height / 2 <= self.bounds.max_y:
            current_x = start_x
            while current_x + width / 2 <= self.bounds.max_x:
                x, y = self.snap_to_grid(current_x, current_y)

                # Validate position after grid snapping
                if self.validate_position(x, y, component_type):
                    candidate = ComponentBounds(component_ref, x, y, width, height)
                    if not self._has_collision(candidate):
                        return x, y

                current_x += self.component_spacing

            current_y += self.component_spacing

        # If no position found, place at origin with warning
        return self.snap_to_grid(self.bounds.min_x + width / 2, self.bounds.min_y + height / 2)

    def _has_collision(self, candidate: ComponentBounds) -> bool:
        """Check if candidate component collides with any placed components."""
        return any(candidate.overlaps_with(placed) for placed in self.placed_components)

    def place_component(
        self,
        component_ref: str,
        component_type: str = "default",
        x: float | None = None,
        y: float | None = None,
    ) -> tuple[float, float]:
        """
        Place a component and record its position.

        Args:
            component_ref: Component reference
            component_type: Type of component
            x: X coordinate (optional, will auto-place if not provided)
            y: Y coordinate (optional, will auto-place if not provided)

        Returns:
            Tuple of final (x, y) coordinates
        """
        final_x, final_y = self.find_valid_position(component_ref, component_type, x, y)

        width, height = self.COMPONENT_SIZES.get(component_type, self.COMPONENT_SIZES["default"])
        component_bounds = ComponentBounds(component_ref, final_x, final_y, width, height)
        self.placed_components.append(component_bounds)

        return final_x, final_y

    def auto_layout_components(
        self, components: list[dict], strategy: LayoutStrategy = LayoutStrategy.GRID
    ) -> list[dict]:
        """
        Automatically layout a list of components.

        Args:
            components: List of component dictionaries
            strategy: Layout strategy to use

        Returns:
            List of components with updated positions
        """
        updated_components = []

        if strategy == LayoutStrategy.GRID:
            updated_components = self._layout_grid(components)
        elif strategy == LayoutStrategy.ROW:
            updated_components = self._layout_row(components)
        elif strategy == LayoutStrategy.COLUMN:
            updated_components = self._layout_column(components)
        elif strategy == LayoutStrategy.CIRCULAR:
            updated_components = self._layout_circular(components)
        elif strategy == LayoutStrategy.HIERARCHICAL:
            updated_components = self._layout_hierarchical(components)
        else:
            # Default to individual placement
            for component in components:
                x, y = self.place_component(
                    component["reference"], component.get("component_type", "default")
                )
                component = component.copy()
                component["position"] = (x, y)
                updated_components.append(component)

        return updated_components

    def _layout_grid(self, components: list[dict]) -> list[dict]:
        """Layout components in a grid pattern."""
        updated_components = []

        # Calculate grid dimensions
        num_components = len(components)
        cols = math.ceil(math.sqrt(num_components))
        rows = math.ceil(num_components / cols)

        # Calculate spacing
        available_width = self.bounds.usable_width
        available_height = self.bounds.usable_height

        col_spacing = available_width / max(1, cols - 1) if cols > 1 else available_width / 2
        row_spacing = available_height / max(1, rows - 1) if rows > 1 else available_height / 2

        # Ensure minimum spacing
        col_spacing = max(col_spacing, self.component_spacing)
        row_spacing = max(row_spacing, self.component_spacing)

        # Place components
        for i, component in enumerate(components):
            row = i // cols
            col = i % cols

            x = self.bounds.min_x + col * col_spacing
            y = self.bounds.min_y + row * row_spacing

            # Snap to grid and validate
            x, y = self.snap_to_grid(x, y)
            component_type = component.get("component_type", "default")

            if not self.validate_position(x, y, component_type):
                # Fall back to auto placement
                x, y = self.place_component(component["reference"], component_type)
            else:
                x, y = self.place_component(component["reference"], component_type, x, y)

            updated_component = component.copy()
            updated_component["position"] = (x, y)
            updated_components.append(updated_component)

        return updated_components

    def _layout_row(self, components: list[dict]) -> list[dict]:
        """Layout components in a single row."""
        updated_components = []

        y = self.bounds.min_y + self.bounds.usable_height / 2
        available_width = self.bounds.usable_width
        spacing = available_width / max(1, len(components) - 1) if len(components) > 1 else 0
        spacing = max(spacing, self.component_spacing)

        for i, component in enumerate(components):
            x = self.bounds.min_x + i * spacing
            x, y = self.snap_to_grid(x, y)

            component_type = component.get("component_type", "default")
            x, y = self.place_component(component["reference"], component_type, x, y)

            updated_component = component.copy()
            updated_component["position"] = (x, y)
            updated_components.append(updated_component)

        return updated_components

    def _layout_column(self, components: list[dict]) -> list[dict]:
        """Layout components in a single column."""
        updated_components = []

        # Clear existing components to avoid collision detection issues during layout
        self.clear_layout()

        x = self.bounds.min_x + self.bounds.usable_width / 2
        available_height = self.bounds.usable_height

        # Calculate proper spacing considering component heights
        max_component_height = max(
            self.COMPONENT_SIZES.get(
                comp.get("component_type", "default"), self.COMPONENT_SIZES["default"]
            )[1]
            for comp in components
        )
        min_spacing = max(self.component_spacing, max_component_height + 5.0)  # Add 5mm buffer

        # Use either calculated spacing or minimum spacing, whichever is larger
        if len(components) > 1:
            calculated_spacing = available_height / (len(components) - 1)
            spacing = max(calculated_spacing, min_spacing)
        else:
            spacing = min_spacing

        # Fix the X coordinate for all components in the column
        column_x, _ = self.snap_to_grid(x, 0)

        for i, component in enumerate(components):
            y = self.bounds.min_y + i * spacing
            _, snapped_y = self.snap_to_grid(0, y)

            component_type = component.get("component_type", "default")

            # Force the x-coordinate to stay in the column by bypassing collision detection
            width, height = self.COMPONENT_SIZES.get(
                component_type, self.COMPONENT_SIZES["default"]
            )
            component_bounds = ComponentBounds(
                component["reference"], column_x, snapped_y, width, height
            )
            self.placed_components.append(component_bounds)

            final_x, final_y = column_x, snapped_y

            updated_component = component.copy()
            updated_component["position"] = (final_x, final_y)
            updated_components.append(updated_component)

        return updated_components

    def _layout_circular(self, components: list[dict]) -> list[dict]:
        """Layout components in a circular pattern."""
        updated_components = []

        center_x = self.bounds.min_x + self.bounds.usable_width / 2
        center_y = self.bounds.min_y + self.bounds.usable_height / 2

        # Calculate radius to fit within bounds
        max_radius = min(self.bounds.usable_width, self.bounds.usable_height) / 3

        num_components = len(components)
        angle_step = 2 * math.pi / num_components if num_components > 0 else 0

        for i, component in enumerate(components):
            angle = i * angle_step
            x = center_x + max_radius * math.cos(angle)
            y = center_y + max_radius * math.sin(angle)

            x, y = self.snap_to_grid(x, y)
            component_type = component.get("component_type", "default")

            if not self.validate_position(x, y, component_type):
                x, y = self.place_component(component["reference"], component_type)
            else:
                x, y = self.place_component(component["reference"], component_type, x, y)

            updated_component = component.copy()
            updated_component["position"] = (x, y)
            updated_components.append(updated_component)

        return updated_components

    def _layout_hierarchical(self, components: list[dict]) -> list[dict]:
        """Layout components in a hierarchical pattern based on component types."""
        updated_components = []

        # Group components by type
        type_groups = {}
        for component in components:
            comp_type = component.get("component_type", "default")
            if comp_type not in type_groups:
                type_groups[comp_type] = []
            type_groups[comp_type].append(component)

        # Layout each group in a different area
        num_groups = len(type_groups)
        if num_groups == 0:
            return updated_components

        # Divide schematic into zones
        cols = math.ceil(math.sqrt(num_groups))
        rows = math.ceil(num_groups / cols)

        zone_width = self.bounds.usable_width / cols
        zone_height = self.bounds.usable_height / rows

        for group_index, (comp_type, group_components) in enumerate(type_groups.items()):
            zone_row = group_index // cols
            zone_col = group_index % cols

            zone_x = self.bounds.min_x + zone_col * zone_width
            zone_y = self.bounds.min_y + zone_row * zone_height

            # Create temporary layout manager for this zone
            zone_bounds = SchematicBounds(width=zone_width, height=zone_height, margin=5.0)
            zone_manager = ComponentLayoutManager(zone_bounds)

            # Layout components in this zone
            for component in group_components:
                x, y = zone_manager.place_component(component["reference"], comp_type)
                # Adjust coordinates to global schematic space
                global_x = zone_x + x
                global_y = zone_y + y

                updated_component = component.copy()
                updated_component["position"] = (global_x, global_y)
                updated_components.append(updated_component)

            group_index += 1

        return updated_components

    def get_layout_statistics(self) -> dict:
        """Get statistics about the current layout."""
        if not self.placed_components:
            return {
                "total_components": 0,
                "area_utilization": 0.0,
                "average_spacing": 0.0,
                "bounds_violations": 0,
            }

        total_area = sum(comp.width * comp.height for comp in self.placed_components)
        schematic_area = self.bounds.usable_width * self.bounds.usable_height
        area_utilization = total_area / schematic_area if schematic_area > 0 else 0

        # Calculate average spacing between components
        total_distance = 0
        distance_count = 0
        for i, comp1 in enumerate(self.placed_components):
            for comp2 in self.placed_components[i + 1 :]:
                distance = math.sqrt((comp1.x - comp2.x) ** 2 + (comp1.y - comp2.y) ** 2)
                total_distance += distance
                distance_count += 1

        average_spacing = total_distance / distance_count if distance_count > 0 else 0

        # Check for bounds violations
        bounds_violations = 0
        for comp in self.placed_components:
            if (
                comp.left < self.bounds.min_x
                or comp.right > self.bounds.max_x
                or comp.top < self.bounds.min_y
                or comp.bottom > self.bounds.max_y
            ):
                bounds_violations += 1

        return {
            "total_components": len(self.placed_components),
            "area_utilization": area_utilization,
            "average_spacing": average_spacing,
            "bounds_violations": bounds_violations,
        }

    def clear_layout(self):
        """Clear all placed components."""
        self.placed_components.clear()
