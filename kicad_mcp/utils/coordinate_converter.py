"""
Coordinate conversion utilities for KiCad schematic positioning.

This module provides conversion between ComponentLayoutManager coordinates
(mm within A4 bounds) and KiCad's internal coordinate system.
"""

# KiCad coordinate system constants
KICAD_UNITS_PER_MM = 1  # KiCad S-expression format uses millimeters directly
KICAD_MILS_PER_MM = 39.37  # 1mm = 39.37 mils


class CoordinateConverter:
    """Converts between ComponentLayoutManager coordinates and KiCad coordinates."""

    def __init__(self):
        """Initialize the coordinate converter."""
        # KiCad schematic coordinate origin (top-left in KiCad units)
        # A4 sheet dimensions in KiCad units
        self.sheet_width_kicad = 297.0 * KICAD_UNITS_PER_MM  # 29700 units
        self.sheet_height_kicad = 210.0 * KICAD_UNITS_PER_MM  # 21000 units

    def mm_to_kicad_units(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Convert mm coordinates to KiCad internal units.

        Args:
            x_mm: X coordinate in millimeters
            y_mm: Y coordinate in millimeters

        Returns:
            Tuple of (x_kicad, y_kicad) in KiCad units
        """
        x_kicad = x_mm * KICAD_UNITS_PER_MM
        y_kicad = y_mm * KICAD_UNITS_PER_MM
        return (x_kicad, y_kicad)

    def kicad_units_to_mm(self, x_kicad: float, y_kicad: float) -> tuple[float, float]:
        """Convert KiCad internal units to mm coordinates.

        Args:
            x_kicad: X coordinate in KiCad units
            y_kicad: Y coordinate in KiCad units

        Returns:
            Tuple of (x_mm, y_mm) in millimeters
        """
        x_mm = x_kicad / KICAD_UNITS_PER_MM
        y_mm = y_kicad / KICAD_UNITS_PER_MM
        return (x_mm, y_mm)

    def layout_to_kicad(self, x_layout: float, y_layout: float) -> tuple[float, float]:
        """Convert ComponentLayoutManager coordinates to KiCad coordinates.

        ComponentLayoutManager uses mm coordinates within A4 bounds.
        This converts them to KiCad's coordinate system.

        Args:
            x_layout: X coordinate from ComponentLayoutManager (mm)
            y_layout: Y coordinate from ComponentLayoutManager (mm)

        Returns:
            Tuple of (x_kicad, y_kicad) in KiCad units
        """
        # ComponentLayoutManager coordinates are already in mm within A4 bounds
        # Just convert to KiCad units
        return self.mm_to_kicad_units(x_layout, y_layout)

    def kicad_to_layout(self, x_kicad: float, y_kicad: float) -> tuple[float, float]:
        """Convert KiCad coordinates to ComponentLayoutManager coordinates.

        Args:
            x_kicad: X coordinate in KiCad units
            y_kicad: Y coordinate in KiCad units

        Returns:
            Tuple of (x_layout, y_layout) in mm for ComponentLayoutManager
        """
        return self.kicad_units_to_mm(x_kicad, y_kicad)

    def validate_layout_coordinates(self, x_mm: float, y_mm: float) -> bool:
        """Validate that coordinates are within A4 schematic bounds.

        Args:
            x_mm: X coordinate in millimeters
            y_mm: Y coordinate in millimeters

        Returns:
            True if coordinates are within A4 bounds
        """
        # A4 dimensions: 297mm x 210mm
        return 0 <= x_mm <= 297.0 and 0 <= y_mm <= 210.0

    def validate_layout_usable_area(self, x_mm: float, y_mm: float, margin: float = 20.0) -> bool:
        """Validate that coordinates are within usable A4 area (excluding margins).

        Args:
            x_mm: X coordinate in millimeters
            y_mm: Y coordinate in millimeters
            margin: Margin from edges in mm

        Returns:
            True if coordinates are within usable area
        """
        return (margin <= x_mm <= 297.0 - margin) and (margin <= y_mm <= 210.0 - margin)


# Global converter instance for easy access
_converter = CoordinateConverter()


# Convenience functions for easy import
def mm_to_kicad(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Convert mm to KiCad units."""
    return _converter.mm_to_kicad_units(x_mm, y_mm)


def kicad_to_mm(x_kicad: float, y_kicad: float) -> tuple[float, float]:
    """Convert KiCad units to mm."""
    return _converter.kicad_units_to_mm(x_kicad, y_kicad)


def layout_to_kicad(x_layout: float, y_layout: float) -> tuple[float, float]:
    """Convert ComponentLayoutManager coordinates to KiCad."""
    return _converter.layout_to_kicad(x_layout, y_layout)


def validate_position(x_mm: float, y_mm: float, use_margins: bool = True) -> bool:
    """Validate position is within A4 bounds."""
    if use_margins:
        return _converter.validate_layout_usable_area(x_mm, y_mm)
    else:
        return _converter.validate_layout_coordinates(x_mm, y_mm)
