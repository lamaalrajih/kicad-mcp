"""
Symbol library utility functions for KiCad circuit creation.
"""

import glob
import os
from typing import Any

from kicad_mcp.config import KICAD_APP_PATH, KICAD_USER_DIR, system


class SymbolLibraryManager:
    """Manager class for KiCad symbol libraries and symbol operations."""

    def __init__(self):
        self.library_paths = self._discover_library_paths()
        self.symbol_cache = {}

    def _discover_library_paths(self) -> list[str]:
        """Discover KiCad symbol library paths based on the operating system.

        Returns:
            List of paths where symbol libraries can be found
        """
        paths = []

        # Standard KiCad library locations
        if system == "Darwin":  # macOS
            kicad_lib_path = os.path.join(KICAD_APP_PATH, "Contents/SharedSupport/symbols")
            if os.path.exists(kicad_lib_path):
                paths.append(kicad_lib_path)
        elif system == "Windows":
            kicad_lib_path = os.path.join(KICAD_APP_PATH, "share", "kicad", "symbols")
            if os.path.exists(kicad_lib_path):
                paths.append(kicad_lib_path)
        elif system == "Linux":
            for lib_path in ["/usr/share/kicad/symbols", "/usr/local/share/kicad/symbols"]:
                if os.path.exists(lib_path):
                    paths.append(lib_path)

        # User library locations
        user_lib_path = os.path.join(KICAD_USER_DIR, "symbols")
        if os.path.exists(user_lib_path):
            paths.append(user_lib_path)

        return paths

    def get_available_libraries(self) -> list[dict[str, Any]]:
        """Get a list of available symbol libraries.

        Returns:
            List of dictionaries containing library information
        """
        libraries = []

        for lib_path in self.library_paths:
            if not os.path.exists(lib_path):
                continue

            # Find .kicad_sym files
            symbol_files = glob.glob(os.path.join(lib_path, "*.kicad_sym"))

            for symbol_file in symbol_files:
                lib_name = os.path.splitext(os.path.basename(symbol_file))[0]

                # Get library metadata if available
                lib_info = {
                    "name": lib_name,
                    "path": symbol_file,
                    "directory": lib_path,
                    "type": "system" if "usr" in lib_path or "Applications" in lib_path else "user",
                }

                # Try to get symbol count and other metadata
                try:
                    symbol_count = self._count_symbols_in_library(symbol_file)
                    lib_info["symbol_count"] = symbol_count
                except Exception:
                    lib_info["symbol_count"] = "unknown"

                libraries.append(lib_info)

        return libraries

    def _count_symbols_in_library(self, library_file: str) -> int:
        """Count the number of symbols in a library file.

        Args:
            library_file: Path to the .kicad_sym file

        Returns:
            Number of symbols in the library
        """
        try:
            with open(library_file, encoding="utf-8") as f:
                content = f.read()
                # Count symbol definitions (simplified)
                return content.count('(symbol "')
        except Exception:
            return 0

    def search_symbols(
        self, search_term: str, library_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Search for symbols matching a search term.

        Args:
            search_term: Term to search for in symbol names and descriptions
            library_name: Optional specific library to search in

        Returns:
            List of matching symbols with metadata
        """
        results = []
        libraries = self.get_available_libraries()

        if library_name:
            libraries = [lib for lib in libraries if lib["name"] == library_name]

        for library in libraries:
            try:
                symbols = self._parse_library_symbols(library["path"])
                for symbol in symbols:
                    if search_term.lower() in symbol["name"].lower():
                        symbol["library"] = library["name"]
                        symbol["library_path"] = library["path"]
                        results.append(symbol)
            except Exception:
                # Skip libraries that can't be parsed
                continue

        return results

    def _parse_library_symbols(self, library_file: str) -> list[dict[str, Any]]:
        """Parse symbols from a library file.

        Args:
            library_file: Path to the .kicad_sym file

        Returns:
            List of symbol information dictionaries
        """
        symbols = []

        try:
            with open(library_file, encoding="utf-8") as f:
                content = f.read()

            # Simple parsing - look for symbol definitions
            # This is a simplified parser and might not catch all edge cases
            import re

            # Find symbol definitions
            symbol_pattern = r'\(symbol\s+"([^"]+)"\s*\((?:[^()]|(?:\([^()]*\)))*?\)\s*\)'
            matches = re.finditer(symbol_pattern, content, re.DOTALL)

            for match in matches:
                symbol_name = match.group(1)
                symbol_content = match.group(0)

                symbol_info = {
                    "name": symbol_name,
                    "pins": self._extract_pin_info(symbol_content),
                    "properties": self._extract_symbol_properties(symbol_content),
                }

                symbols.append(symbol_info)

        except Exception:
            # Return empty list if parsing fails
            pass

        return symbols

    def _extract_pin_info(self, symbol_content: str) -> list[dict[str, Any]]:
        """Extract pin information from symbol content.

        Args:
            symbol_content: Raw symbol definition content

        Returns:
            List of pin information dictionaries
        """
        pins = []

        try:
            import re

            # Look for pin definitions
            pin_pattern = r'\(pin\s+(\w+)\s+(\w+)\s+\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\)\s*\(length\s+([\d.-]+)\)\s*(?:\(name\s+"([^"]+)"\s*\([^)]*\)\s*)?)(?:\(number\s+"([^"]+)"\s*\([^)]*\)\s*)?'

            for match in re.finditer(pin_pattern, symbol_content):
                pin_info = {
                    "type": match.group(1),
                    "style": match.group(2),
                    "x": float(match.group(3)) if match.group(3) else 0,
                    "y": float(match.group(4)) if match.group(4) else 0,
                    "length": float(match.group(6)) if match.group(6) else 0,
                    "name": match.group(7) if match.group(7) else "",
                    "number": match.group(8) if match.group(8) else "",
                }
                pins.append(pin_info)

        except Exception:
            pass

        return pins

    def _extract_symbol_properties(self, symbol_content: str) -> dict[str, Any]:
        """Extract properties from symbol content.

        Args:
            symbol_content: Raw symbol definition content

        Returns:
            Dictionary of symbol properties
        """
        properties = {}

        try:
            import re

            # Look for property definitions
            prop_pattern = r'\(property\s+"([^"]+)"\s+"([^"]*)"\s*\([^)]*\)\s*\)'

            for match in re.finditer(prop_pattern, symbol_content):
                prop_name = match.group(1)
                prop_value = match.group(2)
                properties[prop_name] = prop_value

        except Exception:
            pass

        return properties

    def get_symbol_info(self, library_name: str, symbol_name: str) -> dict[str, Any] | None:
        """Get detailed information about a specific symbol.

        Args:
            library_name: Name of the library containing the symbol
            symbol_name: Name of the symbol

        Returns:
            Symbol information dictionary or None if not found
        """
        # Find the library
        libraries = self.get_available_libraries()
        target_library = None

        for library in libraries:
            if library["name"] == library_name:
                target_library = library
                break

        if not target_library:
            return None

        # Parse symbols from the library
        try:
            symbols = self._parse_library_symbols(target_library["path"])
            for symbol in symbols:
                if symbol["name"] == symbol_name:
                    symbol["library"] = library_name
                    symbol["library_path"] = target_library["path"]
                    return symbol
        except Exception:
            pass

        return None


def get_common_symbols() -> dict[str, dict[str, Any]]:
    """Get a dictionary of commonly used symbols with their library and placement info.

    Returns:
        Dictionary mapping symbol types to their library information
    """
    return {
        # Basic passive components
        "resistor": {
            "library": "Device",
            "symbol": "R",
            "default_value": "10k",
            "description": "Basic resistor",
        },
        "capacitor": {
            "library": "Device",
            "symbol": "C",
            "default_value": "100nF",
            "description": "Basic capacitor",
        },
        "inductor": {
            "library": "Device",
            "symbol": "L",
            "default_value": "10uH",
            "description": "Basic inductor",
        },
        # Power symbols
        "vcc": {
            "library": "power",
            "symbol": "VCC",
            "default_value": "VCC",
            "description": "VCC power rail",
        },
        "gnd": {
            "library": "power",
            "symbol": "GND",
            "default_value": "GND",
            "description": "Ground symbol",
        },
        "+5v": {
            "library": "power",
            "symbol": "+5V",
            "default_value": "+5V",
            "description": "+5V power rail",
        },
        "+3v3": {
            "library": "power",
            "symbol": "+3V3",
            "default_value": "+3V3",
            "description": "+3.3V power rail",
        },
        # Basic semiconductors
        "led": {
            "library": "Device",
            "symbol": "LED",
            "default_value": "LED",
            "description": "Light emitting diode",
        },
        "diode": {
            "library": "Device",
            "symbol": "D",
            "default_value": "1N4007",
            "description": "Basic diode",
        },
        # Common ICs
        "opamp": {
            "library": "Amplifier_Operational",
            "symbol": "LM358",
            "default_value": "LM358",
            "description": "Dual operational amplifier",
        },
        # Connectors
        "conn_2pin": {
            "library": "Connector",
            "symbol": "Conn_01x02_Male",
            "default_value": "Conn_2Pin",
            "description": "2-pin connector",
        },
        "conn_header": {
            "library": "Connector_Generic",
            "symbol": "Conn_01x04",
            "default_value": "Header_4Pin",
            "description": "4-pin header",
        },
    }


def suggest_footprint_for_symbol(
    symbol_library: str, symbol_name: str, package_hint: str = ""
) -> list[str]:
    """Suggest appropriate footprints for a given symbol.

    Args:
        symbol_library: Library containing the symbol
        symbol_name: Name of the symbol
        package_hint: Optional package type hint (e.g., "0805", "DIP", "SOIC")

    Returns:
        List of suggested footprint library:footprint combinations
    """
    suggestions = []

    # Basic component footprint mappings
    footprint_mappings = {
        "R": [
            "Resistor_SMD:R_0805_2012Metric",
            "Resistor_SMD:R_0603_1608Metric",
            "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal",
        ],
        "C": [
            "Capacitor_SMD:C_0805_2012Metric",
            "Capacitor_SMD:C_0603_1608Metric",
            "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm",
        ],
        "L": [
            "Inductor_SMD:L_0805_2012Metric",
            "Inductor_THT:L_Axial_L5.3mm_D2.2mm_P10.16mm_Horizontal",
        ],
        "LED": ["LED_SMD:LED_0805_2012Metric", "LED_THT:LED_D5.0mm"],
        "D": ["Diode_SMD:D_SOD-123", "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal"],
    }

    # Check for direct symbol name match
    if symbol_name in footprint_mappings:
        suggestions.extend(footprint_mappings[symbol_name])

    # Apply package hints
    if package_hint:
        hint_lower = package_hint.lower()
        filtered_suggestions = []

        for suggestion in suggestions:
            if hint_lower in suggestion.lower():
                filtered_suggestions.append(suggestion)

        if filtered_suggestions:
            suggestions = filtered_suggestions

    return suggestions


def create_symbol_placement_grid(
    start_x: float, start_y: float, spacing: float, components: list[str]
) -> list[tuple[float, float]]:
    """Create a grid layout for component placement.

    Args:
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        spacing: Spacing between components in mm
        components: List of component references

    Returns:
        List of (x, y) coordinates for each component
    """
    positions = []

    # Calculate grid dimensions (try to make roughly square)
    import math

    grid_size = math.ceil(math.sqrt(len(components)))

    for i, _component in enumerate(components):
        row = i // grid_size
        col = i % grid_size

        x = start_x + (col * spacing)
        y = start_y + (row * spacing)

        positions.append((x, y))

    return positions


def validate_symbol_library_reference(library_name: str, symbol_name: str) -> bool:
    """Validate that a symbol exists in the specified library.

    Args:
        library_name: Name of the symbol library
        symbol_name: Name of the symbol

    Returns:
        True if the symbol exists, False otherwise
    """
    try:
        manager = SymbolLibraryManager()
        symbol_info = manager.get_symbol_info(library_name, symbol_name)
        return symbol_info is not None
    except Exception:
        return False


def get_symbol_pin_count(library_name: str, symbol_name: str) -> int:
    """Get the number of pins for a specific symbol.

    Args:
        library_name: Name of the symbol library
        symbol_name: Name of the symbol

    Returns:
        Number of pins, or 0 if symbol not found
    """
    try:
        manager = SymbolLibraryManager()
        symbol_info = manager.get_symbol_info(library_name, symbol_name)

        if symbol_info and "pins" in symbol_info:
            return len(symbol_info["pins"])
    except Exception:
        pass

    return 0
