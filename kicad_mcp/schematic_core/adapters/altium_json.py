"""
Altium JSON adapter for the Unified Schematic DSL Generator.

This adapter transforms Altium's JSON format (exported via GetWholeDesignJSON)
into the unified data model. It handles the mapping from Altium-specific field
names to the tool-agnostic Component, Pin, and Net data structures.

Field Mapping Summary:
    Component:
        - designator → refdes
        - parameters.Comment → value (fallback to parameters.PN)
        - footprint → footprint
        - parameters.PN → mpn
        - sheet → page (filename only, path stripped)
        - description → description
        - schematic_x, schematic_y → location
        - pins array → pins (List[Pin])

    Pin:
        - name → designator
        - name → name (if semantic, not just numeric)
        - net → net (empty string becomes "NC")

    Net:
        - Built from component pin connectivity
        - Aggregates all pins on same net
        - Tracks which pages each net appears on
"""

import json
import os
import re
from typing import List, Dict, Any, Set, Optional
from ..interfaces import SchematicProvider
from ..models import Component, Pin, Net


class AltiumJSONAdapter(SchematicProvider):
    """
    Adapter for Altium JSON format exported from Altium Designer.

    This adapter parses JSON data containing components with their parameters,
    pins, and net connectivity. It transforms this data into the unified model
    used by the schematic core library.

    Usage:
        >>> with open('design.json', 'r') as f:
        >>>     json_data = f.read()
        >>> adapter = AltiumJSONAdapter(json_data)
        >>> adapter.fetch_raw_data()
        >>> components = adapter.get_components()
        >>> nets = adapter.get_nets()

    The JSON format expected:
        {
          "components": [
            {
              "designator": "U1",
              "lib_reference": "STM32F407VGT6",
              "description": "ARM MCU",
              "footprint": "LQFP-100",
              "sheet": "C:\\Path\\To\\Main.SchDoc",
              "schematic_x": 1000,
              "schematic_y": 2000,
              "parameters": {
                "PN": "STM32F407VGT6",
                "MFG": "STMicroelectronics",
                "Comment": "STM32F407VGT6"
              },
              "pins": [
                {"name": "1", "net": "VCC"},
                {"name": "22", "net": "UART_TX"}
              ]
            }
          ],
          "nets": [
            {"name": "VCC"},
            {"name": "UART_TX"}
          ]
        }
    """

    def __init__(self, json_data: str):
        """
        Initialize the adapter with JSON data.

        Args:
            json_data: JSON string containing Altium schematic data

        Raises:
            ValueError: If JSON is malformed or missing required structure
        """
        self._raw_json = json_data
        self._parsed_data: Optional[Dict[str, Any]] = None
        self._ready = False

    def fetch_raw_data(self) -> None:
        """
        Parse the JSON data and prepare for component/net extraction.

        This method parses the JSON string provided during initialization and
        validates the basic structure. Since data is provided at construction,
        this primarily serves to mark the adapter as ready and perform
        validation.

        Raises:
            ValueError: If JSON is malformed or missing required fields
            json.JSONDecodeError: If JSON cannot be parsed
        """
        try:
            self._parsed_data = json.loads(self._raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        # Validate required structure
        if not isinstance(self._parsed_data, dict):
            raise ValueError("JSON root must be an object/dictionary")

        if "components" not in self._parsed_data:
            # If no components key, assume empty design
            self._parsed_data["components"] = []

        if not isinstance(self._parsed_data["components"], list):
            raise ValueError("'components' must be an array")

        self._ready = True

    def get_components(self) -> List[Component]:
        """
        Transform Altium component data into unified Component objects.

        Extracts all components from the parsed JSON and maps Altium-specific
        fields to the unified data model. Handles missing fields gracefully
        with appropriate defaults.

        Returns:
            List of Component objects, one per component in the schematic.
            Returns empty list if no components exist.

        Raises:
            RuntimeError: If called before fetch_raw_data()
        """
        if not self._ready or self._parsed_data is None:
            raise RuntimeError("Must call fetch_raw_data() before get_components()")

        components = []
        for comp_data in self._parsed_data["components"]:
            try:
                component = self._transform_component(comp_data)
                components.append(component)
            except Exception as e:
                # Log warning but continue processing other components
                print(f"Warning: Failed to transform component "
                      f"{comp_data.get('designator', 'UNKNOWN')}: {e}")
                continue

        return components

    def get_nets(self) -> List[Net]:
        """
        Build Net objects from component pin connectivity.

        Analyzes all component pins to construct the net list. Each net
        aggregates all pins connected to it and tracks which pages it appears on.

        The Altium JSON does NOT provide complete net-to-pages mappings directly.
        Instead, we must build nets by examining component pin connectivity.

        Returns:
            List of Net objects representing all nets in the design.
            Returns empty list if no nets exist.

        Raises:
            RuntimeError: If called before fetch_raw_data()
        """
        if not self._ready or self._parsed_data is None:
            raise RuntimeError("Must call fetch_raw_data() before get_nets()")

        # Build nets from component pin connectivity
        nets_dict: Dict[str, Dict[str, Any]] = {}

        for comp_data in self._parsed_data["components"]:
            designator = comp_data.get("designator", "")
            sheet = comp_data.get("sheet", "")
            page_name = self._extract_filename(sheet)

            pins = comp_data.get("pins", [])
            for pin_data in pins:
                net_name = pin_data.get("net", "")

                # Handle empty net name (no-connect)
                if not net_name:
                    net_name = "NC"

                # Initialize net entry if not seen before
                if net_name not in nets_dict:
                    nets_dict[net_name] = {
                        "name": net_name,
                        "pages": set(),
                        "members": []
                    }

                # Add this pin to the net
                pin_designator = pin_data.get("name", "")
                nets_dict[net_name]["members"].append((designator, pin_designator))
                nets_dict[net_name]["pages"].add(page_name)

        # Convert dict to Net objects
        nets = []
        for net_data in nets_dict.values():
            net = Net(
                name=net_data["name"],
                pages=net_data["pages"],
                members=net_data["members"]
            )
            nets.append(net)

        return nets

    def _transform_component(self, comp_data: Dict[str, Any]) -> Component:
        """
        Transform a single Altium component dict to unified Component model.

        Args:
            comp_data: Dictionary containing Altium component data

        Returns:
            Component object with all fields populated

        Raises:
            ValueError: If required field 'designator' is missing
        """
        # Required field
        refdes = comp_data.get("designator", "")
        if not refdes:
            raise ValueError("Component missing required 'designator' field")

        # Extract parameters dict
        parameters = comp_data.get("parameters", {})

        # Map fields with fallbacks
        value = self._get_component_value(comp_data)
        footprint = comp_data.get("footprint", "")
        mpn = parameters.get("PN", "")
        sheet = comp_data.get("sheet", "")
        page = self._extract_filename(sheet)
        description = comp_data.get("description", "")

        # Extract location (x, y)
        x = comp_data.get("schematic_x", 0)
        y = comp_data.get("schematic_y", 0)
        location = (x, y)

        # Transform pins
        pins = self._transform_pins(comp_data.get("pins", []))

        # Build properties dict from parameters (exclude fields we've already mapped)
        properties = {}
        for key, value_str in parameters.items():
            # Skip fields we've already mapped to specific Component attributes
            if key not in ["PN", "Comment"]:
                properties[key] = str(value_str)

        return Component(
            refdes=refdes,
            value=value,
            footprint=footprint,
            mpn=mpn,
            page=page,
            description=description,
            pins=pins,
            location=location,
            properties=properties,
            multipart_parent=None  # Altium sample data doesn't show multi-part
        )

    def _transform_pins(self, pins_data: List[Dict[str, Any]]) -> List[Pin]:
        """
        Transform Altium pin data to unified Pin objects.

        Handles both numeric pin designators ("1", "2") and semantic pin names
        ("VCC", "TX", "S", "G", "D").

        Args:
            pins_data: List of pin dictionaries from Altium JSON

        Returns:
            List of Pin objects
        """
        pins = []
        for pin_data in pins_data:
            pin_designator = pin_data.get("name", "")
            net_name = pin_data.get("net", "")

            # Handle empty net name (no-connect)
            if not net_name:
                net_name = "NC"

            # Determine if pin name is semantic or just numeric
            pin_name = ""
            if self._is_semantic_pin_name(pin_designator):
                pin_name = pin_designator

            pin = Pin(
                designator=pin_designator,
                name=pin_name,
                net=net_name
            )
            pins.append(pin)

        return pins

    def _get_component_value(self, comp_data: Dict[str, Any]) -> str:
        """
        Extract component value with fallback logic.

        Priority:
            1. parameters.Comment
            2. parameters.PN
            3. Empty string

        Args:
            comp_data: Component dictionary from Altium JSON

        Returns:
            Component value string
        """
        parameters = comp_data.get("parameters", {})

        # Try Comment first (most common for value)
        comment = parameters.get("Comment", "")
        if comment:
            return comment

        # Fallback to PN
        pn = parameters.get("PN", "")
        if pn:
            return pn

        # Last resort: empty string
        return ""

    def _extract_filename(self, full_path: str) -> str:
        """
        Extract filename from full Windows or Unix path.

        Handles both forward slashes and backslashes. Returns just the filename
        with extension, stripping all directory components.

        Args:
            full_path: Full path like "C:\\Users\\geoff\\project\\Main.SchDoc"

        Returns:
            Filename only, e.g., "Main.SchDoc"

        Examples:
            >>> _extract_filename("C:\\Users\\project\\Main.SchDoc")
            "Main.SchDoc"
            >>> _extract_filename("/home/user/project/Main.SchDoc")
            "Main.SchDoc"
            >>> _extract_filename("Main.SchDoc")
            "Main.SchDoc"
        """
        if not full_path:
            return ""

        # Use os.path.basename which handles both Windows and Unix paths
        filename = os.path.basename(full_path)
        return filename

    def _is_semantic_pin_name(self, pin_name: str) -> bool:
        """
        Determine if a pin name is semantic (meaningful) vs purely numeric.

        Semantic pin names include things like:
            - "VCC", "GND", "VOUT", "VIN"
            - "TX", "RX", "SDA", "SCL"
            - "S", "G", "D" (for transistors)
            - "A", "K" (for diodes/LEDs) - BUT spec says these are simple
            - "Shell" (for connectors)

        Non-semantic (simple) pin names:
            - Pure numbers: "1", "2", "22", "100"
            - Single letters that are standard simple: "A", "K"
            - Empty string

        Args:
            pin_name: Pin name/designator string

        Returns:
            True if pin name is semantic and should be included in Pin.name,
            False if it's just a numeric/simple designator

        Examples:
            >>> _is_semantic_pin_name("1")
            False
            >>> _is_semantic_pin_name("22")
            False
            >>> _is_semantic_pin_name("VCC")
            True
            >>> _is_semantic_pin_name("TX")
            True
            >>> _is_semantic_pin_name("S")
            True
            >>> _is_semantic_pin_name("A")
            False  # Standard simple anode/cathode
            >>> _is_semantic_pin_name("K")
            False
            >>> _is_semantic_pin_name("")
            False
        """
        if not pin_name:
            return False

        # Pure numeric strings are not semantic
        if pin_name.isdigit():
            return False

        # Single-letter simple pin designators (per spec: "A", "K")
        simple_single_letters = {"A", "K"}
        if pin_name in simple_single_letters:
            return False

        # Everything else is considered semantic
        # This includes:
        #   - Multi-character names: "VCC", "GND", "TX", "RX", "SDA", "SCL"
        #   - Single semantic letters: "S", "G", "D" (transistor pins)
        #   - Special names: "Shell", "NC", etc.
        return True
