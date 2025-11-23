"""
KiCAD Schematic Adapter

Converts KiCAD schematic data to our common schematic model format.
This is the Python equivalent of the DelphiScript used in the Altium adapter.

This adapter parses .kicad_sch files and transforms them into the unified
Component/Pin/Net model used by the schematic core library.
"""
from typing import List, Dict, Any, Set, Optional
from pathlib import Path

from kicad_mcp.utils.netlist_parser import SchematicParser
from kicad_mcp.utils.pcb_netlist_parser import PCBNetlistParser
from kicad_mcp.utils.netlist_xml_extractor import export_and_parse_netlist_xml
from ..interfaces import SchematicProvider
from ..models import Component, Pin, Net


class KiCADSchematicAdapter(SchematicProvider):
    """
    Adapter for KiCAD schematic files (.kicad_sch).

    This adapter implements the SchematicProvider interface to extract
    schematic data from KiCAD projects and transform it into the unified
    data model.

    Usage:
        >>> adapter = KiCADSchematicAdapter("/path/to/project")
        >>> adapter.fetch_raw_data()
        >>> components = adapter.get_components()
        >>> nets = adapter.get_nets()
    """

    def __init__(self, project_root: str):
        """
        Initialize KiCAD adapter.

        Args:
            project_root: Path to the directory containing .kicad_sch files
        """
        self.project_root = Path(project_root)
        self._parsed_sheets: Dict[str, Dict[str, Any]] = {}  # sheet_name -> parsed_data
        self._pcb_netlist: Dict[str, Dict[str, str]] = {}  # refdes -> {pad: net}
        self._xml_netlist: Dict[str, Any] = {}  # XML netlist data with pin connectivity
        self._ready = False

    def fetch_raw_data(self) -> None:
        """
        Parse all .kicad_sch files and extract pin connectivity.

        This method:
        1. Parses .kicad_sch files for component metadata (using SchematicParser)
        2. Exports XML netlist using kicad-cli for pin-to-net connectivity (preferred)
        3. Falls back to PCB file parsing if XML netlist export fails
        4. Merges the data to build complete component information

        Raises:
            FileNotFoundError: If no .kicad_sch files found in project_root
            ValueError: If parsing fails on all schematic files
        """
        # Find all .kicad_sch files
        schematic_files = list(self.project_root.glob("*.kicad_sch"))

        if not schematic_files:
            raise FileNotFoundError(
                f"No .kicad_sch files found in {self.project_root}"
            )

        # Parse each schematic file for component metadata
        successful_parses = 0
        root_schematic = None
        for sch_file in schematic_files:
            sheet_name = sch_file.stem

            try:
                parser = SchematicParser(str(sch_file), is_hierarchical=False)
                parsed_data = parser.parse()
                self._parsed_sheets[sheet_name] = parsed_data
                successful_parses += 1

                # Track the first schematic as root for netlist export
                if root_schematic is None:
                    root_schematic = sch_file

            except Exception as e:
                print(f"Warning: Failed to parse {sheet_name}: {e}")
                continue

        if successful_parses == 0:
            raise ValueError(
                f"Failed to parse any .kicad_sch files in {self.project_root}"
            )

        # Try XML netlist export first (preferred method)
        xml_netlist_success = False
        if root_schematic:
            try:
                print(f"Exporting XML netlist from {root_schematic.name}...")
                self._xml_netlist = export_and_parse_netlist_xml(str(root_schematic))
                xml_component_count = self._xml_netlist.get('component_count', 0)
                print(f"Extracted {xml_component_count} components with pin connectivity from XML netlist")
                xml_netlist_success = True

                # Convert XML netlist format to PCB netlist format for compatibility
                for comp_ref, comp_data in self._xml_netlist.get('components', {}).items():
                    pins_dict = {}
                    for pin_num, pin_info in comp_data.get('pins', {}).items():
                        pins_dict[pin_num] = pin_info.get('net', '')
                    if pins_dict:
                        self._pcb_netlist[comp_ref] = pins_dict

            except Exception as e:
                print(f"Warning: Failed to export XML netlist: {e}")
                print("Falling back to PCB file netlist parsing...")

        # Fallback to PCB file for netlist connectivity (if XML failed)
        if not xml_netlist_success:
            pcb_files = list(self.project_root.glob("*.kicad_pcb"))
            if pcb_files:
                try:
                    print(f"Parsing PCB netlist from {pcb_files[0].name}...")
                    pcb_parser = PCBNetlistParser(str(pcb_files[0]))
                    self._pcb_netlist = pcb_parser.parse()
                    print(f"Extracted connectivity for {len(self._pcb_netlist)} components")
                except Exception as e:
                    print(f"Warning: Failed to parse PCB netlist: {e}")
                    print("Continuing without pin connectivity data...")
            else:
                print("Warning: No .kicad_pcb file found - pin connectivity unavailable")

        self._ready = True

    def get_components(self) -> List[Component]:
        """
        Extract all components from parsed KiCAD schematics.

        Transforms KiCAD component data into unified Component objects with
        Pin connectivity information.

        Returns:
            List of Component objects from all schematic sheets

        Raises:
            RuntimeError: If called before fetch_raw_data()
        """
        if not self._ready:
            raise RuntimeError("Must call fetch_raw_data() before get_components()")

        components = []

        for sheet_name, parsed_data in self._parsed_sheets.items():
            comp_info = parsed_data.get("components", {})

            for comp_ref, comp_data in comp_info.items():
                try:
                    component = self._transform_component(
                        comp_ref,
                        comp_data,
                        sheet_name
                    )
                    components.append(component)
                except Exception as e:
                    print(f"Warning: Failed to transform component {comp_ref}: {e}")
                    continue

        return components

    def get_nets(self) -> List[Net]:
        """
        Build net list from PCB netlist connectivity.

        Uses the PCB netlist to build the complete net list with pin connectivity.
        Maps nets to their originating schematic pages.

        Returns:
            List of Net objects representing all nets in the design

        Raises:
            RuntimeError: If called before fetch_raw_data()
        """
        if not self._ready:
            raise RuntimeError("Must call fetch_raw_data() before get_nets()")

        # Build component-to-page mapping
        comp_to_page: Dict[str, str] = {}
        for sheet_name, parsed_data in self._parsed_sheets.items():
            comp_info = parsed_data.get("components", {})
            for comp_ref in comp_info.keys():
                comp_to_page[comp_ref] = sheet_name

        # Build nets from PCB netlist connectivity
        nets_dict: Dict[str, Dict[str, Any]] = {}

        for comp_ref, pin_nets in self._pcb_netlist.items():
            # Get the page this component is on
            page = comp_to_page.get(comp_ref, "unknown")

            for pin_num, net_name in pin_nets.items():
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
                nets_dict[net_name]["members"].append((comp_ref, pin_num))
                nets_dict[net_name]["pages"].add(page)

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

    def _transform_component(
        self,
        comp_ref: str,
        comp_data: Dict[str, Any],
        sheet_name: str
    ) -> Component:
        """
        Transform a single KiCAD component to unified Component model.

        Args:
            comp_ref: Component reference designator (e.g., "U1", "R5")
            comp_data: Dictionary containing KiCAD component data
            sheet_name: Name of the sheet this component appears on

        Returns:
            Component object with all fields populated
        """
        # Extract basic fields
        value = comp_data.get("value", "")
        footprint = comp_data.get("footprint", "")

        # KiCAD sometimes stores MPN in datasheet field or properties
        datasheet = comp_data.get("datasheet", "")
        properties_dict = comp_data.get("properties", {})

        # Try to get MPN from properties first, then datasheet
        mpn = properties_dict.get("PN", "")
        if not mpn:
            mpn = properties_dict.get("MPN", "")
        if not mpn and datasheet and not datasheet.startswith("http"):
            mpn = datasheet

        # Build description from lib_id or properties
        lib_id = comp_data.get("lib_id", "")
        description = properties_dict.get("Description", "")
        if not description and lib_id:
            description = lib_id.split(":")[-1]

        # Transform pins - use PCB netlist if available
        pins = self._transform_pins(comp_ref, comp_data.get("pins", {}))

        # Location from position if available
        position = comp_data.get("position", {})
        x = position.get("x", 0)
        y = position.get("y", 0)
        location = (x, y)

        # Additional properties
        properties = {}
        if lib_id:
            properties["lib_id"] = lib_id
        if datasheet:
            properties["datasheet"] = datasheet
        # Add all component properties
        properties.update(properties_dict)

        return Component(
            refdes=comp_ref,
            value=value,
            footprint=footprint,
            mpn=mpn,
            page=sheet_name,
            description=description,
            pins=pins,
            location=location,
            properties=properties,
            multipart_parent=None  # KiCAD handles multi-unit differently
        )

    def _transform_pins(self, comp_ref: str, pins_data: Dict[str, Dict[str, Any]]) -> List[Pin]:
        """
        Transform KiCAD pin data to unified Pin objects.

        Uses PCB netlist data for pin connectivity if available,
        otherwise falls back to schematic data.

        Args:
            comp_ref: Component reference designator
            pins_data: Dictionary of pin_number -> pin_info (may be empty)

        Returns:
            List of Pin objects
        """
        pins = []

        # Get pin connectivity from PCB netlist
        pcb_pins = self._pcb_netlist.get(comp_ref, {})

        if pcb_pins:
            # Use PCB netlist data (most reliable)
            for pin_num, net_name in pcb_pins.items():
                # Try to get pin name from schematic data if available
                pin_sch_data = pins_data.get(pin_num, {})
                pin_name = pin_sch_data.get("name", pin_num)

                # Handle empty net name (no-connect)
                if not net_name:
                    net_name = "NC"

                # Determine if pin name is semantic or just numeric
                semantic_name = ""
                if self._is_semantic_pin_name(pin_name):
                    semantic_name = pin_name

                pin = Pin(
                    designator=pin_num,
                    name=semantic_name,
                    net=net_name
                )
                pins.append(pin)
        elif pins_data:
            # Fallback to schematic data (if available)
            for pin_num, pin_data in pins_data.items():
                net_name = pin_data.get("net", "")
                pin_name = pin_data.get("name", pin_num)

                # Handle empty net name (no-connect)
                if not net_name:
                    net_name = "NC"

                # Determine if pin name is semantic or just numeric
                semantic_name = ""
                if self._is_semantic_pin_name(pin_name):
                    semantic_name = pin_name

                pin = Pin(
                    designator=pin_num,
                    name=semantic_name,
                    net=net_name
                )
                pins.append(pin)

        return pins

    def _is_semantic_pin_name(self, pin_name: str) -> bool:
        """
        Determine if a pin name is semantic (meaningful) vs purely numeric.

        Semantic pin names include:
            - Power pins: "VCC", "GND", "VOUT", "VIN"
            - Signal pins: "TX", "RX", "SDA", "SCL"
            - Transistor pins: "S", "G", "D"

        Non-semantic (simple) pin names:
            - Pure numbers: "1", "2", "22"
            - Standard simple: "A", "K"

        Args:
            pin_name: Pin name string

        Returns:
            True if pin name is semantic, False if simple
        """
        if not pin_name:
            return False

        # Pure numeric strings are not semantic
        if pin_name.isdigit():
            return False

        # Single-letter simple pin designators (per spec)
        simple_single_letters = {"A", "K"}
        if pin_name in simple_single_letters:
            return False

        # Everything else is semantic
        return True
