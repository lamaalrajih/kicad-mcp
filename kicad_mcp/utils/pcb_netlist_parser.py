"""
KiCAD PCB Netlist Parser

Extracts pin-to-net connectivity from .kicad_pcb files.
The PCB file contains the complete netlist that was imported from the schematic.
"""
import re
from typing import Dict, List, Tuple, Set
from pathlib import Path


class PCBNetlistParser:
    """Parser for extracting netlist from KiCAD PCB files (.kicad_pcb)"""

    def __init__(self, pcb_file_path: str):
        """
        Initialize the PCB netlist parser.

        Args:
            pcb_file_path: Path to the .kicad_pcb file
        """
        self.pcb_file_path = Path(pcb_file_path)
        self.content = ""

        # Parsed data
        self.net_names: Dict[int, str] = {}  # net_id -> net_name
        self.component_pads: Dict[str, Dict[str, str]] = {}  # refdes -> {pad_num: net_name}

        self._load_pcb_file()

    def _load_pcb_file(self) -> None:
        """Load the PCB file content."""
        if not self.pcb_file_path.exists():
            raise FileNotFoundError(f"PCB file not found: {self.pcb_file_path}")

        with open(self.pcb_file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    def parse(self) -> Dict[str, Dict[str, str]]:
        """
        Parse the PCB file to extract pin-to-net connectivity.

        Returns:
            Dictionary mapping component refdes to pad connectivity:
            {
                "R213": {
                    "1": "VCC_3.3",
                    "2": "/battery_charger/CHARGER_INTB"
                },
                ...
            }
        """
        # Extract net definitions
        self._extract_net_definitions()

        # Extract footprints and pad connectivity
        self._extract_footprints()

        return self.component_pads

    def _extract_net_definitions(self) -> None:
        """
        Extract net ID to name mappings.

        Format: (net 371 "VCC_3.3")
        """
        # Pattern matches: (net <number> "<name>")
        net_pattern = r'\(net\s+(\d+)\s+"([^"]*)"\)'

        for match in re.finditer(net_pattern, self.content):
            net_id = int(match.group(1))
            net_name = match.group(2)
            self.net_names[net_id] = net_name

    def _extract_footprints(self) -> None:
        """
        Extract footprints and their pad-to-net assignments.

        Footprint format:
            (footprint "..."
                (property "Reference" "R213" ...)
                (pad "1" ... (net 371 "VCC_3.3") ...)
                (pad "2" ... (net 404 "/battery_charger/CHARGER_INTB") ...)
            )
        """
        # Find all footprint blocks
        footprint_blocks = self._extract_s_expressions(r'\(footprint\s+"[^"]*"')

        for footprint_block in footprint_blocks:
            # Extract reference designator
            ref_match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', footprint_block)
            if not ref_match:
                continue

            refdes = ref_match.group(1)

            # Extract all pads and their nets using proper S-expression parsing
            pad_nets = {}

            # Find all pad S-expressions within this footprint
            for pad_match in re.finditer(r'\(pad\s+"([^"]+)"', footprint_block):
                pad_num = pad_match.group(1)
                pad_start = pad_match.start()

                # Extract the complete pad S-expression using balanced parentheses
                depth = 0
                pad_end = pad_start
                for i in range(pad_start, len(footprint_block)):
                    if footprint_block[i] == '(':
                        depth += 1
                    elif footprint_block[i] == ')':
                        depth -= 1
                        if depth == 0:
                            pad_end = i + 1
                            break

                pad_block = footprint_block[pad_start:pad_end]

                # Extract net from this specific pad block
                net_match = re.search(r'\(net\s+(\d+)\s+"([^"]*)"\)', pad_block)
                if net_match:
                    net_id = int(net_match.group(1))
                    net_name = net_match.group(2)

                    # For pads with same number (multiple instances), keep first one
                    # This handles PowerPAK footprints with multiple physical pads per pin
                    if pad_num not in pad_nets:
                        pad_nets[pad_num] = net_name

            if pad_nets:
                self.component_pads[refdes] = pad_nets

    def _extract_s_expressions(self, pattern: str) -> List[str]:
        """
        Extract all matching S-expressions from the PCB content.

        Args:
            pattern: Regex pattern to match the start of S-expressions

        Returns:
            List of matching S-expressions (balanced parentheses)
        """
        matches = []
        positions = []

        # Find all starting positions
        for match in re.finditer(pattern, self.content):
            positions.append(match.start())

        # Extract balanced S-expressions
        for start_pos in positions:
            depth = 0
            end_pos = start_pos

            for i in range(start_pos, len(self.content)):
                if self.content[i] == '(':
                    depth += 1
                elif self.content[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

            if end_pos > start_pos:
                matches.append(self.content[start_pos:end_pos])

        return matches

    def get_component_nets(self, refdes: str) -> Dict[str, str]:
        """
        Get pad-to-net mappings for a specific component.

        Args:
            refdes: Component reference designator

        Returns:
            Dictionary mapping pad numbers to net names
        """
        return self.component_pads.get(refdes, {})

    def get_all_nets(self) -> Set[str]:
        """
        Get set of all unique net names in the design.

        Returns:
            Set of net names
        """
        nets = set()
        for pad_nets in self.component_pads.values():
            nets.update(pad_nets.values())
        return nets

    def get_net_members(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Build net membership list (which pins are on each net).

        Returns:
            Dictionary mapping net names to list of (refdes, pad_num) tuples
        """
        net_members: Dict[str, List[Tuple[str, str]]] = {}

        for refdes, pad_nets in self.component_pads.items():
            for pad_num, net_name in pad_nets.items():
                if net_name not in net_members:
                    net_members[net_name] = []
                net_members[net_name].append((refdes, pad_num))

        return net_members
