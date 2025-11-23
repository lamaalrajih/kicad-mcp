"""
KiCad schematic netlist extraction utilities.
"""
import os
import re
from typing import Any, Dict, List
from collections import defaultdict

class SchematicParser:
    """Parser for KiCad schematic files to extract netlist information."""

    def __init__(self, schematic_path: str, is_hierarchical: bool = True, timeout: float = 30.0):
        """Initialize the schematic parser.

        Args:
            schematic_path: Path to the KiCad schematic file (.kicad_sch)
            is_hierarchical: Whether to parse hierarchical sub-sheets
            timeout: Maximum time in seconds to parse each schematic file (default: 30.0)
        """
        self.schematic_path = schematic_path
        self.is_hierarchical = is_hierarchical
        self.timeout = timeout
        self.content = ""
        self.components = []
        self.labels = []
        self.wires = []
        self.junctions = []
        self.no_connects = []
        self.power_symbols = []
        self.hierarchical_labels = []
        self.global_labels = []
        self.sheets = []  # Hierarchical sheets

        # Netlist information
        self.nets = defaultdict(list)  # Net name -> connected pins
        self.component_pins = {}  # (component_ref, pin_num) -> net_name

        # Component information
        self.component_info = {}  # component_ref -> component details

        # Load the file
        self._load_schematic()

    def _load_schematic(self) -> None:
        """Load the schematic file content."""
        if not os.path.exists(self.schematic_path):
            print(f"Schematic file not found: {self.schematic_path}")
            raise FileNotFoundError(f"Schematic file not found: {self.schematic_path}")
        
        try:
            with open(self.schematic_path, 'r') as f:
                self.content = f.read()
                print(f"Successfully loaded schematic: {self.schematic_path}")
        except Exception as e:
            print(f"Error reading schematic file: {str(e)}")
            raise

    def parse(self) -> Dict[str, Any]:
        """Parse the schematic to extract netlist information.

        Returns:
            Dictionary with parsed netlist information
        """
        import time
        overall_start = time.time()
        print(f"=" * 60)
        print(f"Starting schematic parsing: {os.path.basename(self.schematic_path)}")
        print(f"=" * 60)

        # Extract hierarchical sheets first
        if self.is_hierarchical:
            step_start = time.time()
            self._extract_sheets()
            print(f"[Step 1/8] Extracted sheets in {time.time() - step_start:.2f}s")

        # Extract symbols (components)
        step_start = time.time()
        self._extract_components()
        print(f"[Step 2/8] Extracted components in {time.time() - step_start:.2f}s")

        # Extract wires
        step_start = time.time()
        self._extract_wires()
        print(f"[Step 3/8] Extracted wires in {time.time() - step_start:.2f}s")

        # Extract junctions
        step_start = time.time()
        self._extract_junctions()
        print(f"[Step 4/8] Extracted junctions in {time.time() - step_start:.2f}s")

        # Extract labels
        step_start = time.time()
        self._extract_labels()
        print(f"[Step 5/8] Extracted labels in {time.time() - step_start:.2f}s")

        # Extract power symbols
        step_start = time.time()
        self._extract_power_symbols()
        print(f"[Step 6/8] Extracted power symbols in {time.time() - step_start:.2f}s")

        # Extract no-connects
        step_start = time.time()
        self._extract_no_connects()
        print(f"[Step 7/8] Extracted no-connects in {time.time() - step_start:.2f}s")

        # If this is a hierarchical schematic with sub-sheets, parse them too
        if self.is_hierarchical and self.sheets:
            step_start = time.time()
            print(f"[Step 8/8] Found {len(self.sheets)} hierarchical sheets, parsing sub-schematics...")
            self._parse_hierarchical_sheets()
            print(f"[Step 8/8] Completed hierarchical parsing in {time.time() - step_start:.2f}s")

        # Build netlist
        step_start = time.time()
        self._build_netlist()
        print(f"Built netlist in {time.time() - step_start:.2f}s")

        # Create result
        result = {
            "components": self.component_info,
            "nets": dict(self.nets),
            "labels": self.labels,
            "wires": self.wires,
            "junctions": self.junctions,
            "power_symbols": self.power_symbols,
            "sheets": self.sheets,
            "component_count": len(self.component_info),
            "net_count": len(self.nets)
        }

        total_time = time.time() - overall_start
        print(f"=" * 60)
        print(f"Schematic parsing complete in {total_time:.2f}s")
        print(f"Found {len(self.component_info)} components and {len(self.nets)} nets")
        print(f"=" * 60)
        return result

    def _extract_s_expressions(self, pattern: str) -> List[str]:
        """Extract all matching S-expressions from the schematic content.
        
        Args:
            pattern: Regex pattern to match the start of S-expressions
            
        Returns:
            List of matching S-expressions
        """
        matches = []
        positions = []
        
        # Find all starting positions of matches
        for match in re.finditer(pattern, self.content):
            positions.append(match.start())
        
        # Extract full S-expressions for each match
        for pos in positions:
            # Start from the matching position
            current_pos = pos
            depth = 0
            s_exp = ""
            
            # Extract the full S-expression by tracking parentheses
            while current_pos < len(self.content):
                char = self.content[current_pos]
                s_exp += char
                
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        # Found the end of the S-expression
                        break
                
                current_pos += 1
            
            matches.append(s_exp)
        
        return matches

    def _extract_components(self) -> None:
        """Extract component information from schematic."""
        import time
        print(f"Extracting components from {os.path.basename(self.schematic_path)}")
        start_time = time.time()

        # Find the lib_symbols section and remove it to avoid extracting symbol definitions
        print("  Removing lib_symbols section...")
        lib_start = time.time()
        content_without_lib_symbols = self._remove_lib_symbols_section()
        print(f"  Removed lib_symbols in {time.time() - lib_start:.2f}s")

        # Extract all symbol expressions (components) from the cleaned content
        print("  Extracting symbol expressions...")
        extract_start = time.time()
        symbols = self._extract_s_expressions_from_content(r'\(symbol\s+', content_without_lib_symbols)
        print(f"  Found {len(symbols)} symbols in {time.time() - extract_start:.2f}s")

        print("  Parsing components...")
        parse_start = time.time()
        for idx, symbol in enumerate(symbols, 1):
            if idx % 50 == 0:
                print(f"    Parsing component {idx}/{len(symbols)}...")
            component = self._parse_component(symbol)
            if component:
                self.components.append(component)

                # Add to component info dictionary
                ref = component.get('reference', 'Unknown')
                self.component_info[ref] = component

        total_time = time.time() - start_time
        print(f"Extracted {len(self.components)} components in {total_time:.2f}s")

    def _remove_lib_symbols_section(self) -> str:
        """Remove the lib_symbols section from the content to avoid parsing symbol definitions.

        Returns:
            Content with lib_symbols section removed
        """
        # Find the lib_symbols section
        lib_symbols_match = re.search(r'\(lib_symbols\s*', self.content)
        if not lib_symbols_match:
            return self.content

        # Find the end of the lib_symbols section by tracking parentheses
        start_pos = lib_symbols_match.start()
        current_pos = start_pos
        depth = 0

        while current_pos < len(self.content):
            char = self.content[current_pos]

            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    # Found the end of lib_symbols
                    end_pos = current_pos + 1
                    # Remove the lib_symbols section
                    return self.content[:start_pos] + self.content[end_pos:]

            current_pos += 1

        # If we didn't find the end, return the original content
        return self.content

    def _extract_s_expressions_from_content(self, pattern: str, content: str) -> List[str]:
        """Extract all matching S-expressions from given content.

        Args:
            pattern: Regex pattern to match the start of S-expressions
            content: The content to search in

        Returns:
            List of matching S-expressions
        """
        matches = []
        positions = []

        # Find all starting positions of matches
        for match in re.finditer(pattern, content):
            positions.append(match.start())

        # Extract full S-expressions for each match
        for pos in positions:
            # Start from the matching position
            current_pos = pos
            depth = 0
            s_exp = ""

            # Extract the full S-expression by tracking parentheses
            while current_pos < len(content):
                char = content[current_pos]
                s_exp += char

                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        # Found the end of the S-expression
                        break

                current_pos += 1

            matches.append(s_exp)

        return matches

    def _parse_component(self, symbol_expr: str) -> Dict[str, Any]:
        """Parse a component from a symbol S-expression.
        
        Args:
            symbol_expr: Symbol S-expression
            
        Returns:
            Component information dictionary
        """
        component = {}
        
        # Extract library component ID
        lib_id_match = re.search(r'\(lib_id\s+"([^"]+)"\)', symbol_expr)
        if lib_id_match:
            component['lib_id'] = lib_id_match.group(1)
        
        # Extract reference (e.g., R1, C2)
        property_matches = re.finditer(r'\(property\s+"([^"]+)"\s+"([^"]+)"', symbol_expr)
        for match in property_matches:
            prop_name = match.group(1)
            prop_value = match.group(2)
            
            if prop_name == "Reference":
                component['reference'] = prop_value
            elif prop_name == "Value":
                component['value'] = prop_value
            elif prop_name == "Footprint":
                component['footprint'] = prop_value
            else:
                # Store other properties
                if 'properties' not in component:
                    component['properties'] = {}
                component['properties'][prop_name] = prop_value
        
        # Extract position
        pos_match = re.search(r'\(at\s+([\d\.-]+)\s+([\d\.-]+)(\s+[\d\.-]+)?\)', symbol_expr)
        if pos_match:
            component['position'] = {
                'x': float(pos_match.group(1)),
                'y': float(pos_match.group(2)),
                'angle': float(pos_match.group(3).strip() if pos_match.group(3) else 0)
            }
        
        # Extract pins
        pins = []
        pin_matches = re.finditer(r'\(pin\s+\(num\s+"([^"]+)"\)\s+\(name\s+"([^"]+)"\)', symbol_expr)
        for match in pin_matches:
            pin_num = match.group(1)
            pin_name = match.group(2)
            pins.append({
                'num': pin_num,
                'name': pin_name
            })
        
        if pins:
            component['pins'] = pins
        
        return component

    def _extract_wires(self) -> None:
        """Extract wire information from schematic."""
        print("Extracting wires")
        
        # Extract all wire expressions
        wires = self._extract_s_expressions(r'\(wire\s+')
        
        for wire in wires:
            # Extract the wire coordinates
            pts_match = re.search(r'\(pts\s+\(xy\s+([\d\.-]+)\s+([\d\.-]+)\)\s+\(xy\s+([\d\.-]+)\s+([\d\.-]+)\)\)', wire)
            if pts_match:
                self.wires.append({
                    'start': {
                        'x': float(pts_match.group(1)),
                        'y': float(pts_match.group(2))
                    },
                    'end': {
                        'x': float(pts_match.group(3)),
                        'y': float(pts_match.group(4))
                    }
                })
        
        print(f"Extracted {len(self.wires)} wires")

    def _extract_junctions(self) -> None:
        """Extract junction information from schematic."""
        print("Extracting junctions")
        
        # Extract all junction expressions
        junctions = self._extract_s_expressions(r'\(junction\s+')
        
        for junction in junctions:
            # Extract the junction coordinates
            xy_match = re.search(r'\(junction\s+\(xy\s+([\d\.-]+)\s+([\d\.-]+)\)\)', junction)
            if xy_match:
                self.junctions.append({
                    'x': float(xy_match.group(1)),
                    'y': float(xy_match.group(2))
                })
        
        print(f"Extracted {len(self.junctions)} junctions")

    def _extract_labels(self) -> None:
        """Extract label information from schematic."""
        print("Extracting labels")
        
        # Extract local labels
        local_labels = self._extract_s_expressions(r'\(label\s+')
        
        for label in local_labels:
            # Extract label text and position
            label_match = re.search(r'\(label\s+"([^"]+)"\s+\(at\s+([\d\.-]+)\s+([\d\.-]+)(\s+[\d\.-]+)?\)', label)
            if label_match:
                self.labels.append({
                    'type': 'local',
                    'text': label_match.group(1),
                    'position': {
                        'x': float(label_match.group(2)),
                        'y': float(label_match.group(3)),
                        'angle': float(label_match.group(4).strip() if label_match.group(4) else 0)
                    }
                })
        
        # Extract global labels
        global_labels = self._extract_s_expressions(r'\(global_label\s+')
        
        for label in global_labels:
            # Extract global label text and position
            label_match = re.search(r'\(global_label\s+"([^"]+)"\s+\(shape\s+([^\s\)]+)\)\s+\(at\s+([\d\.-]+)\s+([\d\.-]+)(\s+[\d\.-]+)?\)', label)
            if label_match:
                self.global_labels.append({
                    'type': 'global',
                    'text': label_match.group(1),
                    'shape': label_match.group(2),
                    'position': {
                        'x': float(label_match.group(3)),
                        'y': float(label_match.group(4)),
                        'angle': float(label_match.group(5).strip() if label_match.group(5) else 0)
                    }
                })
        
        # Extract hierarchical labels
        hierarchical_labels = self._extract_s_expressions(r'\(hierarchical_label\s+')
        
        for label in hierarchical_labels:
            # Extract hierarchical label text and position
            label_match = re.search(r'\(hierarchical_label\s+"([^"]+)"\s+\(shape\s+([^\s\)]+)\)\s+\(at\s+([\d\.-]+)\s+([\d\.-]+)(\s+[\d\.-]+)?\)', label)
            if label_match:
                self.hierarchical_labels.append({
                    'type': 'hierarchical',
                    'text': label_match.group(1),
                    'shape': label_match.group(2),
                    'position': {
                        'x': float(label_match.group(3)),
                        'y': float(label_match.group(4)),
                        'angle': float(label_match.group(5).strip() if label_match.group(5) else 0)
                    }
                })
        
        print(f"Extracted {len(self.labels)} local labels, {len(self.global_labels)} global labels, and {len(self.hierarchical_labels)} hierarchical labels")

    def _extract_power_symbols(self) -> None:
        """Extract power symbol information from schematic."""
        print("Extracting power symbols")
        
        # Extract all power symbol expressions
        power_symbols = self._extract_s_expressions(r'\(symbol\s+\(lib_id\s+"power:')
        
        for symbol in power_symbols:
            # Extract power symbol type and position
            type_match = re.search(r'\(lib_id\s+"power:([^"]+)"\)', symbol)
            pos_match = re.search(r'\(at\s+([\d\.-]+)\s+([\d\.-]+)(\s+[\d\.-]+)?\)', symbol)
            
            if type_match and pos_match:
                self.power_symbols.append({
                    'type': type_match.group(1),
                    'position': {
                        'x': float(pos_match.group(1)),
                        'y': float(pos_match.group(2)),
                        'angle': float(pos_match.group(3).strip() if pos_match.group(3) else 0)
                    }
                })
        
        print(f"Extracted {len(self.power_symbols)} power symbols")

    def _extract_no_connects(self) -> None:
        """Extract no-connect information from schematic."""
        print("Extracting no-connects")

        # Extract all no-connect expressions
        no_connects = self._extract_s_expressions(r'\(no_connect\s+')

        for no_connect in no_connects:
            # Extract the no-connect coordinates
            xy_match = re.search(r'\(no_connect\s+\(at\s+([\d\.-]+)\s+([\d\.-]+)\)', no_connect)
            if xy_match:
                self.no_connects.append({
                    'x': float(xy_match.group(1)),
                    'y': float(xy_match.group(2))
                })

        print(f"Extracted {len(self.no_connects)} no-connects")

    def _extract_sheets(self) -> None:
        """Extract hierarchical sheet information from schematic."""
        print("Extracting hierarchical sheets")

        # Extract all sheet expressions
        sheets = self._extract_s_expressions(r'\(sheet\s+')

        for sheet in sheets:
            # Extract sheet file name
            file_match = re.search(r'\(property\s+"Sheetfile"\s+"([^"]+)"', sheet)
            if file_match:
                sheet_file = file_match.group(1)
                self.sheets.append({
                    'file': sheet_file,
                    'path': None  # Will be resolved later
                })

        print(f"Extracted {len(self.sheets)} hierarchical sheets")

    def _parse_hierarchical_sheets(self) -> None:
        """Parse hierarchical sub-sheets and merge their data."""
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        # Get the directory of the current schematic
        schematic_dir = os.path.dirname(self.schematic_path)
        total_sheets = len(self.sheets)

        print(f"Starting hierarchical sheet parsing: {total_sheets} sheets to process")

        for idx, sheet in enumerate(self.sheets, 1):
            sheet_file = sheet['file']
            sheet_path = os.path.join(schematic_dir, sheet_file)
            sheet['path'] = sheet_path

            if not os.path.exists(sheet_path):
                print(f"[{idx}/{total_sheets}] Warning: Sub-schematic not found: {sheet_path}")
                continue

            print(f"[{idx}/{total_sheets}] Starting parse of sub-schematic: {sheet_file}")
            start_time = time.time()

            try:
                # Parse the sub-schematic with timeout enforcement
                print(f"  [{idx}/{total_sheets}] Creating parser for {sheet_file}")

                def parse_sheet():
                    sub_parser = SchematicParser(sheet_path, is_hierarchical=False, timeout=self.timeout)
                    return sub_parser.parse()

                # Use ThreadPoolExecutor to enforce timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(parse_sheet)
                    try:
                        print(f"  [{idx}/{total_sheets}] Calling parse() for {sheet_file} (timeout: {self.timeout}s)")
                        sub_result = future.result(timeout=self.timeout)

                        parse_time = time.time() - start_time
                        print(f"  [{idx}/{total_sheets}] Parse completed in {parse_time:.2f}s")

                        # Merge components from sub-schematic
                        print(f"  [{idx}/{total_sheets}] Merging components from {sheet_file}")
                        merged_count = 0
                        for ref, component in sub_result.get('components', {}).items():
                            if ref not in self.component_info:
                                self.component_info[ref] = component
                                # Mark which sheet this component came from
                                component['sheet'] = sheet_file
                                merged_count += 1
                            else:
                                print(f"  Warning: Duplicate component reference {ref} found in {sheet_file}")

                        print(f"  [{idx}/{total_sheets}] Merged {merged_count} components from {sheet_file}")

                        # Merge nets from sub-schematic
                        print(f"  [{idx}/{total_sheets}] Merging nets from {sheet_file}")
                        merged_nets = 0
                        for net_name, pins in sub_result.get('nets', {}).items():
                            if net_name in self.nets:
                                # Merge pins for existing net
                                self.nets[net_name].extend(pins)
                            else:
                                self.nets[net_name] = pins
                                merged_nets += 1

                        total_time = time.time() - start_time
                        print(f"  [{idx}/{total_sheets}] Completed {sheet_file} in {total_time:.2f}s: {merged_count} components, {merged_nets} new nets")

                    except FutureTimeoutError:
                        error_time = time.time() - start_time
                        print(f"  [{idx}/{total_sheets}] TIMEOUT: Parsing {sheet_file} exceeded {self.timeout}s limit (stopped at {error_time:.2f}s)")
                        raise TimeoutError(f"Schematic parsing timed out after {self.timeout}s for file: {sheet_file}")

            except TimeoutError as e:
                error_time = time.time() - start_time
                print(f"  [{idx}/{total_sheets}] Timeout error parsing {sheet_file} after {error_time:.2f}s")
                raise  # Re-raise to propagate timeout to caller
            except Exception as e:
                error_time = time.time() - start_time
                print(f"  [{idx}/{total_sheets}] Error parsing sub-schematic {sheet_file} after {error_time:.2f}s: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"Completed hierarchical sheet parsing: processed {total_sheets} sheets")

    def _build_netlist(self) -> None:
        """Build the netlist from extracted components and connections."""
        print("Building netlist from schematic data")
        
        # TODO: Implement netlist building algorithm
        # This is a complex task that involves:
        # 1. Tracking connections between components via wires
        # 2. Handling labels (local, global, hierarchical)
        # 3. Processing power symbols
        # 4. Resolving junctions
        
        # For now, we'll implement a basic version that creates a list of nets
        # based on component references and pin numbers
        
        # Process global labels as nets
        for label in self.global_labels:
            net_name = label['text']
            self.nets[net_name] = []  # Initialize empty list for this net
        
        # Process power symbols as nets
        for power in self.power_symbols:
            net_name = power['type']
            if net_name not in self.nets:
                self.nets[net_name] = []
        
        # In a full implementation, we would now trace connections between
        # components, but that requires a more complex algorithm to follow wires
        # and detect connected pins
        
        # For demonstration, we'll add a placeholder note
        print("Note: Full netlist building requires complex connectivity tracing")
        print(f"Found {len(self.nets)} potential nets from labels and power symbols")


def extract_netlist(schematic_path: str, timeout: float = 60.0, is_hierarchical: bool = False) -> Dict[str, Any]:
    """Extract netlist information from a KiCad schematic file.

    Args:
        schematic_path: Path to the KiCad schematic file (.kicad_sch)
        timeout: Maximum time in seconds for parsing (default: 60.0)
        is_hierarchical: Whether to parse hierarchical sub-sheets (default: False for speed)

    Returns:
        Dictionary with netlist information
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    def parse_with_timeout():
        parser = SchematicParser(schematic_path, is_hierarchical=is_hierarchical, timeout=timeout)
        return parser.parse()

    try:
        # Wrap the entire parsing operation with a timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(parse_with_timeout)
            try:
                print(f"Starting schematic parsing with {timeout}s timeout...")
                result = future.result(timeout=timeout)
                print(f"Schematic parsing completed successfully")
                return result
            except FutureTimeoutError:
                print(f"TIMEOUT: Schematic parsing exceeded {timeout}s limit")
                return {
                    "error": f"Timeout: Schematic parsing exceeded {timeout}s limit",
                    "timeout": True,
                    "components": {},
                    "nets": {},
                    "component_count": 0,
                    "net_count": 0
                }
    except TimeoutError as e:
        print(f"Timeout extracting netlist: {str(e)}")
        return {
            "error": f"Timeout: {str(e)}",
            "timeout": True,
            "components": {},
            "nets": {},
            "component_count": 0,
            "net_count": 0
        }
    except Exception as e:
        print(f"Error extracting netlist: {str(e)}")
        return {
            "error": str(e),
            "timeout": False,
            "components": {},
            "nets": {},
            "component_count": 0,
            "net_count": 0
        }


def analyze_netlist(netlist_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze netlist data to provide insights.
    
    Args:
        netlist_data: Dictionary with netlist information
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "component_count": netlist_data.get("component_count", 0),
        "net_count": netlist_data.get("net_count", 0),
        "component_types": defaultdict(int),
        "power_nets": []
    }
    
    # Analyze component types
    for ref, component in netlist_data.get("components", {}).items():
        # Extract component type from reference (e.g., R1 -> R)
        comp_type = re.match(r'^([A-Za-z_]+)', ref)
        if comp_type:
            results["component_types"][comp_type.group(1)] += 1
    
    # Identify power nets
    for net_name in netlist_data.get("nets", {}):
        if any(net_name.startswith(prefix) for prefix in ["VCC", "VDD", "GND", "+5V", "+3V3", "+12V"]):
            results["power_nets"].append(net_name)
    
    # Count pin connections
    total_pins = sum(len(pins) for pins in netlist_data.get("nets", {}).values())
    results["total_pin_connections"] = total_pins
    
    return results
