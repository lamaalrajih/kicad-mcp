"""
Core data models for the Unified Schematic DSL Generator.

This module provides tool-agnostic data structures representing electronic
schematic components, pins, and nets. These models form the foundation for
the schematic core library and are used by all provider adapters.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional
import re


@dataclass
class Pin:
    """
    Represents a single pin on a component.

    Attributes:
        designator: Pin number or identifier (e.g., "1", "22", "A1")
        name: Semantic pin name (e.g., "VCC", "PA9_TX", "" for unnamed pins)
        net: Net name this pin connects to (empty string for no-connects)
    """
    designator: str
    name: str
    net: str


@dataclass
class Component:
    """
    Represents an electronic component in the schematic.

    Attributes:
        refdes: Reference designator (e.g., "U1", "R5", "U1A" for multi-part)
        value: Component value (e.g., "10k", "STM32F407VGT6")
        footprint: PCB footprint name (e.g., "0805", "LQFP-100")
        mpn: Manufacturer Part Number
        page: Page/sheet name this component appears on
        description: Human-readable component description
        pins: List of Pin objects with connectivity information
        location: (x, y) coordinates in mils (captured but not emitted in DSL)
        properties: Additional metadata as key-value pairs
        multipart_parent: Parent component designator for multi-part components
                         (e.g., "U1" for component "U1A")
    """
    refdes: str
    value: str
    footprint: str
    mpn: str
    page: str
    description: str
    pins: List[Pin]
    location: tuple
    properties: Dict[str, str]
    multipart_parent: Optional[str] = None

    def derived_type(self) -> str:
        """
        Map reference designator prefix to standard component type category.

        Returns:
            Component type string (RES, CAP, IND, FUSE, DIODE, TRANSISTOR,
            IC, CONN, SWITCH, OSC, or ACTIVE)
        """
        # Extract prefix from refdes (e.g., "U" from "U1", "FB" from "FB3")
        # Handle multi-part components (e.g., "U1A" -> "U")
        prefix = ""
        for char in self.refdes:
            if char.isalpha():
                prefix += char
            else:
                break

        prefix_upper = prefix.upper()

        # Type mapping table from spec section 3.2
        if prefix_upper == "R":
            return "RES"
        elif prefix_upper == "C":
            return "CAP"
        elif prefix_upper in ("L", "FB"):
            return "IND"
        elif prefix_upper == "F":
            return "FUSE"
        elif prefix_upper in ("D", "LED"):
            return "DIODE"
        elif prefix_upper == "Q":
            return "TRANSISTOR"
        elif prefix_upper == "U":
            return "IC"
        elif prefix_upper in ("J", "P", "CN", "CONN"):
            return "CONN"
        elif prefix_upper == "SW":
            return "SWITCH"
        elif prefix_upper in ("X", "Y"):
            return "OSC"
        else:
            return "ACTIVE"

    def is_complex(self) -> bool:
        """
        Determine if component needs full DEF block in DSL output.

        A component is considered complex if it has more than 4 pins OR
        if any pin has a semantic name (not just simple numeric or
        single-letter designators like "1", "2", "A", "K").

        Returns:
            True if component is complex and needs expanded DSL format,
            False if it's simple and can be inline in net definitions
        """
        # Complex if more than 4 pins
        if len(self.pins) > 4:
            return True

        # Check if any pin has semantic name (not just "1", "2", "3", "4", "A", "K")
        simple_names = {"1", "2", "3", "4", "A", "K", ""}
        for pin in self.pins:
            if pin.name and pin.name not in simple_names:
                return True

        return False

    def is_passive(self) -> bool:
        """
        Determine if component is a passive component type.

        Returns:
            True if component type is RES, CAP, IND, or FUSE
        """
        passive_types = {"RES", "CAP", "IND", "FUSE"}
        return self.derived_type() in passive_types


@dataclass
class Net:
    """
    Represents a net (electrical connection) in the schematic.

    Attributes:
        name: Net name (e.g., "UART_TX", "GND", "Net_U1_5")
        pages: Set of page names where this net appears
        members: List of (component_refdes, pin_designator) tuples
                 representing all pins connected to this net
    """
    name: str
    pages: Set[str] = field(default_factory=set)
    members: List[Tuple[str, str]] = field(default_factory=list)

    def is_global(self) -> bool:
        """
        Determine if net should be summarized rather than fully expanded.

        A net is considered global if it matches common power/ground patterns,
        has many connections (>15), or spans many pages (>3).

        Returns:
            True if net should be summarized in DSL output
        """
        # Check power/ground net naming patterns
        # Pattern matches: GND, PGND, VSS, VCC, VDD, VEE, VBAT,
        # voltage rails like 3V3, 3.3V, +5V, 12V, 1V8, etc.
        # and domain-specific like NET_GND, SIGNAL_VCC, VCC_DIGITAL
        power_pattern = r'^(P?GND|VSS|VCC|VDD|VEE|VBAT)($|_.*)|^(\+?(\d+\.?\d*V\d*|\d*\.?\d*V\d+)|\+?(\d+V))|^.*_(GND|VCC|VDD)$'
        if re.match(power_pattern, self.name, re.IGNORECASE):
            return True

        # More than 15 connections
        if len(self.members) > 15:
            return True

        # More than 3 pages
        if len(self.pages) > 3:
            return True

        return False

    def is_inter_page(self) -> bool:
        """
        Determine if net spans multiple pages.

        Returns:
            True if net appears on more than one page
        """
        return len(self.pages) > 1
