"""
Adapters for various EDA tool formats.

This module contains provider implementations that transform tool-specific
data formats into the unified schematic core data model.

Available Adapters:
    - AltiumJSONAdapter: Parses Altium JSON export format
    - KiCadSExpAdapter: (Future) Parses KiCad S-expression format
"""

from .altium_json import AltiumJSONAdapter

__all__ = ["AltiumJSONAdapter"]
