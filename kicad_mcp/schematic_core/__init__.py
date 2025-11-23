"""
Unified Schematic Core - Tool-Agnostic Schematic Data Model

This package provides a unified data model and DSL generator for electronic
schematics, supporting multiple EDA tools through a hexagonal architecture.

Main Components:
    - models: Core data structures (Component, Net, Pin)
    - interfaces: Abstract provider interface (SchematicProvider)
    - librarian: State manager and navigation (to be implemented)
    - dsl_emitter: DSL v0.3 text generation (implemented)
    - adapters: Tool-specific adapters (Altium, KiCad, etc.)

Example Usage:
    from schematic_core import Component, Net, Pin
    from schematic_core.interfaces import SchematicProvider
    from schematic_core.adapters.altium_json import AltiumJSONAdapter

    # Create adapter for your EDA tool
    provider = AltiumJSONAdapter(mcp_client)

    # Fetch and process data
    provider.fetch_raw_data()
    components = provider.get_components()
    nets = provider.get_nets()
"""

from .models import Pin, Component, Net
from .interfaces import SchematicProvider
from .dsl_emitter import emit_page_dsl, emit_context_dsl

__all__ = [
    'Pin',
    'Component',
    'Net',
    'SchematicProvider',
    'emit_page_dsl',
    'emit_context_dsl',
]

__version__ = '0.3.0'
__author__ = 'Altium MCP Team'
