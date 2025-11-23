# KiCAD-MCP Tests

This directory contains test scripts for the KiCAD MCP server functionality.

## Test Scripts

### PCB Parser Tests
- **`test_pcb_parser_debug.py`** - Debug PCB netlist parser, verify pad extraction
- **`find_q200_pads.py`** - Find all pads in Q200 footprint for debugging
- **`extract_footprint_sample.py`** - Extract footprint samples from PCB file
- **`debug_parser_output.py`** - Debug parser output

### Adapter Tests
- **`test_adapter_direct.py`** - Test KiCAD schematic adapter directly
- **`test_adapter_standalone.py`** - Standalone adapter testing
- **`test_battery_charger.py`** - Test battery charger page extraction

### Integration Tests
- **`test_kicad_dsl.py`** - Test KiCAD DSL generation
- **`test_schematic_tools_comprehensive.py`** - Comprehensive test of all 3 schematic DSL tools
  - get_schematic_index()
  - get_schematic_page()
  - get_schematic_context()
- **`test_server_tools.py`** - Test MCP server tool registration

## Test Outputs

Test output files are stored in `tests/outputs/` and are ignored by git.

## Running Tests

```bash
# Run comprehensive test suite
python tests/test_schematic_tools_comprehensive.py

# Test PCB parser
python tests/test_pcb_parser_debug.py

# Test specific page
python tests/test_battery_charger.py
```

## Test Project

Tests use the Astro-DB_rev00005 KiCAD project located at:
```
C:\Users\geoff\Desktop\projects\kicad-astro-daughterboard2\Astro-DB_rev00005\
```

This project contains:
- 13 schematic pages
- 384 components
- Battery charger circuit (108 components, 37 nets)
- Power supplies, connectors, microcontroller, etc.
