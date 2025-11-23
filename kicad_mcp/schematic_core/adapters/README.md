# Altium JSON Adapter

The Altium JSON Adapter transforms Altium Designer's JSON export format into the unified schematic core data model.

## Overview

This adapter bridges the gap between Altium-specific data structures and the tool-agnostic Component, Pin, and Net models used by the schematic core library. It handles field mapping, data validation, and edge case handling to ensure robust parsing of Altium schematic data.

## Installation

No additional dependencies required. The adapter uses only Python standard library modules.

```python
from schematic_core.adapters import AltiumJSONAdapter
```

## Usage

### Basic Usage

```python
import json
from schematic_core.adapters import AltiumJSONAdapter

# Load JSON data (from file or API)
with open('design.json', 'r') as f:
    json_data = f.read()

# Create adapter and parse
adapter = AltiumJSONAdapter(json_data)
adapter.fetch_raw_data()

# Get components and nets
components = adapter.get_components()
nets = adapter.get_nets()

# Use the data
for comp in components:
    print(f"{comp.refdes}: {comp.value} ({comp.derived_type()})")
    for pin in comp.pins:
        print(f"  Pin {pin.designator}: {pin.net}")
```

### Integration with Librarian

```python
from schematic_core.adapters import AltiumJSONAdapter
from schematic_core.librarian import Librarian

# Create adapter
adapter = AltiumJSONAdapter(json_data)

# Create librarian with adapter
librarian = Librarian(adapter)

# Get DSL output
index = librarian.get_index()
page_dsl = librarian.get_page("Main_Sheet")
context_dsl = librarian.get_context(["U1", "U2"])
```

## Data Format

### Expected JSON Structure

```json
{
  "components": [
    {
      "designator": "U1",
      "lib_reference": "STM32F407VGT6",
      "description": "ARM Cortex-M4 MCU",
      "footprint": "LQFP-100",
      "sheet": "C:\\Path\\To\\Main.SchDoc",
      "schematic_x": 1000,
      "schematic_y": 2000,
      "parameters": {
        "PN": "STM32F407VGT6",
        "MFG": "STMicroelectronics",
        "Comment": "STM32F407VGT6",
        "Voltage": "3.3V"
      },
      "pins": [
        {"name": "1", "net": "VCC"},
        {"name": "10", "net": "GND"},
        {"name": "22", "net": "UART_TX"}
      ]
    }
  ],
  "nets": [
    {"name": "VCC"},
    {"name": "GND"},
    {"name": "UART_TX"}
  ]
}
```

### Field Mapping

#### Component Fields

| Altium Field | Unified Model | Fallback/Default | Notes |
|--------------|---------------|------------------|-------|
| `designator` | `refdes` | **Required** | Component reference designator |
| `parameters.Comment` | `value` | `parameters.PN` or `""` | Component value (priority: Comment > PN) |
| `footprint` | `footprint` | `""` | PCB footprint name |
| `parameters.PN` | `mpn` | `""` | Manufacturer Part Number |
| `sheet` | `page` | `""` | Filename extracted from full path |
| `description` | `description` | `""` | Human-readable description |
| `schematic_x`, `schematic_y` | `location` | `(0, 0)` | Tuple of coordinates |
| `parameters.*` | `properties` | `{}` | All other parameters |
| `pins` | `pins` | `[]` | List of Pin objects |

#### Pin Fields

| Altium Field | Unified Model | Transform | Notes |
|--------------|---------------|-----------|-------|
| `name` | `designator` | Direct | Pin number or name |
| `name` | `name` | Semantic check | Only if semantic (not "1", "2", "A", "K") |
| `net` | `net` | Empty → "NC" | Net name or no-connect |

#### Net Building

Nets are **constructed from component pin connectivity**, not provided directly:

```python
# Algorithm:
for component in components:
    page = extract_filename(component.sheet)
    for pin in component.pins:
        net_name = pin.net or "NC"
        nets[net_name].members.append((component.refdes, pin.name))
        nets[net_name].pages.add(page)
```

## Edge Cases Handled

### 1. No-Connect Pins

Pins with empty `net` field are assigned to net "NC":

```json
{"name": "1", "net": ""}  →  Pin(designator="1", name="", net="NC")
```

### 2. Missing Fields

All optional fields have sensible defaults:

```python
# Missing footprint
"footprint": ""  # Default

# Missing parameters
"parameters": {}  # Default

# Missing location
"schematic_x": 0, "schematic_y": 0  # Default
```

### 3. Path Extraction

Full Windows/Unix paths are reduced to filename only:

```python
"C:\\Users\\geoff\\project\\Main.SchDoc"  →  "Main.SchDoc"
"/home/user/project/Main.SchDoc"         →  "Main.SchDoc"
"Main.SchDoc"                            →  "Main.SchDoc"
```

### 4. Semantic Pin Names

Pin name detection distinguishes semantic vs numeric:

```python
"1", "2", "22"     →  numeric (name="")
"VCC", "TX", "S"   →  semantic (name=designator)
"A", "K"           →  simple (name="")  # Standard anode/cathode
```

### 5. Multi-Pin Same Net

Components with multiple pins on the same net (e.g., MOSFET with 4 source pins) preserve all pin entries:

```json
{
  "designator": "Q1",
  "pins": [
    {"name": "S", "net": "VOUT"},
    {"name": "S", "net": "VOUT"},
    {"name": "S", "net": "VOUT"},
    {"name": "G", "net": "GATE"}
  ]
}
```

All pins are preserved. Net "VOUT" will have 3 members from Q1.

### 6. Malformed JSON

Invalid JSON raises descriptive errors:

```python
try:
    adapter.fetch_raw_data()
except ValueError as e:
    print(f"Invalid JSON: {e}")
```

## Implementation Details

### Helper Functions

#### `_extract_filename(full_path: str) -> str`

Extracts filename from full path, handling both Windows and Unix separators:

```python
"C:\\Users\\test\\Main.SchDoc"  →  "Main.SchDoc"
```

#### `_get_component_value(comp_data: dict) -> str`

Extracts component value with fallback logic:

1. Try `parameters.Comment`
2. Fallback to `parameters.PN`
3. Default to empty string

#### `_is_semantic_pin_name(pin_name: str) -> bool`

Determines if pin name is semantic:

- **False**: Pure numeric ("1", "22"), simple letters ("A", "K"), empty string
- **True**: Everything else ("VCC", "TX", "S", "G", "D", "Shell")

### Net Building Algorithm

```python
nets_dict = {}

for component in components:
    page = extract_filename(component.sheet)
    for pin in component.pins:
        net_name = pin.net or "NC"

        if net_name not in nets_dict:
            nets_dict[net_name] = {
                "name": net_name,
                "pages": set(),
                "members": []
            }

        nets_dict[net_name].members.append((component.designator, pin.name))
        nets_dict[net_name].pages.add(page)

# Convert to Net objects
return [Net(**data) for data in nets_dict.values()]
```

## Testing

Comprehensive test suite included in `test_altium_json.py`:

```bash
cd server/schematic_core/adapters
python test_altium_json.py
```

Tests cover:
- Basic component parsing
- Pin transformation (semantic vs numeric)
- No-connect pin handling
- Net building from connectivity
- Missing field handling
- Value fallback logic
- Multi-pin same net scenarios
- Path extraction (Windows/Unix)
- Empty designs
- Malformed JSON
- Sample JSON file validation

## Example Output

Running `example_usage.py` with sample data:

```
Component Details:
  U1: LTC7003EMSE#TRPBF
    Type: IC
    MPN: LTC7003EMSE#TRPBF
    Footprint: MSE16
    Pins: 17
    Complex: True
    Passive: False

  R1: 0.01
    Type: RES
    MPN: LVK12R010DER
    Footprint: LVK-RESC6432X65N
    Pins: 4
    Complex: False
    Passive: True

Net Details:
  GND:
    Members: 6
    Pages: Power_Switches.SchDoc, battery_charger.SchDoc
    Inter-page: True
```

## Performance Considerations

- **JSON parsing**: Done once in `fetch_raw_data()`, cached for subsequent calls
- **Net building**: O(n*p) where n=components, p=average pins per component
- **Memory**: Holds full parsed JSON + transformed objects in memory
- **Recommended**: For large designs (>1000 components), consider batching or streaming

## Error Handling

### Common Errors

1. **Malformed JSON**: `ValueError` with descriptive message
2. **Missing designator**: Warning logged, component skipped
3. **Invalid structure**: `ValueError` during `fetch_raw_data()`
4. **Called before fetch**: `RuntimeError` in `get_components()` / `get_nets()`

### Best Practices

```python
try:
    adapter = AltiumJSONAdapter(json_data)
    adapter.fetch_raw_data()
    components = adapter.get_components()
    nets = adapter.get_nets()
except ValueError as e:
    print(f"JSON parsing error: {e}")
except RuntimeError as e:
    print(f"Adapter usage error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Future Enhancements

Planned improvements:

1. **Multi-part component support**: Detect U1A/U1B patterns, set `multipart_parent`
2. **Validation warnings**: Report suspicious data (e.g., component with no pins)
3. **Hierarchical sheets**: Support for hierarchical design structures
4. **Bulk export optimization**: Direct API integration with Altium MCP server
5. **Streaming parser**: For very large designs (>10,000 components)

## See Also

- [Unified Schematic Core Specification](../../SCHEMATIC_CORE_SPEC.md)
- [Altium Data Mapping Documentation](../../ALTIUM_DATA_MAPPING.md)
- [Models Documentation](../../models.py)
- [Interfaces Documentation](../../interfaces.py)
