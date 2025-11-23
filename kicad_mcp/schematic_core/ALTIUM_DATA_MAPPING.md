# Altium Data Structure Mapping

**Date:** 2025-11-21
**Source:** Astro-DB project (409 components, 13 schematic pages)

## Overview

This document describes the JSON data structure returned by the Altium MCP server and how it maps to the Unified Schematic Core data model.

---

## Data Sources

### 1. `get_schematic_components_with_parameters()`

Returns array of all components with parameters but **NO pin/net data**.

```json
{
  "designator": "U1",
  "lib_reference": "LTC7003EMSE#TRPBF",
  "description": "IC GATE DRVR N-CH MOSFET 16 MSOP",
  "footprint": "MSE16",
  "parameters": {
    "MFG": "Linear Tech",
    "PN": "LTC7003EMSE#TRPBF",
    "Comment": "LTC7003EMSE#TRPBF",
    "Description": "IC GATE DRVR N-CH MOSFET 16 MSOP",
    "Component Kind": "Standard",
    "Library Name": "SAMD21_Xplained_Pro.SCHLIB"
  }
}
```

### 2. `get_schematic_data(cmp_designators)`

Returns component location and parameters but **NO pin/net data**.

```json
{
  "designator": "U1",
  "sheet": "C:\\Users\\geoff\\...\\Power_Switches.SchDoc",
  "schematic_x": 6100,
  "schematic_y": 3700,
  "schematic_width": 2440,
  "schematic_height": -2240,
  "schematic_rotation": 0,
  "parameters": {
    "MFG": "Linear Tech",
    "PN": "LTC7003EMSE#TRPBF",
    "Comment": "LTC7003EMSE#TRPBF"
  }
}
```

### 3. `get_component_pins(cmp_designators)`

Returns **pin connectivity data** - THIS IS CRITICAL!

```json
{
  "designator": "U1",
  "pins": [
    {
      "name": "1",
      "net": "NetR46_1",
      "x": -1145.527,
      "y": -2510.185,
      "rotation": 90,
      "layer": "TopLayer",
      "width": 40,
      "height": 11.6,
      "shape": "Rectangular"
    },
    {
      "name": "17",
      "net": "GND",
      "x": -1076.628,
      "y": -2417.685,
      "rotation": 0,
      "layer": "TopLayer",
      "width": 112,
      "height": 65,
      "shape": "Rectangular"
    }
  ]
}
```

**Key observations:**
- Pin `name` is usually a number ("1", "2", "17") but can be semantic ("S", "G", "D" for FETs, "Shell" for connectors)
- Pin `net` is the net name (e.g., "GND", "FUSE_VOUT", "NetR46_1")
- PCB coordinates (x, y) are in mils
- Some pins share the same net (e.g., Q1 has multiple "S" pins all on "CHARGER_VOUT")

### 4. `get_all_nets()`

Returns simple array of net names:

```json
[
  {"name": "GND"},
  {"name": "VCC_3.3"},
  {"name": "UART0_TXD"},
  {"name": "NetR46_1"}
]
```

---

## Mapping to Unified Model

### Component Mapping

| Altium Field | Unified Model Field | Notes |
|--------------|---------------------|-------|
| `designator` | `refdes` | Direct mapping |
| `parameters.Comment` | `value` | Best source for component value |
| `footprint` | `footprint` | Direct mapping |
| `parameters.PN` | `mpn` | Manufacturer Part Number |
| `sheet` | `page` | Extract filename only (e.g., "Power_Switches.SchDoc") |
| `description` | `description` | Direct mapping |
| `schematic_x`, `schematic_y` | `location` | Tuple (x, y) - capture but don't emit in DSL |
| `parameters.*` | `properties` | All other parameters go here |

### Pin Mapping

| Altium Field | Unified Model Field | Notes |
|--------------|---------------------|-------|
| `name` | `designator` | Pin number/name |
| `net` | `net` | Net name |
| N/A | `name` | For semantic pin names, copy from `name` field |

**Pin Name Heuristic:**
- If `name` is numeric ("1", "22"): Use it as designator, leave name empty
- If `name` is semantic ("VCC", "TX", "S", "G", "D"): Use as both designator AND name
- Multi-part pins (e.g., FET with 4 "S" pins): Keep all, DSL will handle deduplication

### Net Mapping

| Altium Field | Unified Model Field | Notes |
|--------------|---------------------|-------|
| `name` | `name` | Direct mapping |
| N/A | `pages` | Build from component pin analysis |
| N/A | `members` | Build from pin connectivity: [(refdes, pin_designator), ...] |

---

## Multi-Part Component Handling

**Observation from Q1 (MOSFET):**
```json
{
  "designator": "Q1",
  "pins": [
    {"name": "S", "net": "CHARGER_VOUT"},
    {"name": "S", "net": "CHARGER_VOUT"},
    {"name": "S", "net": "CHARGER_VOUT"},
    {"name": "G", "net": "NetE8_1"},
    {"name": "D", "net": "FUSE_VIN"},
    {"name": "D", "net": "FUSE_VIN"},
    {"name": "D", "net": "FUSE_VIN"},
    {"name": "D", "net": "FUSE_VIN"}
  ]
}
```

**Decision:** Treat as single component with repeated pin names. In DSL, show as:
```
COMP Q1 (SI4459BDY-T1-GE3)
  PINS:
    S: (x3) CHARGER_VOUT
    G: NetE8_1
    D: (x4) FUSE_VIN
```

---

## Net Name Patterns

### Auto-Generated Nets
Format: `NetCOMP_PIN` (e.g., "NetR46_1", "NetC1_1", "NetLED1_3")

These are unnamed nets that Altium auto-generates. They connect exactly 2 pins.

### Named Nets
User-assigned names like:
- Power: "GND", "VCC_3.3", "FUSE_VOUT", "GIMBAL_PWR"
- Signals: "UART0_TXD", "HDMI_SCL", "CHARGER_SDA"
- Internal: "NetE8_1" (auto-generated but might cross pages)

### Global Net Detection
Based on the actual net list, these nets appear on many pages:
- GND (appears in almost every component)
- VCC_3.3, VCC_ASTRO_3.3, VCC_MCU_CORE
- Power domain nets: FUSE_VIN, FUSE_VOUT, CHARGER_VOUT

---

## Page Names

**Format:** Filename only (not full path)

Examples:
- `power_input.SchDoc`
- `battery_charger.SchDoc`
- `Power_Switches.SchDoc`
- `uC_Peripherals.SchDoc`
- `Astro_Connectors1.SchDoc`
- `Bluetooth.SchDoc`
- `LED_Drivers.SchDoc`

**Total pages in sample project:** 13

---

## Adapter Implementation Strategy

### Phase 1: Data Collection
```python
# 1. Get all components with parameters
components_params = get_schematic_components_with_parameters()

# 2. Get all designators
all_designators = get_all_designators()

# 3. Get pin/net data for all components (may need batching)
pins_data = get_component_pins(all_designators)

# 4. Get schematic location data (optional, for spatial queries)
location_data = get_schematic_data(all_designators)
```

### Phase 2: Data Merging
```python
# Merge parameters, location, and pins by designator
for comp in components_params:
    comp['location'] = find_location(comp['designator'], location_data)
    comp['pins'] = find_pins(comp['designator'], pins_data)
```

### Phase 3: Net Building
```python
# Build net list from pin connectivity
nets = {}
for comp in components:
    for pin in comp['pins']:
        if pin['net'] not in nets:
            nets[pin['net']] = Net(name=pin['net'])
        nets[pin['net']].members.append((comp['refdes'], pin['name']))
        nets[pin['net']].pages.add(extract_page_name(comp['sheet']))
```

---

## Sample Component Examples

### IC (U1 - 17 pins)
- **Type:** Complex
- **Has semantic pin names:** No (all numeric)
- **Page:** Power_Switches.SchDoc
- **Pin count:** 17

### Battery Charger (U200 - 48 pins)
- **Type:** Complex
- **Has semantic pin names:** No (all numeric)
- **Page:** battery_charger.SchDoc
- **Pin count:** 48 (TSSOP-48)

### Resistor (R1 - 4 pins)
- **Type:** Complex (4-terminal current sense resistor)
- **Page:** Power_Switches.SchDoc
- **Pin count:** 4

### Capacitor (C1 - 2 pins)
- **Type:** Simple (passive)
- **Page:** Power_Switches.SchDoc
- **Pin count:** 2

### MOSFET (Q1 - 8 pins)
- **Type:** Complex
- **Has semantic pin names:** Yes (S, G, D)
- **Page:** Power_Misc.SchDoc
- **Pin count:** 8 (multi-pin S and D)

### Connector (J1 - 23 pins)
- **Type:** Complex
- **Has semantic pin names:** Yes (numbered 1-19, plus "Shell")
- **Page:** Astro_Connectors2.SchDoc
- **Pin count:** 23 (19 signal + 4 shell)

### LED (LED1 - 4 pins)
- **Type:** Simple/Complex (borderline - RGB LED)
- **Has semantic pin names:** No (numeric)
- **Page:** Astro_Connectors1.SchDoc
- **Pin count:** 4 (R, G, B, Common)

### Button (BTN1 - 4 pins)
- **Type:** Simple
- **Has semantic pin names:** No
- **Page:** Astro_Connectors1.SchDoc
- **Pin count:** 4 (2 pairs for NO switch)

---

## Edge Cases

### No-Connect Pins
LED1 pin 1 has `"net": ""` (empty string) - this is a no-connect.

**Handling:** Include in pin list with net name "NC" or "[no connect]".

### Multi-Part Components
No `U1A`/`U1B` style designators found in this project. All components are unified.

**Decision:** If we encounter multi-part, treat as separate components with `multipart_parent` field.

### Shell/Mechanical Pins
Connector J1 has 4 "Shell" pins all on net "NetJ1_Shell".

**Handling:** Treat like regular pins, show in DSL.

---

## Next Steps

1. ✅ Create `altium_sample.json` with representative examples
2. ⏳ Implement `AltiumJSONAdapter` using this mapping
3. ⏳ Test with real project data
4. ⏳ Handle edge cases as discovered

---

**Sample Data File:** `server/schematic_core/altium_sample.json`
