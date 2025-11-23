# DSL Emitter v0.3 - Implementation Guide

## Overview

The DSL Emitter converts normalized schematic data models into a token-efficient Domain Specific Language (DSL) optimized for Large Language Model consumption. It implements the v0.3 specification with net-centric connectivity and adaptive component formatting.

## Key Principles

### 1. Net-Centric Connectivity
All electrical connections are defined ONLY in the `# NETS` section. Component blocks do NOT include connection information.

### 2. Adaptive Component Formatting
- **Simple components** (2-pin R/C/L): NO `COMP` block, appear inline in nets only
- **Complex components** (ICs, connectors, >4 pins): Full `DEF` block with pin listings

### 3. Inline Pin Hints
Pin references include semantic names when available:
- Simple pin: `R1.1`
- Named pin: `U1.22(PA9_TX)`

### 4. Global Net Summaries
Power/ground nets and nets with many connections are truncated:
- Show first 10 connections
- Append `(+ N others)` for remaining

### 5. Inter-Page Links
Nets spanning multiple pages show their distribution:
- Global: `LINKS: ALL_PAGES`
- Inter-page: `LINKS: Main_Sheet, Connector_Page`

## API Reference

### `emit_page_dsl(components, nets, net_page_map) -> str`

Generates DSL for a single schematic page.

**Parameters:**
- `components: List[Component]` - Components on this page
- `nets: List[Net]` - Nets with pins on this page
- `net_page_map: Dict[str, Set[str]]` - The Atlas (net name → pages)

**Returns:**
- `str` - Formatted DSL text

**Output Format:**
```
# PAGE: <page_name>

# COMPONENTS
<complex component blocks>

# NETS
<net blocks>
```

### `emit_context_dsl(primary_components, neighbor_components, nets) -> str`

Generates DSL for a context bubble (1-hop traversal).

**Parameters:**
- `primary_components: List[Component]` - Explicitly requested components
- `neighbor_components: List[Component]` - Components found in 1-hop
- `nets: List[Net]` - Nets connecting primary and neighbors

**Returns:**
- `str` - Formatted DSL text

**Output Format:**
```
# CONTEXT: <refdes1>, <refdes2>, ...

# COMPONENTS
<complex component blocks for primary>

# CONTEXT_NEIGHBORS
<simplified component summaries>

# NETS
<net blocks>
```

## Component Block Format

Complex components get full DEF blocks:

```
DEF <type> <description>
COMP <refdes> (<value>)
  MPN: <mpn>
  FP: <footprint>
  PINS:
    <pin_designator>: <pin_name>
    ...
```

**Rules:**
- Omit `MPN:` line if empty
- Omit `FP:` line if empty
- If description is empty, use type only
- Sort pins alphabetically by designator (natural sort)

**Example:**
```
DEF IC ARM Cortex-M4 MCU, 168MHz
COMP U1 (STM32F407VGT6)
  MPN: STM32F407VGT6
  FP: LQFP-100
  PINS:
    1: VDD
    2: PA0
    10: GND
    22: PA9_TX
```

## Net Block Format

### Standard Net
```
NET <net_name>
  CON: <comp.pin>, <comp.pin>, ...
```

### Inter-Page Net
```
NET <net_name>
  LINKS: <page1>, <page2>
  CON: <comp.pin>, ...
```

### Global Net (Summarized)
```
NET <net_name>
  LINKS: ALL_PAGES
  CON: <comp.pin>, ... (+ N others)
```

## Pin Reference Format

Pin references follow these rules:

| Pin Type | Format | Example |
|----------|--------|---------|
| Simple passive | `refdes.pin` | `R1.1` |
| Unnamed pin | `refdes.pin` | `U1.5` |
| Named pin | `refdes.pin(name)` | `U1.22(PA9_TX)` |

Simple pin names ("1", "2", "3", "4", "A", "K") are treated as unnamed.

## Sorting Rules

All output is alphabetically sorted:
- **Components**: By refdes
- **Nets**: By net name
- **Pins**: By designator (natural sort for numbers)

## Component Classification

### `is_complex()` Logic
A component is complex if:
- It has more than 4 pins, OR
- Any pin has a semantic name (not "1", "2", "3", "4", "A", "K", "")

**Examples:**
- `R1` (2 pins, no names) → Simple
- `LED1` (2 pins, "A"/"K") → Simple
- `U1` (8 pins) → Complex (by count)
- `Q1` (3 pins, "G"/"D"/"S") → Complex (by pin names)

### `derived_type()` Mapping

| Prefix | Type | Category |
|--------|------|----------|
| R | RES | Passive |
| C | CAP | Passive |
| L, FB | IND | Passive |
| F | FUSE | Passive |
| D, LED | DIODE | Active |
| Q | TRANSISTOR | Active |
| U | IC | Active |
| J, P, CN, CONN | CONN | Active |
| SW, BTN | SWITCH | Active |
| X, Y | OSC | Active |

## Net Classification

### `is_global()` Logic
A net is global if:
- Name matches power/ground pattern (GND, VCC, VDD, 3V3, etc.), OR
- Has more than 15 connections, OR
- Appears on more than 3 pages

**Power Pattern Regex:**
```
^(P?GND|VSS|VCC|VDD|VEE|VBAT)($|_.*)|^(\+?(\d+\.?\d*V\d*|\d*\.?\d*V\d+)|\+?(\d+V))|^.*_(GND|VCC|VDD)$
```

**Matches:**
- GND, PGND, VSS, VCC, VDD, VEE, VBAT
- 3V3, 3.3V, +5V, 12V, 1V8
- NET_GND, SIGNAL_VCC, VCC_DIGITAL

### `is_inter_page()` Logic
A net is inter-page if it appears on more than one page.

## Formatting Conventions

- **Indentation**: 2 spaces per level
- **Line wrapping**: None (LLM can handle long lines)
- **Truncation**: Global nets show first 10 connections, then `(+ N others)`

## Usage Examples

### Basic Page DSL
```python
from schematic_core import emit_page_dsl, Component, Net, Pin

# Create components and nets
components = [...]
nets = [...]
net_page_map = {"GND": {"Page1", "Page2"}, ...}

# Generate DSL
dsl = emit_page_dsl(components, nets, net_page_map)
print(dsl)
```

### Context Bubble DSL
```python
from schematic_core import emit_context_dsl, Component, Net

# Define primary component (e.g., U1)
primary = [u1_component]

# Find neighbors via 1-hop traversal
neighbors = [u2, r1, r2, c1, c2]

# Get connecting nets
context_nets = [net1, net2, net3]

# Generate DSL
dsl = emit_context_dsl(primary, neighbors, context_nets)
print(dsl)
```

## Testing

Run the test suite:
```bash
cd server/schematic_core
python test_dsl_emitter.py
```

Run the example output generator:
```bash
cd server/schematic_core
python example_output.py
```

## Implementation Notes

### Helper Functions

#### `_format_component_block(component)`
Formats a complex component as a DEF block with all metadata and pins.

#### `_format_net_block(net, net_pages, components)`
Formats a net block with LINKS and CON lines, applying truncation for global nets.

#### `_format_pin_reference(refdes, pin_designator, components)`
Formats a pin reference with optional semantic name in parentheses.

#### `_format_neighbor_summary(component)`
Creates one-line summary for context neighbors.

#### `_natural_sort_key(text)`
Generates sort key for natural number sorting (1, 2, 10 not 1, 10, 2).

### Design Decisions

1. **Why no connections in COMP blocks?**
   - Reduces token count significantly
   - Avoids duplication (same info in nets)
   - LLMs can infer connectivity from nets

2. **Why truncate global nets?**
   - GND and power nets can have 100+ connections
   - Full listing wastes tokens with little value
   - LLM understands "(+ N others)" pattern

3. **Why inline simple passives?**
   - 2-pin resistors/caps are self-evident
   - No semantic value in separate blocks
   - Dramatically reduces output size

4. **Why named pin hints?**
   - Semantic names provide critical context
   - Format `U1.22(PA9_TX)` is human and LLM readable
   - Helps LLM reason about circuit function

## Token Efficiency

Example token savings compared to verbose format:

| Component | Verbose Format | DSL v0.3 | Savings |
|-----------|---------------|----------|---------|
| 100 resistors | ~5000 tokens | ~500 tokens | 90% |
| GND net (200 pins) | ~2000 tokens | ~150 tokens | 92% |
| Complex IC | ~800 tokens | ~600 tokens | 25% |

**Overall:** 60-80% token reduction for typical schematics.

## Integration with Librarian

The Librarian module calls DSL Emitter functions:

```python
class Librarian:
    def get_page(self, page_name: str) -> str:
        components = [c for c in self.components if c.page == page_name]
        nets = self._filter_nets_for_page(page_name)
        return emit_page_dsl(components, nets, self.net_page_map)

    def get_context(self, refdes_list: List[str]) -> str:
        primary = [c for c in self.components if c.refdes in refdes_list]
        neighbors, context_nets = self._traverse_one_hop(primary)
        return emit_context_dsl(primary, neighbors, context_nets)
```

## Future Enhancements

Potential improvements for future versions:

1. **Multiline descriptions**: Handle very long component descriptions
2. **Custom truncation thresholds**: Configurable limits for global nets
3. **Spatial hints**: Optional location data for layout reasoning
4. **Hierarchical blocks**: Support for hierarchical schematics
5. **Bus notation**: Compressed format for data buses

## References

- SCHEMATIC_CORE_SPEC.md - Full specification
- Appendix A - Example output format
- Section 3 - Data model details
- Section 6 - DSL Emitter specification
