# Footprint Tools Test Plan

## 1. List Libraries
`mcp__kicad__list_footprint_libraries`
- Returns 150+ libraries, sorted, without `.pretty` extension

## 2. Rebuild Index
`mcp__kicad__rebuild_footprint_index`
- Creates cache at `~/.cache/kicad-mcp/footprint_index.json`
- Returns 15000+ footprints, 150+ libraries
- Progress 0-100%, takes 30-90s

## 3. Search
`mcp__kicad__search_footprints`
- Query `"0805"` → resistors and capacitors
- Query `"SOIC-8"` → Package_SO results
- With library filter → only that library
- With limit=5 → truncated results
- Nonexistent query → 0 matches

## 4. Validate
`mcp__kicad__validate_footprint_exists`
- `Resistor_SMD:R_0805_2012Metric` → exists=true with description
- `InvalidFormat` → exists=false, format error
- `NonExistent:R_0805` → exists=false, suggestions
- `Resistor_SMD:R_9999` → exists=false, similar footprints

## 5. Get Pads
`mcp__kicad__get_footprint_pads`
- `Resistor_SMD:R_0805_2012Metric` → 2 SMD pads, no drill
- `Package_DIP:DIP-8_W7.62mm` → 8 thru_hole pads with drill
- Invalid footprint → error with suggestions

## Edge Cases
- Search without cache → "missing" status, rebuild prompt
- Search with stale cache → "stale" status, rebuild prompt
- Rebuild over existing cache → updates timestamp
