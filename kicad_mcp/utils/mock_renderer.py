"""
Mock schematic renderer for development and testing when KiCad CLI is not available.
Creates simple SVG representations of schematics for visualization testing.
"""
import os
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path


class MockSchematicRenderer:
    """Simple SVG renderer for KiCad schematics when CLI is not available."""
    
    def __init__(self):
        """Initialize the mock renderer."""
        self.canvas_width = 800
        self.canvas_height = 600
        self.grid_size = 20
        self.component_width = 60
        self.component_height = 40
    
    def parse_schematic_components(self, schematic_file: str) -> Dict[str, Any]:
        """Parse components from a KiCad schematic file.
        
        Args:
            schematic_file: Path to .kicad_sch file
            
        Returns:
            Dictionary with parsed components and metadata
        """
        if not os.path.exists(schematic_file):
            return {"components": [], "power_symbols": [], "wires": [], "title": "Unknown"}
        
        with open(schematic_file, 'r') as f:
            content = f.read()
        
        # Extract title
        title_match = re.search(r'\(title\s+"([^"]+)"\)', content)
        title = title_match.group(1) if title_match else "Schematic"
        
        # Parse components (symbols)
        components = []
        power_symbols = []
        wires = []
        
        # Find all symbol instances
        symbol_pattern = r'\(symbol\s+\(lib_id\s+"([^"]+)"\)\s+\(at\s+([\d.-]+)\s+([\d.-]+)\s+[\d.-]+\)\s+[^)]*?\(property\s+"Reference"\s+"([^"]+)"[^)]*?\)\s+[^)]*?\(property\s+"Value"\s+"([^"]+)"'
        
        for match in re.finditer(symbol_pattern, content, re.DOTALL):
            lib_id, x, y, reference, value = match.groups()
            
            component = {
                "lib_id": lib_id,
                "reference": reference,
                "value": value,
                "position": (float(x), float(y)),
                "type": self._get_component_type(lib_id, reference)
            }
            
            if reference.startswith("#PWR"):
                power_symbols.append(component)
            else:
                components.append(component)
        
        # Parse wires (simplified)
        wire_pattern = r'\(wire\s+\(pts\s+\(xy\s+([\d.-]+)\s+([\d.-]+)\)\s+\(xy\s+([\d.-]+)\s+([\d.-]+)\)\)'
        
        for match in re.finditer(wire_pattern, content):
            x1, y1, x2, y2 = match.groups()
            wires.append({
                "start": (float(x1), float(y1)),
                "end": (float(x2), float(y2))
            })
        
        return {
            "title": title,
            "components": components,
            "power_symbols": power_symbols,
            "wires": wires
        }
    
    def _get_component_type(self, lib_id: str, reference: str) -> str:
        """Determine component type from library ID and reference."""
        lib_id_lower = lib_id.lower()
        ref_lower = reference.lower()
        
        if "resistor" in lib_id_lower or reference.startswith("R"):
            return "resistor"
        elif "capacitor" in lib_id_lower or reference.startswith("C"):
            return "capacitor"
        elif "led" in lib_id_lower or "diode" in lib_id_lower:
            return "led"
        elif "inductor" in lib_id_lower or reference.startswith("L"):
            return "inductor"
        elif reference.startswith("U") or reference.startswith("IC"):
            return "ic"
        elif reference.startswith("SW"):
            return "switch"
        else:
            return "generic"
    
    def render_to_svg(self, schematic_data: Dict[str, Any]) -> str:
        """Render schematic data to SVG format.
        
        Args:
            schematic_data: Parsed schematic data
            
        Returns:
            SVG content as string
        """
        # Calculate bounds
        all_positions = []
        for comp in schematic_data["components"] + schematic_data["power_symbols"]:
            all_positions.append(comp["position"])
        
        if all_positions:
            min_x = min(pos[0] for pos in all_positions) - 50
            max_x = max(pos[0] for pos in all_positions) + 50
            min_y = min(pos[1] for pos in all_positions) - 50
            max_y = max(pos[1] for pos in all_positions) + 50
        else:
            min_x, max_x, min_y, max_y = 0, 400, 0, 300
        
        width = max(400, max_x - min_x)
        height = max(300, max_y - min_y)
        
        # Start SVG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .component {{ fill: lightblue; stroke: black; stroke-width: 2; }}
      .power {{ fill: lightgreen; stroke: darkgreen; stroke-width: 2; }}
      .wire {{ stroke: red; stroke-width: 2; fill: none; }}
      .text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
      .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
    </style>
  </defs>
  
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="white" stroke="gray" stroke-width="1"/>
  
  <!-- Grid -->
'''
        
        # Add grid
        for x in range(0, int(width), self.grid_size):
            svg_content += f'  <line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="lightgray" stroke-width="0.5"/>\n'
        for y in range(0, int(height), self.grid_size):
            svg_content += f'  <line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="lightgray" stroke-width="0.5"/>\n'
        
        # Add title
        svg_content += f'  <text x="{width/2}" y="20" class="title">{schematic_data["title"]}</text>\n\n'
        
        # Transform coordinates (flip Y and offset)
        def transform_coords(x, y):
            return (x - min_x, height - (y - min_y) - 50)
        
        # Render wires first (so they appear behind components)
        svg_content += "  <!-- Wires -->\n"
        for wire in schematic_data["wires"]:
            x1, y1 = transform_coords(*wire["start"])
            x2, y2 = transform_coords(*wire["end"])
            svg_content += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="wire"/>\n'
        
        svg_content += "\n  <!-- Components -->\n"
        
        # Render components
        for comp in schematic_data["components"]:
            x, y = transform_coords(*comp["position"])
            self._render_component(svg_content, comp, x, y)
        
        # Render power symbols
        svg_content += "\n  <!-- Power Symbols -->\n"
        for power in schematic_data["power_symbols"]:
            x, y = transform_coords(*power["position"])
            self._render_power_symbol(svg_content, power, x, y)
        
        svg_content += "\n</svg>"
        return svg_content
    
    def _render_component(self, svg_content: str, component: Dict[str, Any], x: float, y: float) -> None:
        """Add component to SVG content."""
        comp_type = component["type"]
        reference = component["reference"]
        value = component["value"]
        
        if comp_type == "resistor":
            # Draw resistor symbol (zigzag)
            svg_content += f'  <g>\n'
            svg_content += f'    <rect x="{x-30}" y="{y-10}" width="60" height="20" class="component"/>\n'
            svg_content += f'    <text x="{x}" y="{y-15}" class="text">{reference}</text>\n'
            svg_content += f'    <text x="{x}" y="{y+25}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
        
        elif comp_type == "capacitor":
            # Draw capacitor symbol (two parallel lines)
            svg_content += f'  <g>\n'
            svg_content += f'    <rect x="{x-25}" y="{y-15}" width="50" height="30" class="component"/>\n'
            svg_content += f'    <line x1="{x-5}" y1="{y-10}" x2="{x-5}" y2="{y+10}" stroke="black" stroke-width="2"/>\n'
            svg_content += f'    <line x1="{x+5}" y1="{y-10}" x2="{x+5}" y2="{y+10}" stroke="black" stroke-width="2"/>\n'
            svg_content += f'    <text x="{x}" y="{y-20}" class="text">{reference}</text>\n'
            svg_content += f'    <text x="{x}" y="{y+25}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
        
        elif comp_type == "led":
            # Draw LED symbol (diode with arrows)
            svg_content += f'  <g>\n'
            svg_content += f'    <circle cx="{x}" cy="{y}" r="15" class="component"/>\n'
            svg_content += f'    <text x="{x}" y="{y-20}" class="text">{reference}</text>\n'
            svg_content += f'    <text x="{x}" y="{y+25}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
        
        elif comp_type == "ic":
            # Draw IC symbol (rectangle)
            svg_content += f'  <g>\n'
            svg_content += f'    <rect x="{x-40}" y="{y-20}" width="80" height="40" class="component"/>\n'
            svg_content += f'    <text x="{x}" y="{y-25}" class="text">{reference}</text>\n'
            svg_content += f'    <text x="{x}" y="{y+5}" class="text" font-size="10">{value}</text>\n'
            svg_content += f'  </g>\n'
        
        else:
            # Generic component
            svg_content += f'  <g>\n'
            svg_content += f'    <rect x="{x-20}" y="{y-10}" width="40" height="20" class="component"/>\n'
            svg_content += f'    <text x="{x}" y="{y-15}" class="text">{reference}</text>\n'
            svg_content += f'    <text x="{x}" y="{y+25}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
    
    def _render_power_symbol(self, svg_content: str, power: Dict[str, Any], x: float, y: float) -> None:
        """Add power symbol to SVG content."""
        value = power["value"]
        reference = power["reference"]
        
        if "VCC" in value or "VDD" in value or "+" in value:
            # Power supply symbol (up arrow)
            svg_content += f'  <g>\n'
            svg_content += f'    <polygon points="{x},{y-15} {x-10},{y+5} {x+10},{y+5}" class="power"/>\n'
            svg_content += f'    <text x="{x}" y="{y+20}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
        
        elif "GND" in value or "GROUND" in value:
            # Ground symbol (horizontal lines)
            svg_content += f'  <g>\n'
            svg_content += f'    <line x1="{x-15}" y1="{y}" x2="{x+15}" y2="{y}" stroke="darkgreen" stroke-width="3"/>\n'
            svg_content += f'    <line x1="{x-10}" y1="{y+5}" x2="{x+10}" y2="{y+5}" stroke="darkgreen" stroke-width="2"/>\n'
            svg_content += f'    <line x1="{x-5}" y1="{y+10}" x2="{x+5}" y2="{y+10}" stroke="darkgreen" stroke-width="1"/>\n'
            svg_content += f'    <text x="{x}" y="{y+25}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
        
        else:
            # Generic power symbol
            svg_content += f'  <g>\n'
            svg_content += f'    <circle cx="{x}" cy="{y}" r="8" class="power"/>\n'
            svg_content += f'    <text x="{x}" y="{y+20}" class="text">{value}</text>\n'
            svg_content += f'  </g>\n'
    
    def render_schematic_file(self, schematic_file: str, output_svg: str) -> bool:
        """Render a schematic file to SVG.
        
        Args:
            schematic_file: Path to .kicad_sch file
            output_svg: Path to output SVG file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse schematic
            schematic_data = self.parse_schematic_components(schematic_file)
            
            # Render to SVG
            svg_content = self.render_to_svg(schematic_data)
            
            # Write SVG file
            os.makedirs(os.path.dirname(output_svg), exist_ok=True)
            with open(output_svg, 'w') as f:
                f.write(svg_content)
            
            return True
            
        except Exception as e:
            print(f"Error rendering schematic: {e}")
            return False


def create_mock_schematic_screenshot(schematic_file: str, output_dir: str) -> str:
    """Create a mock screenshot of a schematic using the mock renderer.
    
    Args:
        schematic_file: Path to .kicad_sch file
        output_dir: Directory to save the screenshot
        
    Returns:
        Path to generated PNG file
    """
    renderer = MockSchematicRenderer()
    
    # Generate SVG
    schematic_name = os.path.splitext(os.path.basename(schematic_file))[0]
    svg_file = os.path.join(output_dir, f"{schematic_name}_mock.svg")
    
    success = renderer.render_schematic_file(schematic_file, svg_file)
    if not success:
        return None
    
    # Try to convert to PNG if cairosvg is available
    try:
        import cairosvg
        png_file = svg_file.replace('.svg', '.png')
        
        cairosvg.svg2png(
            url=svg_file,
            write_to=png_file,
            output_width=800,
            output_height=600,
            background_color='white'
        )
        
        return png_file if os.path.exists(png_file) else svg_file
        
    except ImportError:
        # Return SVG if PNG conversion not available
        return svg_file


if __name__ == "__main__":
    # Test the mock renderer
    schematic_file = "tests/visual_output/test_circuit.kicad_sch"
    output_dir = "tests/visual_output"
    
    if os.path.exists(schematic_file):
        print(f"🔧 Testing mock renderer with: {schematic_file}")
        
        screenshot = create_mock_schematic_screenshot(schematic_file, output_dir)
        if screenshot:
            print(f"✅ Mock screenshot created: {screenshot}")
        else:
            print("❌ Mock screenshot failed")
    else:
        print(f"⚠️  Test schematic not found: {schematic_file}")
        print("Run test_visualization_standalone.py first to create it")
