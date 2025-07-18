"""
Circuit creation tools for KiCad projects.
"""

import json
import os
import re
import shutil
from typing import Any
import uuid

from fastmcp import Context, FastMCP

from kicad_mcp.config import KICAD_APP_PATH, system
from kicad_mcp.utils.boundary_validator import BoundaryValidator
from kicad_mcp.utils.component_layout import ComponentLayoutManager
from kicad_mcp.utils.file_utils import get_project_files
from kicad_mcp.utils.sexpr_generator import SExpressionGenerator


def _get_component_type_from_symbol(symbol_library: str, symbol_name: str) -> str:
    """Determine component type from symbol library and name."""
    library = symbol_library.lower()
    name = symbol_name.lower()

    # Map symbol names to component types
    if name in ["r", "resistor"]:
        return "resistor"
    elif name in ["c", "capacitor"]:
        return "capacitor"
    elif name in ["l", "inductor"]:
        return "inductor"
    elif name in ["led"]:
        return "led"
    elif name in ["d", "diode"]:
        return "diode"
    elif "transistor" in name or "npn" in name or "pnp" in name:
        return "transistor"
    elif library == "switch":
        return "switch"
    elif library == "connector":
        return "connector"
    elif (
        "ic" in name
        or "mcu" in name
        or "esp32" in name
        or library in ["mcu", "microcontroller", "mcu_espressif"]
    ):
        return "ic"
    else:
        return "default"


async def create_new_project(
    project_name: str, project_path: str, description: str = "", ctx: Context = None
) -> dict[str, Any]:
    """Create a new KiCad project with basic files.

    Args:
        project_name: Name of the new project
        project_path: Directory path where the project will be created
        description: Optional project description
        ctx: Context for MCP communication

    Returns:
        Dictionary with project creation status and file paths
    """
    try:
        if ctx:
            await ctx.info(f"Creating new KiCad project: {project_name}")
            await ctx.report_progress(10, 100)

        # Ensure project directory exists
        os.makedirs(project_path, exist_ok=True)

        # Define file paths
        project_file = os.path.join(project_path, f"{project_name}.kicad_pro")
        schematic_file = os.path.join(project_path, f"{project_name}.kicad_sch")
        pcb_file = os.path.join(project_path, f"{project_name}.kicad_pcb")

        if ctx:
            await ctx.report_progress(30, 100)

        # Check if project already exists
        if os.path.exists(project_file):
            return {"success": False, "error": f"Project already exists at {project_file}"}

        # Create basic project file
        project_data = {
            "board": {
                "3dviewports": [],
                "design_settings": {
                    "defaults": {"board_outline_line_width": 0.1, "copper_line_width": 0.2}
                },
                "layer_presets": [],
                "viewports": [],
            },
            "libraries": {"pinned_footprint_libs": [], "pinned_symbol_libs": []},
            "meta": {"filename": f"{project_name}.kicad_pro", "version": 1},
            "net_settings": {
                "classes": [
                    {
                        "clearance": 0.2,
                        "diff_pair_gap": 0.25,
                        "diff_pair_via_gap": 0.25,
                        "diff_pair_width": 0.2,
                        "line_style": 0,
                        "microvia_diameter": 0.3,
                        "microvia_drill": 0.1,
                        "name": "Default",
                        "pcb_color": "rgba(0, 0, 0, 0.000)",
                        "schematic_color": "rgba(0, 0, 0, 0.000)",
                        "track_width": 0.25,
                        "via_diameter": 0.8,
                        "via_drill": 0.4,
                        "wire_width": 6,
                    }
                ],
                "meta": {"version": 3},
            },
            "pcbnew": {
                "last_paths": {
                    "gencad": "",
                    "idf": "",
                    "netlist": "",
                    "specctra_dsn": "",
                    "step": "",
                    "vrml": "",
                },
                "page_layout_descr_file": "",
            },
            "schematic": {
                "annotate_start_num": 0,
                "drawing": {
                    "dashed_lines_dash_length_ratio": 12.0,
                    "dashed_lines_gap_length_ratio": 3.0,
                    "default_line_thickness": 6.0,
                    "default_text_size": 50.0,
                    "field_names": [],
                    "intersheets_ref_own_page": False,
                    "intersheets_ref_prefix": "",
                    "intersheets_ref_short": False,
                    "intersheets_ref_show": False,
                    "intersheets_ref_suffix": "",
                    "junction_size_choice": 3,
                    "label_size_ratio": 0.375,
                    "pin_symbol_size": 25.0,
                    "text_offset_ratio": 0.15,
                },
                "legacy_lib_dir": "",
                "legacy_lib_list": [],
                "meta": {"version": 1},
                "net_format_name": "",
                "page_layout_descr_file": "",
                "plot_directory": "",
                "spice_current_sheet_as_root": False,
                "spice_external_command": 'spice "%I"',
                "spice_model_current_sheet_as_root": True,
                "spice_save_all_currents": False,
                "spice_save_all_voltages": False,
                "subpart_first_id": 65,
                "subpart_id_separator": 0,
            },
            "sheets": [["e63e39d7-6ac0-4ffd-8aa3-1841a4541b55", ""]],
            "text_variables": {},
        }

        if description:
            project_data["meta"]["description"] = description

        with open(project_file, "w") as f:
            json.dump(project_data, f, indent=2)

        if ctx:
            await ctx.report_progress(60, 100)

        # Create basic schematic file using S-expression format
        generator = SExpressionGenerator()
        schematic_content = generator.generate_schematic(
            circuit_name=project_name, components=[], power_symbols=[], connections=[]
        )

        with open(schematic_file, "w") as f:
            f.write(schematic_content)

        if ctx:
            await ctx.report_progress(80, 100)

        # Create basic PCB file in S-expression format
        pcb_content = f"""(kicad_pcb
  (version 20240618)
  (generator "kicad-mcp")
  (general
    (thickness 1.6)
  )
  (paper "A4")
  (title_block
    (title "{project_name}")
    (date "")
    (rev "")
    (company "")
    (comment (number 1) (value "{description if description else ""}"))
  )
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )
  (setup
    (pad_to_mask_clearance 0)
    (pcbplotparams
      (layerselection 0x00010fc_ffffffff)
      (plot_on_all_layers_selection 0x0000000_00000000)
      (disableapertmacros false)
      (usegerberextensions false)
      (usegerberattributes true)
      (usegerberadvancedattributes true)
      (creategerberjobfile true)
      (dashed_line_dash_ratio 12.000000)
      (dashed_line_gap_ratio 3.000000)
      (svgprecision 4)
      (plotframeref false)
      (viasonmask false)
      (mode 1)
      (useauxorigin false)
      (hpglpennumber 1)
      (hpglpenspeed 20)
      (hpglpendiameter 15.000000)
      (pdf_front_fp_property_popups true)
      (pdf_back_fp_property_popups true)
      (dxfpolygonmode true)
      (dxfimperialunits true)
      (dxfusepcbnewfont true)
      (psnegative false)
      (psa4output false)
      (plotreference true)
      (plotvalue true)
      (plotfptext true)
      (plotinvisibletext false)
      (sketchpadsonfab false)
      (subtractmaskfromsilk false)
      (outputformat 1)
      (mirror false)
      (drillshape 1)
      (scaleselection 1)
      (outputdirectory "")
    )
  )
  (nets
    (net 0 "")
  )
)"""

        with open(pcb_file, "w") as f:
            f.write(pcb_content)

        if ctx:
            await ctx.report_progress(90, 100)
            await ctx.info("Generating visual feedback...")

            # Generate visual feedback for the created schematic
            try:
                from kicad_mcp.tools.visualization_tools import capture_schematic_screenshot

                screenshot_result = await capture_schematic_screenshot(project_path, ctx)
                if screenshot_result:
                    await ctx.info("✓ Schematic screenshot captured successfully")
                else:
                    await ctx.info(
                        "⚠ Screenshot capture failed - proceeding without visual feedback"
                    )
            except ImportError:
                await ctx.info(
                    "⚠ Visualization tools not available - proceeding without visual feedback"
                )
            except Exception as e:
                await ctx.info(f"⚠ Screenshot capture failed: {str(e)}")

            await ctx.report_progress(100, 100)
            await ctx.info(f"Successfully created project at {project_file}")

        return {
            "success": True,
            "project_file": project_file,
            "schematic_file": schematic_file,
            "pcb_file": pcb_file,
            "project_path": project_path,
            "project_name": project_name,
        }

    except Exception as e:
        error_msg = f"Error creating project '{project_name}' at '{project_path}': {str(e)}"
        if ctx:
            await ctx.info(error_msg)
            await ctx.info(f"Exception type: {type(e).__name__}")
        return {"success": False, "error": error_msg, "error_type": type(e).__name__}


async def add_component(
    project_path: str,
    component_reference: str,
    component_value: str,
    symbol_library: str,
    symbol_name: str,
    x_position: float,
    y_position: float,
    ctx: Context = None,
) -> dict[str, Any]:
    """Add a component to a KiCad schematic.

    WARNING: This tool modifies existing schematic files but may not preserve
    S-expression format if the file was created with proper KiCad tools.
    Prefer using create_kicad_schematic_from_text for new schematics.

    Args:
        project_path: Path to the KiCad project file (.kicad_pro)
        component_reference: Component reference designator (e.g., "R1", "C1")
        component_value: Component value (e.g., "10k", "100nF")
        symbol_library: Name of the symbol library
        symbol_name: Name of the symbol in the library
        x_position: X coordinate for component placement (in mm)
        y_position: Y coordinate for component placement (in mm)
        ctx: Context for MCP communication

    Returns:
        Dictionary with component addition status
    """
    try:
        if ctx:
            await ctx.info(f"Adding component {component_reference} to schematic")
            await ctx.report_progress(10, 100)

        # Get project files
        files = get_project_files(project_path)
        if "schematic" not in files:
            return {"success": False, "error": "No schematic file found in project"}

        schematic_file = files["schematic"]

        if ctx:
            await ctx.report_progress(30, 100)

        # Read existing schematic and check if it can be modified
        read_result = _read_schematic_for_modification(schematic_file)
        if not read_result["success"]:
            return read_result

        schematic_data = read_result["data"]

        # Create component UUID
        component_uuid = str(uuid.uuid4())

        # Validate and fix component position using boundary validator
        validator = BoundaryValidator()
        layout_manager = ComponentLayoutManager()

        # Determine component type from symbol information
        component_type = _get_component_type_from_symbol(symbol_library, symbol_name)

        # Validate position using boundary validator
        validation_issue = validator.validate_component_position(
            component_reference, x_position, y_position, component_type
        )

        if ctx:
            await ctx.info(f"Position validation: {validation_issue.message}")

        # Handle validation result
        if validation_issue.suggested_position:
            # Use suggested corrected position
            final_x, final_y = validation_issue.suggested_position
            if ctx:
                await ctx.info(
                    f"Component position corrected: ({x_position}, {y_position}) → ({final_x}, {final_y})"
                )
        else:
            # Position is valid, snap to grid
            final_x, final_y = layout_manager.snap_to_grid(x_position, y_position)
            if ctx:
                await ctx.info(
                    f"Component position validated and snapped to grid: ({final_x}, {final_y})"
                )

        # Convert positions to KiCad internal units (0.1mm)
        x_pos_internal = int(final_x * 10)
        y_pos_internal = int(final_y * 10)

        if ctx:
            await ctx.report_progress(50, 100)

        # Create symbol entry
        symbol_entry = {
            "lib_id": f"{symbol_library}:{symbol_name}",
            "at": [x_pos_internal, y_pos_internal, 0],
            "uuid": component_uuid,
            "property": [
                {
                    "name": "Reference",
                    "value": component_reference,
                    "at": [x_pos_internal, y_pos_internal - 254, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}},
                },
                {
                    "name": "Value",
                    "value": component_value,
                    "at": [x_pos_internal, y_pos_internal + 254, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}},
                },
                {
                    "name": "Footprint",
                    "value": "",
                    "at": [x_pos_internal, y_pos_internal, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}, "hide": True},
                },
                {
                    "name": "Datasheet",
                    "value": "",
                    "at": [x_pos_internal, y_pos_internal, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}, "hide": True},
                },
            ],
            "pin": [],
        }

        # Add symbol to schematic
        if "symbol" not in schematic_data:
            schematic_data["symbol"] = []

        schematic_data["symbol"].append(symbol_entry)

        if ctx:
            await ctx.report_progress(80, 100)
            await ctx.info(
                f"Writing component to schematic. Total components now: {len(schematic_data.get('symbol', []))}"
            )

        # Write updated schematic
        with open(schematic_file, "w") as f:
            json.dump(schematic_data, f, indent=2)

        if ctx:
            await ctx.report_progress(90, 100)
            await ctx.info("Generating visual feedback for updated schematic...")

            # Generate visual feedback after adding component
            try:
                from kicad_mcp.tools.visualization_tools import capture_schematic_screenshot

                screenshot_result = await capture_schematic_screenshot(project_path, ctx)
                if screenshot_result:
                    await ctx.info("✓ Updated schematic screenshot captured")
                else:
                    await ctx.info(
                        "⚠ Screenshot capture failed - proceeding without visual feedback"
                    )
            except ImportError:
                await ctx.info("⚠ Visualization tools not available")
            except Exception as e:
                await ctx.info(f"⚠ Screenshot capture failed: {str(e)}")

            await ctx.report_progress(100, 100)
            await ctx.info(f"Successfully added component {component_reference}")

        return {
            "success": True,
            "component_reference": component_reference,
            "component_uuid": component_uuid,
            "position": [x_position, y_position],
            "debug_info": {
                "total_components": len(schematic_data.get("symbol", [])),
                "schematic_file": schematic_file,
                "symbol_entry_keys": list(symbol_entry.keys()),
            },
        }

    except Exception as e:
        error_msg = f"Error adding component '{component_reference}' ({symbol_library}:{symbol_name}) to '{project_path}': {str(e)}"
        if ctx:
            await ctx.info(error_msg)
            await ctx.info(f"Exception type: {type(e).__name__}")
            await ctx.info(f"Position: ({x_position}, {y_position})")
        return {
            "success": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "component_reference": component_reference,
            "position": [x_position, y_position],
        }


async def create_wire_connection(
    project_path: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a wire connection between two points in a schematic.

    Args:
        project_path: Path to the KiCad project file (.kicad_pro)
        start_x: Starting X coordinate (in mm)
        start_y: Starting Y coordinate (in mm)
        end_x: Ending X coordinate (in mm)
        end_y: Ending Y coordinate (in mm)
        ctx: Context for MCP communication

    Returns:
        Dictionary with wire creation status
    """
    try:
        if ctx:
            await ctx.info("Creating wire connection")
            await ctx.report_progress(10, 100)

        # Get project files
        files = get_project_files(project_path)
        if "schematic" not in files:
            return {"success": False, "error": "No schematic file found in project"}

        schematic_file = files["schematic"]

        if ctx:
            await ctx.report_progress(30, 100)

        # Read existing schematic and check if it can be modified
        read_result = _read_schematic_for_modification(schematic_file)
        if not read_result["success"]:
            return read_result

        schematic_data = read_result["data"]

        # Validate wire positions using boundary validator
        validator = BoundaryValidator()
        layout_manager = ComponentLayoutManager()

        # Validate wire connection endpoints
        wire_issues = validator.validate_wire_connection(start_x, start_y, end_x, end_y)

        if ctx and wire_issues:
            for issue in wire_issues:
                await ctx.info(f"Wire validation: {issue.message}")

        # Apply corrections if needed
        if wire_issues:
            # Correct start position if needed
            start_issue = next(
                (issue for issue in wire_issues if issue.component_ref == "WIRE_START"), None
            )
            if start_issue:
                start_x, start_y = layout_manager.snap_to_grid(
                    max(layout_manager.bounds.min_x, min(start_x, layout_manager.bounds.max_x)),
                    max(layout_manager.bounds.min_y, min(start_y, layout_manager.bounds.max_y)),
                )
                if ctx:
                    await ctx.info(f"Wire start position corrected to ({start_x}, {start_y})")

            # Correct end position if needed
            end_issue = next(
                (issue for issue in wire_issues if issue.component_ref == "WIRE_END"), None
            )
            if end_issue:
                end_x, end_y = layout_manager.snap_to_grid(
                    max(layout_manager.bounds.min_x, min(end_x, layout_manager.bounds.max_x)),
                    max(layout_manager.bounds.min_y, min(end_y, layout_manager.bounds.max_y)),
                )
                if ctx:
                    await ctx.info(f"Wire end position corrected to ({end_x}, {end_y})")
        else:
            # Positions are valid, just snap to grid
            start_x, start_y = layout_manager.snap_to_grid(start_x, start_y)
            end_x, end_y = layout_manager.snap_to_grid(end_x, end_y)

        # Convert positions to KiCad internal units
        start_x_internal = int(start_x * 10)
        start_y_internal = int(start_y * 10)
        end_x_internal = int(end_x * 10)
        end_y_internal = int(end_y * 10)

        if ctx:
            await ctx.report_progress(50, 100)

        # Create wire entry
        wire_entry = {
            "pts": [[start_x_internal, start_y_internal], [end_x_internal, end_y_internal]],
            "stroke": {"width": 0, "type": "default"},
            "uuid": str(uuid.uuid4()),
        }

        # Add wire to schematic
        if "wire" not in schematic_data:
            schematic_data["wire"] = []

        schematic_data["wire"].append(wire_entry)

        if ctx:
            await ctx.report_progress(80, 100)

        # Write updated schematic
        with open(schematic_file, "w") as f:
            json.dump(schematic_data, f, indent=2)

        if ctx:
            await ctx.report_progress(90, 100)
            await ctx.info("Generating visual feedback for wire connection...")

            # Generate visual feedback after adding wire
            try:
                from kicad_mcp.tools.visualization_tools import capture_schematic_screenshot

                screenshot_result = await capture_schematic_screenshot(project_path, ctx)
                if screenshot_result:
                    await ctx.info("✓ Wire connection screenshot captured")
                else:
                    await ctx.info(
                        "⚠ Screenshot capture failed - proceeding without visual feedback"
                    )
            except ImportError:
                await ctx.info("⚠ Visualization tools not available")
            except Exception as e:
                await ctx.info(f"⚠ Screenshot capture failed: {str(e)}")

            await ctx.report_progress(100, 100)
            await ctx.info("Successfully created wire connection")

        return {
            "success": True,
            "start_position": [start_x, start_y],
            "end_position": [end_x, end_y],
            "wire_uuid": wire_entry["uuid"],
        }

    except Exception as e:
        if ctx:
            await ctx.info(f"Error creating wire: {str(e)}")
        return {"success": False, "error": str(e)}


async def add_power_symbol(
    project_path: str, power_type: str, x_position: float, y_position: float, ctx: Context = None
) -> dict[str, Any]:
    """Add a power symbol (VCC, GND, etc.) to the schematic.

    Args:
        project_path: Path to the KiCad project file (.kicad_pro)
        power_type: Type of power symbol ("VCC", "GND", "+5V", "+3V3", etc.)
        x_position: X coordinate for symbol placement (in mm)
        y_position: Y coordinate for symbol placement (in mm)
        ctx: Context for MCP communication

    Returns:
        Dictionary with power symbol addition status
    """
    try:
        if ctx:
            await ctx.info(f"Adding {power_type} power symbol")
            await ctx.report_progress(10, 100)

        # Map power types to KiCad symbols
        power_symbols = {
            "VCC": "power:VCC",
            "GND": "power:GND",
            "+5V": "power:+5V",
            "+3V3": "power:+3V3",
            "+12V": "power:+12V",
            "-12V": "power:-12V",
        }

        if power_type not in power_symbols:
            return {
                "success": False,
                "error": f"Unknown power type: {power_type}. Available types: {list(power_symbols.keys())}",
            }

        symbol_lib_id = power_symbols[power_type]

        # Manually create power symbol component
        # Get project files
        files = get_project_files(project_path)
        if "schematic" not in files:
            return {"success": False, "error": "No schematic file found in project"}

        schematic_file = files["schematic"]

        if ctx:
            await ctx.report_progress(30, 100)

        # Read existing schematic and check if it can be modified
        read_result = _read_schematic_for_modification(schematic_file)
        if not read_result["success"]:
            return read_result

        schematic_data = read_result["data"]

        # Create component UUID
        component_uuid = str(uuid.uuid4())

        # Convert positions to KiCad internal units (0.1mm)
        x_pos_internal = int(x_position * 10)
        y_pos_internal = int(y_position * 10)

        # Generate power reference
        power_ref = f"#PWR0{len([s for s in schematic_data.get('symbol', []) if s.get('lib_id', '').startswith('power:')]) + 1:03d}"

        if ctx:
            await ctx.report_progress(50, 100)

        # Create symbol entry
        symbol_entry = {
            "lib_id": symbol_lib_id,
            "at": [x_pos_internal, y_pos_internal, 0],
            "uuid": component_uuid,
            "property": [
                {
                    "name": "Reference",
                    "value": power_ref,
                    "at": [x_pos_internal, y_pos_internal - 254, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}},
                },
                {
                    "name": "Value",
                    "value": power_type,
                    "at": [x_pos_internal, y_pos_internal + 254, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}},
                },
                {
                    "name": "Footprint",
                    "value": "",
                    "at": [x_pos_internal, y_pos_internal, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}, "hide": True},
                },
                {
                    "name": "Datasheet",
                    "value": "",
                    "at": [x_pos_internal, y_pos_internal, 0],
                    "effects": {"font": {"size": [1.27, 1.27]}, "hide": True},
                },
            ],
            "pin": [],
        }

        # Add symbol to schematic
        if "symbol" not in schematic_data:
            schematic_data["symbol"] = []

        schematic_data["symbol"].append(symbol_entry)

        if ctx:
            await ctx.report_progress(80, 100)

        # Write updated schematic
        with open(schematic_file, "w") as f:
            json.dump(schematic_data, f, indent=2)

        if ctx:
            await ctx.report_progress(90, 100)
            await ctx.info("Generating visual feedback for power symbol...")

            # Generate visual feedback after adding power symbol
            try:
                from kicad_mcp.tools.visualization_tools import capture_schematic_screenshot

                screenshot_result = await capture_schematic_screenshot(project_path, ctx)
                if screenshot_result:
                    await ctx.info("✓ Power symbol screenshot captured")
                else:
                    await ctx.info(
                        "⚠ Screenshot capture failed - proceeding without visual feedback"
                    )
            except ImportError:
                await ctx.info("⚠ Visualization tools not available")
            except Exception as e:
                await ctx.info(f"⚠ Screenshot capture failed: {str(e)}")

            await ctx.report_progress(100, 100)
            await ctx.info(f"Successfully added power symbol {power_ref}")

        result = {
            "success": True,
            "component_reference": power_ref,
            "component_uuid": component_uuid,
            "position": [x_position, y_position],
        }

        if result["success"]:
            result["power_type"] = power_type

        return result

    except Exception as e:
        if ctx:
            await ctx.info(f"Error adding power symbol: {str(e)}")
        return {"success": False, "error": str(e)}


def _read_schematic_for_modification(schematic_file: str) -> dict[str, Any]:
    """Read a schematic file and determine if it can be modified by JSON operations.

    Returns appropriate error if the file is in S-expression format.
    """
    with open(schematic_file) as f:
        content = f.read().strip()

    # Check if it's S-expression format (which KiCad expects)
    if content.startswith("(kicad_sch"):
        return {
            "success": False,
            "error": "Schematic is in S-expression format. Use create_kicad_schematic_from_text for modifying S-expression schematics.",
            "suggestion": "Use the text-to-schematic tools for better S-expression support",
            "schematic_file": schematic_file,
        }
    else:
        # Legacy JSON format
        try:
            return {"success": True, "data": json.loads(content)}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Schematic file is not valid JSON or S-expression format",
            }


def _parse_sexpr_for_validation(content: str) -> dict[str, Any]:
    """Parse S-expression schematic content for validation purposes.

    This is a simplified parser that extracts basic information needed for validation.
    """
    result = {"symbol": [], "wire": []}

    # Find all symbol instances in the schematic
    # Pattern matches: (symbol (lib_id "Device:R") ... (property "Reference" "R1") ... (property "Value" "10k") ... )
    re.findall(r'\(symbol\s+[^)]*\(lib_id\s+"([^"]+)"[^)]*\)', content, re.DOTALL)

    # For each symbol, find its properties
    for symbol_block in re.finditer(r'\(symbol[^)]*\(lib_id\s+"([^"]+)".*?\)', content, re.DOTALL):
        lib_id = symbol_block.group(1)
        symbol_content = symbol_block.group(0)

        # Extract Reference and Value properties
        ref_match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', symbol_content)
        val_match = re.search(r'\(property\s+"Value"\s+"([^"]+)"', symbol_content)

        reference = ref_match.group(1) if ref_match else "Unknown"
        value = val_match.group(1) if val_match else ""

        result["symbol"].append(
            {
                "lib_id": lib_id,
                "property": [
                    {"name": "Reference", "value": reference},
                    {"name": "Value", "value": value},
                ],
            }
        )

    # Find wire connections
    # Pattern matches: (wire (pts (xy 63.5 87.63) (xy 74.93 87.63)))
    wires = re.findall(
        r"\(wire\s+\(pts\s+\(xy\s+[\d.]+\s+[\d.]+\)\s+\(xy\s+[\d.]+\s+[\d.]+\)\)", content
    )
    result["wire"] = [{"found": True} for _ in wires]

    return result


async def validate_schematic(project_path: str, ctx: Context = None) -> dict[str, Any]:
    """Validate a KiCad schematic for common issues.

    Args:
        project_path: Path to the KiCad project file (.kicad_pro)
        ctx: Context for MCP communication

    Returns:
        Dictionary with validation results
    """
    try:
        if ctx:
            await ctx.info("Validating schematic")
            await ctx.report_progress(10, 100)

        # Get project files
        files = get_project_files(project_path)
        if "schematic" not in files:
            return {"success": False, "error": "No schematic file found in project"}

        schematic_file = files["schematic"]

        if ctx:
            await ctx.report_progress(30, 100)

        # Read schematic file - determine if it's S-expression or JSON format
        with open(schematic_file) as f:
            content = f.read().strip()

        # Check if it's S-expression format
        if content.startswith("(kicad_sch"):
            # Parse S-expression format
            schematic_data = _parse_sexpr_for_validation(content)
        else:
            # Try JSON format (legacy)
            try:
                schematic_data = json.loads(content)
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Schematic file is neither valid S-expression nor JSON format",
                }

        validation_results = {
            "success": True,
            "issues": [],
            "warnings": [],
            "component_count": 0,
            "wire_count": 0,
            "unconnected_pins": [],
        }

        # Count components and wires
        if "symbol" in schematic_data:
            validation_results["component_count"] = len(schematic_data["symbol"])

        if "wire" in schematic_data:
            validation_results["wire_count"] = len(schematic_data["wire"])

        if ctx:
            await ctx.report_progress(60, 100)

        # Check for components without values
        if "symbol" in schematic_data:
            for symbol in schematic_data["symbol"]:
                ref = "Unknown"
                value = "Unknown"

                if "property" in symbol:
                    for prop in symbol["property"]:
                        if prop["name"] == "Reference":
                            ref = prop["value"]
                        elif prop["name"] == "Value":
                            value = prop["value"]

                if not value or value == "Unknown" or value == "":
                    validation_results["warnings"].append(f"Component {ref} has no value assigned")

        # Check for empty schematic
        if validation_results["component_count"] == 0:
            validation_results["warnings"].append("Schematic contains no components")

        # Check for isolated components (no wires)
        if validation_results["component_count"] > 0 and validation_results["wire_count"] == 0:
            validation_results["warnings"].append(
                "Schematic has components but no wire connections"
            )

        if ctx:
            await ctx.report_progress(100, 100)
            await ctx.info(
                f"Validation complete: {len(validation_results['issues'])} issues, {len(validation_results['warnings'])} warnings"
            )

        return validation_results

    except Exception as e:
        if ctx:
            await ctx.info(f"Error validating schematic: {str(e)}")
        return {"success": False, "error": str(e)}


# Alias functions for compatibility with tests and other modules
create_new_circuit = create_new_project
add_component_to_circuit = add_component
connect_components = create_wire_connection


async def add_power_symbols(
    project_path: str, power_symbols: list[dict[str, Any]], ctx: Context = None
) -> dict[str, Any]:
    """Add multiple power symbols to the schematic.

    Args:
        project_path: Path to the KiCad project file (.kicad_pro)
        power_symbols: List of power symbol definitions with power_type, x_position, y_position
        ctx: Context for MCP communication

    Returns:
        Dictionary with batch addition results
    """
    results = []
    for symbol_def in power_symbols:
        result = await add_power_symbol(
            project_path=project_path,
            power_type=symbol_def["power_type"],
            x_position=symbol_def["x_position"],
            y_position=symbol_def["y_position"],
            ctx=ctx,
        )
        results.append(result)

    return {"success": all(r["success"] for r in results), "results": results}


# Alias for validation
validate_circuit = validate_schematic


def get_kicad_cli_path() -> str | None:
    """Get the path to kicad-cli executable based on the operating system.

    Returns:
        Path to kicad-cli executable or None if not found
    """
    if system == "Darwin":  # macOS
        kicad_cli_path = os.path.join(KICAD_APP_PATH, "Contents/MacOS/kicad-cli")
        if os.path.exists(kicad_cli_path):
            return kicad_cli_path
        elif shutil.which("kicad-cli") is not None:
            return "kicad-cli"
    elif system == "Windows":
        kicad_cli_path = os.path.join(KICAD_APP_PATH, "bin", "kicad-cli.exe")
        if os.path.exists(kicad_cli_path):
            return kicad_cli_path
        elif shutil.which("kicad-cli.exe") is not None:
            return "kicad-cli.exe"
        elif shutil.which("kicad-cli") is not None:
            return "kicad-cli"
    elif system == "Linux":
        kicad_cli = shutil.which("kicad-cli")
        if kicad_cli:
            return kicad_cli

    return None


def register_circuit_tools(mcp: FastMCP) -> None:
    """Register circuit creation tools with the MCP server.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.tool(name="create_new_project")
    async def create_new_project_tool(
        project_name: str, project_path: str, description: str = "", ctx: Context = None
    ) -> dict[str, Any]:
        """Create a new KiCad project with basic files."""
        return await create_new_project(project_name, project_path, description, ctx)

    @mcp.tool(name="add_component")
    async def add_component_tool(
        project_path: str,
        component_reference: str,
        component_value: str,
        symbol_library: str,
        symbol_name: str,
        x_position: float,
        y_position: float,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Add a component to a KiCad schematic."""
        return await add_component(
            project_path,
            component_reference,
            component_value,
            symbol_library,
            symbol_name,
            x_position,
            y_position,
            ctx,
        )

    @mcp.tool(name="create_wire_connection")
    async def create_wire_connection_tool(
        project_path: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Create a wire connection between two points in a schematic."""
        return await create_wire_connection(project_path, start_x, start_y, end_x, end_y, ctx)

    @mcp.tool(name="add_power_symbol")
    async def add_power_symbol_tool(
        project_path: str,
        power_type: str,
        x_position: float,
        y_position: float,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Add a power symbol (VCC, GND, etc.) to the schematic."""
        return await add_power_symbol(project_path, power_type, x_position, y_position, ctx)

    @mcp.tool(name="validate_schematic")
    async def validate_schematic_tool(project_path: str, ctx: Context = None) -> dict[str, Any]:
        """Validate a KiCad schematic for common issues."""
        return await validate_schematic(project_path, ctx)
