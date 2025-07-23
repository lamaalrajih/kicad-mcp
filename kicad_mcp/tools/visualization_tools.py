"""
Visualization tools for KiCad schematics and PCBs.
Provides screenshot and rendering capabilities for debugging and testing.
"""
import os
import tempfile
import subprocess
import shutil
import asyncio
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context, Image

from kicad_mcp.utils.file_utils import get_project_files


def register_visualization_tools(mcp: FastMCP) -> None:
    """Register visualization tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    
    @mcp.tool()
    async def export_schematic_svg(project_path: str, ctx: Context) -> Dict[str, Any]:
        """Export KiCad schematic to SVG format using kicad-cli.

        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            ctx: Context for MCP communication

        Returns:
            Dictionary with export results including SVG file path
        """
        try:
            await ctx.info(f"Exporting schematic to SVG: {project_path}")
            
            if not os.path.exists(project_path):
                return {
                    "success": False,
                    "error": f"Project not found: {project_path}"
                }

            # Get schematic file from project
            files = get_project_files(project_path)
            if "schematic" not in files:
                return {
                    "success": False,
                    "error": "Schematic file not found in project"
                }

            schematic_file = files["schematic"]
            await ctx.info(f"Found schematic file: {schematic_file}")

            # Create output directory for SVG
            project_dir = os.path.dirname(project_path)
            svg_output_dir = os.path.join(project_dir, "visual_output")
            os.makedirs(svg_output_dir, exist_ok=True)

            # Export to SVG using kicad-cli
            svg_result = await export_schematic_to_svg(schematic_file, svg_output_dir, ctx)
            
            if svg_result["success"]:
                await ctx.info(f"SVG export successful: {svg_result['svg_file']}")
                return {
                    "success": True,
                    "svg_file": svg_result["svg_file"],
                    "output_dir": svg_output_dir
                }
            else:
                return {
                    "success": False,
                    "error": svg_result["error"]
                }
                
        except Exception as e:
            error_msg = f"Error exporting schematic to SVG: {str(e)}"
            await ctx.info(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    @mcp.tool()
    async def convert_svg_to_png(svg_path: str, ctx: Context) -> Dict[str, Any]:
        """Convert SVG file to PNG format.

        Args:
            svg_path: Path to the SVG file
            ctx: Context for MCP communication

        Returns:
            Dictionary with conversion results including PNG file path
        """
        try:
            await ctx.info(f"Converting SVG to PNG: {svg_path}")
            
            if not os.path.exists(svg_path):
                return {
                    "success": False,
                    "error": f"SVG file not found: {svg_path}"
                }

            # Generate PNG path
            png_path = svg_path.replace('.svg', '.png')
            
            # Convert using cairosvg
            conversion_result = await convert_svg_to_png_file(svg_path, png_path, ctx)
            
            if conversion_result["success"]:
                await ctx.info(f"PNG conversion successful: {png_path}")
                return {
                    "success": True,
                    "png_file": png_path,
                    "svg_file": svg_path
                }
            else:
                return {
                    "success": False,
                    "error": conversion_result["error"]
                }
                
        except Exception as e:
            error_msg = f"Error converting SVG to PNG: {str(e)}"
            await ctx.info(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    @mcp.tool()
    async def capture_schematic_screenshot(project_path: str, ctx: Context) -> Optional[Image]:
        """Capture screenshot of KiCad schematic as PNG image.

        Args:
            project_path: Path to the KiCad project file (.kicad_pro)
            ctx: Context for MCP communication

        Returns:
            Image object containing the schematic screenshot or None if failed
        """
        try:
            await ctx.info(f"Capturing schematic screenshot: {project_path}")
            
            # First export to SVG
            svg_result = await export_schematic_svg(project_path, ctx)
            if not svg_result["success"]:
                await ctx.info(f"SVG export failed: {svg_result['error']}")
                return None

            # Then convert to PNG
            png_result = await convert_svg_to_png(svg_result["svg_file"], ctx)
            if not png_result["success"]:
                await ctx.info(f"PNG conversion failed: {png_result['error']}")
                return None

            # Load PNG as Image object
            png_file = png_result["png_file"]
            if os.path.exists(png_file):
                with open(png_file, 'rb') as f:
                    image_data = f.read()
                
                await ctx.info(f"Screenshot captured successfully: {png_file}")
                return Image(data=image_data, format="png")
            else:
                await ctx.info("PNG file not found after conversion")
                return None
                
        except Exception as e:
            error_msg = f"Error capturing schematic screenshot: {str(e)}"
            await ctx.info(error_msg)
            return None

    @mcp.tool()
    async def create_visual_comparison(before_project: str, after_project: str, ctx: Context) -> Dict[str, Any]:
        """Create side-by-side visual comparison of two schematics.

        Args:
            before_project: Path to the "before" KiCad project file
            after_project: Path to the "after" KiCad project file  
            ctx: Context for MCP communication

        Returns:
            Dictionary with comparison results and image paths
        """
        try:
            await ctx.info(f"Creating visual comparison: {before_project} vs {after_project}")
            
            # Capture screenshots of both projects
            before_image = await capture_schematic_screenshot(before_project, ctx)
            after_image = await capture_schematic_screenshot(after_project, ctx)
            
            if not before_image or not after_image:
                return {
                    "success": False,
                    "error": "Failed to capture one or both screenshots"
                }

            # Create comparison image (side-by-side)
            comparison_result = await create_side_by_side_comparison(
                before_image, after_image, before_project, after_project, ctx
            )
            
            if comparison_result["success"]:
                await ctx.info(f"Visual comparison created: {comparison_result['comparison_file']}")
                return comparison_result
            else:
                return {
                    "success": False,
                    "error": comparison_result["error"]
                }
                
        except Exception as e:
            error_msg = f"Error creating visual comparison: {str(e)}"
            await ctx.info(error_msg)
            return {
                "success": False,
                "error": error_msg
            }


# Helper functions for KiCad CLI operations

async def export_schematic_to_svg(schematic_file: str, output_dir: str, ctx: Context) -> Dict[str, Any]:
    """Export schematic to SVG using kicad-cli.
    
    Args:
        schematic_file: Path to .kicad_sch file
        output_dir: Directory to save SVG files
        ctx: Context for progress reporting
        
    Returns:
        Dictionary with export results
    """
    try:
        # Check if kicad-cli is available (try macOS path first)
        kicad_cli_paths = [
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            "kicad-cli"  # fallback to PATH
        ]
        
        kicad_cli = None
        for cli_path in kicad_cli_paths:
            try:
                result = subprocess.run(
                    [cli_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    kicad_cli = cli_path
                    break
            except FileNotFoundError:
                continue
        
        if kicad_cli is None:
            # Fallback to mock renderer
            await ctx.info("KiCad CLI not available, using mock renderer")
            return await export_schematic_to_svg_mock(schematic_file, output_dir, ctx)
        
        await ctx.report_progress(25, 100)
        
        # Export schematic to SVG
        cmd = [
            kicad_cli, "sch", "export", "svg",
            "--output", output_dir,
            "--no-background-color",  # Transparent background
            "--exclude-drawing-sheet",  # Skip title block for cleaner view
            schematic_file
        ]
        
        await ctx.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        await ctx.report_progress(75, 100)
        
        if result.returncode == 0:
            # Find the generated SVG file
            schematic_name = os.path.splitext(os.path.basename(schematic_file))[0]
            svg_file = os.path.join(output_dir, f"{schematic_name}.svg")
            
            if os.path.exists(svg_file):
                await ctx.report_progress(100, 100)
                return {
                    "success": True,
                    "svg_file": svg_file,
                    "command_output": result.stdout
                }
            else:
                # Look for any SVG files in output directory
                svg_files = [f for f in os.listdir(output_dir) if f.endswith('.svg')]
                if svg_files:
                    svg_file = os.path.join(output_dir, svg_files[0])
                    return {
                        "success": True,
                        "svg_file": svg_file,
                        "command_output": result.stdout
                    }
                else:
                    return {
                        "success": False,
                        "error": f"SVG file not found after export. Expected: {svg_file}",
                        "command_output": result.stdout,
                        "command_error": result.stderr
                    }
        else:
            return {
                "success": False,
                "error": f"kicad-cli export failed: {result.stderr}",
                "command_output": result.stdout,
                "command_error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "kicad-cli export timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during SVG export: {str(e)}"
        }


async def convert_svg_to_png_file(svg_path: str, png_path: str, ctx: Context) -> Dict[str, Any]:
    """Convert SVG file to PNG using cairosvg.
    
    Args:
        svg_path: Path to input SVG file
        png_path: Path to output PNG file  
        ctx: Context for progress reporting
        
    Returns:
        Dictionary with conversion results
    """
    try:
        # Try to import cairosvg
        try:
            import cairosvg
        except ImportError:
            return {
                "success": False,
                "error": "cairosvg not installed. Install with: pip install cairosvg"
            }

        await ctx.report_progress(25, 100)
        
        # Convert SVG to PNG
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=1200,   # High resolution for clarity
            output_height=800,
            background_color='white'  # White background for better visibility
        )
        
        await ctx.report_progress(75, 100)
        
        if os.path.exists(png_path):
            # Get file size for verification
            file_size = os.path.getsize(png_path)
            await ctx.report_progress(100, 100)
            return {
                "success": True,
                "png_file": png_path,
                "file_size": file_size
            }
        else:
            return {
                "success": False,
                "error": "PNG file not created"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"SVG to PNG conversion failed: {str(e)}"
        }


async def create_side_by_side_comparison(before_image: Image, after_image: Image, 
                                       before_path: str, after_path: str, ctx: Context) -> Dict[str, Any]:
    """Create side-by-side comparison image.
    
    Args:
        before_image: Image object for "before" screenshot
        after_image: Image object for "after" screenshot
        before_path: Path to before project (for labeling)
        after_path: Path to after project (for labeling)
        ctx: Context for progress reporting
        
    Returns:
        Dictionary with comparison results
    """
    try:
        # Try to import PIL
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
        except ImportError:
            return {
                "success": False,
                "error": "Pillow not installed. Install with: pip install Pillow"
            }

        await ctx.report_progress(25, 100)
        
        # Load images from bytes
        before_pil = PILImage.open(io.BytesIO(before_image.data))
        after_pil = PILImage.open(io.BytesIO(after_image.data))
        
        # Resize images to same height
        max_height = max(before_pil.height, after_pil.height)
        before_pil = before_pil.resize((
            int(before_pil.width * max_height / before_pil.height),
            max_height
        ))
        after_pil = after_pil.resize((
            int(after_pil.width * max_height / after_pil.height), 
            max_height
        ))
        
        await ctx.report_progress(50, 100)
        
        # Create comparison image
        total_width = before_pil.width + after_pil.width + 50  # 50px padding
        comparison = PILImage.new('RGB', (total_width, max_height + 100), 'white')
        
        # Paste images
        comparison.paste(before_pil, (0, 50))
        comparison.paste(after_pil, (before_pil.width + 50, 50))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("Arial", 24)
        except:
            font = ImageFont.load_default()
        
        before_label = f"Before: {os.path.basename(before_path)}"
        after_label = f"After: {os.path.basename(after_path)}"
        
        draw.text((10, 10), before_label, fill='black', font=font)
        draw.text((before_pil.width + 60, 10), after_label, fill='black', font=font)
        
        await ctx.report_progress(75, 100)
        
        # Save comparison image
        comparison_dir = os.path.join(os.path.dirname(before_path), "visual_output")
        os.makedirs(comparison_dir, exist_ok=True)
        comparison_file = os.path.join(comparison_dir, "schematic_comparison.png")
        comparison.save(comparison_file)
        
        await ctx.report_progress(100, 100)
        
        return {
            "success": True,
            "comparison_file": comparison_file,
            "before_project": before_path,
            "after_project": after_path
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create comparison image: {str(e)}"
        }


# Add missing import
import io


async def export_schematic_to_svg_mock(schematic_file: str, output_dir: str, ctx: Context) -> Dict[str, Any]:
    """Export schematic to SVG using mock renderer when KiCad CLI is not available.
    
    Args:
        schematic_file: Path to .kicad_sch file
        output_dir: Directory to save SVG files
        ctx: Context for progress reporting
        
    Returns:
        Dictionary with export results
    """
    try:
        from kicad_mcp.utils.mock_renderer import MockSchematicRenderer
        
        await ctx.report_progress(25, 100)
        
        # Generate SVG using mock renderer
        renderer = MockSchematicRenderer()
        schematic_name = os.path.splitext(os.path.basename(schematic_file))[0]
        svg_file = os.path.join(output_dir, f"{schematic_name}.svg")
        
        await ctx.report_progress(50, 100)
        
        success = renderer.render_schematic_file(schematic_file, svg_file)
        
        await ctx.report_progress(100, 100)
        
        if success and os.path.exists(svg_file):
            return {
                "success": True,
                "svg_file": svg_file,
                "command_output": "Generated using mock renderer"
            }
        else:
            return {
                "success": False,
                "error": "Mock renderer failed to generate SVG"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Mock renderer error: {str(e)}"
        }
