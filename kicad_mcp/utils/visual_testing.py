"""
Visual testing utilities for KiCad MCP.
Provides functions for screenshot comparison and visual regression testing.
"""
import os
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path


class VisualTestUtils:
    """Utilities for visual testing and screenshot management."""
    
    def __init__(self, output_dir: str = "tests/visual_output"):
        """Initialize visual test utilities.
        
        Args:
            output_dir: Directory to store visual test outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def capture_project_screenshot(self, project_path: str, test_name: str) -> Optional[str]:
        """Capture screenshot of a KiCad project for testing.
        
        Args:
            project_path: Path to the KiCad project
            test_name: Name for the test (used in filename)
            
        Returns:
            Path to the generated screenshot or None if failed
        """
        try:
            # Import the visualization tools
            from kicad_mcp.tools.visualization_tools import export_schematic_to_svg, convert_svg_to_png_file
            from unittest.mock import AsyncMock
            
            # Create mock context
            ctx = AsyncMock()
            ctx.info = AsyncMock()
            ctx.report_progress = AsyncMock()
            
            # Export to SVG
            svg_output_dir = str(self.output_dir / test_name)
            os.makedirs(svg_output_dir, exist_ok=True)
            
            # Get schematic file
            from kicad_mcp.utils.file_utils import get_project_files
            files = get_project_files(project_path)
            if "schematic" not in files:
                return None
                
            schematic_file = files["schematic"]
            
            # Export to SVG
            svg_result = await export_schematic_to_svg(schematic_file, svg_output_dir, ctx)
            if not svg_result["success"]:
                return None
            
            # Convert to PNG
            svg_file = svg_result["svg_file"]
            png_file = svg_file.replace('.svg', '.png')
            
            png_result = await convert_svg_to_png_file(svg_file, png_file, ctx)
            if not png_result["success"]:
                return None
                
            return png_file
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def list_test_screenshots(self) -> List[str]:
        """List all test screenshots in the output directory.
        
        Returns:
            List of screenshot file paths
        """
        screenshots = []
        for file_path in self.output_dir.rglob("*.png"):
            screenshots.append(str(file_path))
        return sorted(screenshots)
    
    def cleanup_test_screenshots(self, test_pattern: Optional[str] = None) -> int:
        """Clean up test screenshots.
        
        Args:
            test_pattern: Pattern to match test names (if None, cleans all)
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        for file_path in self.output_dir.rglob("*.png"):
            if test_pattern is None or test_pattern in str(file_path):
                try:
                    file_path.unlink()
                    cleaned_count += 1
                except Exception as e:
                    print(f"Error cleaning up {file_path}: {e}")
        
        # Also clean up empty directories
        for dir_path in self.output_dir.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception as e:
                    print(f"Error removing empty directory {dir_path}: {e}")
        
        return cleaned_count
    
    def create_test_report(self, test_results: List[Dict[str, Any]]) -> str:
        """Create HTML test report with screenshots.
        
        Args:
            test_results: List of test result dictionaries
            
        Returns:
            Path to the generated HTML report
        """
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>KiCad MCP Visual Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .test-result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        .test-success { background-color: #d4edda; border-color: #c3e6cb; }
        .test-failure { background-color: #f8d7da; border-color: #f5c6cb; }
        .screenshot { max-width: 800px; margin: 10px 0; }
        .comparison { display: flex; gap: 20px; }
        .comparison img { max-width: 400px; }
    </style>
</head>
<body>
    <h1>KiCad MCP Visual Test Report</h1>
    <p>Generated on: <span id="timestamp"></span></p>
"""
        
        for result in test_results:
            status_class = "test-success" if result.get("success", False) else "test-failure"
            html_content += f"""
    <div class="test-result {status_class}">
        <h2>{result.get('test_name', 'Unknown Test')}</h2>
        <p><strong>Status:</strong> {'✅ PASS' if result.get('success', False) else '❌ FAIL'}</p>
"""
            
            if result.get('error'):
                html_content += f"<p><strong>Error:</strong> {result['error']}</p>"
            
            if result.get('screenshot'):
                screenshot_rel = os.path.relpath(result['screenshot'], self.output_dir)
                html_content += f"""
        <p><strong>Screenshot:</strong></p>
        <img class="screenshot" src="{screenshot_rel}" alt="Test screenshot">
"""
            
            if result.get('before_screenshot') and result.get('after_screenshot'):
                before_rel = os.path.relpath(result['before_screenshot'], self.output_dir)
                after_rel = os.path.relpath(result['after_screenshot'], self.output_dir)
                html_content += f"""
        <p><strong>Comparison:</strong></p>
        <div class="comparison">
            <div>
                <h4>Before</h4>
                <img src="{before_rel}" alt="Before screenshot">
            </div>
            <div>
                <h4>After</h4>
                <img src="{after_rel}" alt="After screenshot">
            </div>
        </div>
"""
            
            html_content += "    </div>"
        
        html_content += """
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
"""
        
        report_path = self.output_dir / "test_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)


async def test_visualization_tools():
    """Test the visualization tools with example projects."""
    print("🔧 Testing KiCad Visualization Tools")
    print("=" * 40)
    
    visual_utils = VisualTestUtils()
    test_results = []
    
    # Test 1: LED Blinker Example
    led_project = "led_blinker_test_output/led_blinker_test.kicad_pro"
    if os.path.exists(led_project):
        print("\n1. Testing LED Blinker visualization...")
        screenshot = await visual_utils.capture_project_screenshot(led_project, "led_blinker")
        
        if screenshot and os.path.exists(screenshot):
            test_results.append({
                "test_name": "LED Blinker Visualization",
                "success": True,
                "screenshot": screenshot
            })
            print(f"✅ LED Blinker screenshot: {screenshot}")
        else:
            test_results.append({
                "test_name": "LED Blinker Visualization",
                "success": False,
                "error": "Failed to generate screenshot"
            })
            print("❌ LED Blinker screenshot failed")
    else:
        print(f"⚠️  LED Blinker project not found: {led_project}")
    
    # Test 2: ESP32 Example  
    esp32_project = "esp32_dev_board_example/esp32_dev_board.kicad_pro"
    if os.path.exists(esp32_project):
        print("\n2. Testing ESP32 visualization...")
        screenshot = await visual_utils.capture_project_screenshot(esp32_project, "esp32_dev_board")
        
        if screenshot and os.path.exists(screenshot):
            test_results.append({
                "test_name": "ESP32 Dev Board Visualization", 
                "success": True,
                "screenshot": screenshot
            })
            print(f"✅ ESP32 screenshot: {screenshot}")
        else:
            test_results.append({
                "test_name": "ESP32 Dev Board Visualization",
                "success": False,
                "error": "Failed to generate screenshot"
            })
            print("❌ ESP32 screenshot failed")
    else:
        print(f"⚠️  ESP32 project not found: {esp32_project}")
    
    # Generate test report
    if test_results:
        report_path = visual_utils.create_test_report(test_results)
        print(f"\n📊 Test report generated: {report_path}")
        
        # Summary
        success_count = sum(1 for result in test_results if result.get("success", False))
        total_count = len(test_results)
        print(f"\n📈 Results: {success_count}/{total_count} tests passed")
        
        if success_count == total_count:
            print("🎉 All visualization tests passed!")
        else:
            print("⚠️  Some visualization tests failed")
    else:
        print("\n⚠️  No test projects found to visualize")
    
    return test_results


if __name__ == "__main__":
    # Run the test when executed directly
    asyncio.run(test_visualization_tools())
