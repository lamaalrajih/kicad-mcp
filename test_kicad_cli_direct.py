"""Direct test of kicad-cli netlist export."""
import subprocess
import sys
import os

# Add the directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import directly from utils module
from kicad_mcp.utils import kicad_cli

schematic_path = r"c:\Users\geoff\Desktop\projects\kicad-astro-daughterboard2\Astro-DB_rev00005\Astro-DB_rev00005.kicad_sch"

print("Finding kicad-cli...")
kicad_cli_path = kicad_cli.find_kicad_cli()
if not kicad_cli_path:
    print("ERROR: kicad-cli not found")
    sys.exit(1)

print(f"Found kicad-cli at: {kicad_cli_path}")

# Test command
cmd = [
    kicad_cli_path,
    "sch",
    "export",
    "netlist",
    "--format", "kicadxml",
    "--output", "test_output.xml",
    schematic_path
]

print(f"Running: {' '.join(cmd)}")
print("Starting subprocess...")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(f"Return code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
except subprocess.TimeoutExpired:
    print("ERROR: Command timed out after 30 seconds")
except Exception as e:
    print(f"ERROR: {e}")
