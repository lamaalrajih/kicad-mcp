"""Simple test of kicad-cli netlist export without imports."""
import subprocess
import os

schematic_path = r"c:\Users\geoff\Desktop\projects\kicad-astro-daughterboard2\Astro-DB_rev00005\Astro-DB_rev00005.kicad_sch"

# Common KiCad CLI locations on Windows
kicad_cli_paths = [
    r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe",
    r"C:\Program Files\KiCad\8.0\bin\kicad-cli.exe",
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
]

kicad_cli = None
for path in kicad_cli_paths:
    if os.path.exists(path):
        kicad_cli = path
        print(f"Found kicad-cli at: {kicad_cli}")
        break

if not kicad_cli:
    print("ERROR: kicad-cli not found in common locations")
    exit(1)

# Test command
cmd = [
    kicad_cli,
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
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    if os.path.exists("test_output.xml"):
        size = os.path.getsize("test_output.xml")
        print(f"SUCCESS: Output file created ({size} bytes)")
    else:
        print("ERROR: Output file not created")

except subprocess.TimeoutExpired:
    print("ERROR: Command timed out after 30 seconds")
except Exception as e:
    print(f"ERROR: {e}")
