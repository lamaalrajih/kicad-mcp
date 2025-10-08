"""
Design Rule Check (DRC) implementation using KiCad command-line interface.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, Any, Optional
from mcp.server.fastmcp import Context

from kicad_mcp.config import system


def run_drc_via_cli(pcb_file: str) -> Dict[str, Any]:
    """Run DRC using KiCad command line tools.

    Args:
        pcb_file: Path to the PCB file (.kicad_pcb)

    Returns:
        Dictionary with DRC results
    """
    results = {"success": False, "method": "cli", "pcb_file": pcb_file}

    try:
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Output file for DRC report
            output_file = os.path.join(temp_dir, "drc_report.json")

            # Find kicad-cli executable
            kicad_cli = find_kicad_cli()
            if not kicad_cli:
                print("kicad-cli not found in PATH or common installation locations")
                results["error"] = (
                    "kicad-cli not found. Please ensure KiCad 9.0+ is installed and kicad-cli is available."
                )
                return results

            # Report progress
            print("Running DRC using KiCad CLI...")

            # Build the DRC command
            cmd = [kicad_cli, "pcb", "drc", "--format", "json", "--output", output_file, pcb_file]

            print(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True)

            # Check if the command was successful
            if process.returncode != 0:
                print(f"DRC command failed with code {process.returncode}")
                print(f"Error output: {process.stderr}")
                results["error"] = f"DRC command failed: {process.stderr}"
                return results

            # Check if the output file was created
            if not os.path.exists(output_file):
                print("DRC report file not created")
                results["error"] = "DRC report file not created"
                return results

            # Read the DRC report
            with open(output_file, "r") as f:
                try:
                    drc_report = json.load(f)
                except json.JSONDecodeError:
                    print("Failed to parse DRC report JSON")
                    results["error"] = "Failed to parse DRC report JSON"
                    return results

            # Process the DRC report
            violations = drc_report.get("violations", [])
            violation_count = len(violations)
            print(f"DRC completed with {violation_count} violations")
            print(f"DRC completed with {violation_count} violations")

            # Categorize violations by type
            error_types = {}
            for violation in violations:
                error_type = violation.get("message", "Unknown")
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1

            # Create success response
            results = {
                "success": True,
                "method": "cli",
                "pcb_file": pcb_file,
                "total_violations": violation_count,
                "violation_categories": error_types,
                "violations": violations,
            }

            # Progress reporting removed
            return results

    except Exception as e:
        print(f"Error in CLI DRC: {str(e)}", exc_info=True)
        results["error"] = f"Error in CLI DRC: {str(e)}"
        return results


def find_kicad_cli() -> Optional[str]:
    """Find the kicad-cli executable in the system PATH.

    Returns:
        Path to kicad-cli if found, None otherwise
    """
    # Check if kicad-cli is in PATH
    try:
        if system == "Windows":
            # On Windows, check for kicad-cli.exe
            result = subprocess.run(["where", "kicad-cli.exe"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        else:
            # On Unix-like systems, use which
            result = subprocess.run(["which", "kicad-cli"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()

    except Exception as e:
        print(f"Error finding kicad-cli: {str(e)}")

    # If we get here, kicad-cli is not in PATH
    # Try common installation locations
    if system == "Windows":
        # Common Windows installation path
        potential_paths = [
            r"C:\Program Files\KiCad\bin\kicad-cli.exe",
            r"C:\Program Files (x86)\KiCad\bin\kicad-cli.exe",
        ]
    elif system == "Darwin":  # macOS
        # Common macOS installation paths
        potential_paths = [
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            "/Applications/KiCad/kicad-cli",
        ]
    else:  # Linux and other Unix-like systems
        # Common Linux installation paths
        potential_paths = [
            "/usr/bin/kicad-cli",
            "/usr/local/bin/kicad-cli",
            "/opt/kicad/bin/kicad-cli",
        ]

    # Check each potential path
    for path in potential_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    # If still not found, return None
    return None
