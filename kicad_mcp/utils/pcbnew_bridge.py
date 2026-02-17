"""
Bridge to KiCad's pcbnew Python API.

Runs pcbnew operations via KiCad's bundled Python 3.9 as a subprocess,
since pcbnew is a compiled C++ module that only works with KiCad's own Python.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_kicad_python() -> Optional[str]:
    """Find KiCad's bundled Python interpreter."""
    system = platform.system()

    if system == "Darwin":
        candidates = [
            "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3.9",
            "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3",
        ]
    elif system == "Linux":
        candidates = [
            "/usr/bin/python3",  # pcbnew typically installed system-wide on Linux
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Program Files\KiCad\bin\python.exe",
        ]
    else:
        candidates = []

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def _get_kicad_env() -> Dict[str, str]:
    """Build environment variables for KiCad's Python."""
    env = os.environ.copy()
    system = platform.system()

    if system == "Darwin":
        kicad_app = "/Applications/KiCad/KiCad.app"
        env["PYTHONPATH"] = (
            f"{kicad_app}/Contents/Frameworks/Python.framework"
            f"/Versions/3.9/lib/python3.9/site-packages"
        )
        env["DYLD_FRAMEWORK_PATH"] = f"{kicad_app}/Contents/Frameworks"

    return env


def run_pcbnew_script(script: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Run a Python script using KiCad's Python with pcbnew available.

    The script MUST print a single JSON object to stdout as its final output.
    Any other stdout output should be avoided; use stderr for logging.

    Args:
        script: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Parsed JSON dict from the script's stdout.

    Raises:
        RuntimeError: If KiCad Python is not found or the script fails.
    """
    kicad_python = _get_kicad_python()
    if not kicad_python:
        raise RuntimeError(
            "KiCad Python interpreter not found. "
            "Ensure KiCad is installed."
        )

    # Write script to a temp file (avoids shell quoting issues)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        env = _get_kicad_env()
        result = subprocess.run(
            [kicad_python, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Filter out the harmless wxApp assert
            stderr_lines = [
                line for line in stderr.split("\n")
                if "assert" not in line and "wxApp" not in line
            ]
            error_msg = "\n".join(stderr_lines).strip()
            raise RuntimeError(
                f"pcbnew script failed (exit {result.returncode}): {error_msg}"
            )

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("pcbnew script produced no output")

        # Take the last line as JSON (in case of stray prints)
        json_line = stdout.split("\n")[-1]
        return json.loads(json_line)

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pcbnew script timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"pcbnew script output is not valid JSON: {e}\n"
            f"Output was: {result.stdout[:500]}"
        )
    finally:
        os.unlink(script_path)
