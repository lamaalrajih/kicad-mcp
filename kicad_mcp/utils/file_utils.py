"""
File handling utilities for KiCad MCP Server.
"""

import json
import os
from typing import Any

from kicad_mcp.utils.kicad_utils import get_project_name_from_path


def get_project_files(project_path: str) -> dict[str, str]:
    """Get all files related to a KiCad project.

    Args:
        project_path: Path to the .kicad_pro file

    Returns:
        Dictionary mapping file types to file paths
    """
    from kicad_mcp.config import DATA_EXTENSIONS, KICAD_EXTENSIONS

    project_dir = os.path.dirname(project_path)
    project_name = get_project_name_from_path(project_path)

    files = {}

    # Check for standard KiCad files
    for file_type, extension in KICAD_EXTENSIONS.items():
        if file_type == "project":
            # We already have the project file
            files[file_type] = project_path
            continue

        file_path = os.path.join(project_dir, f"{project_name}{extension}")
        if os.path.exists(file_path):
            files[file_type] = file_path

    # Check for data files
    try:
        for ext in DATA_EXTENSIONS:
            for file in os.listdir(project_dir):
                if file.startswith(project_name) and file.endswith(ext):
                    # Extract the type from filename (e.g., project_name-bom.csv -> bom)
                    file_type = file[len(project_name) :].strip("-_")
                    file_type = file_type.split(".")[0]
                    if not file_type:
                        file_type = ext[1:]  # Use extension if no specific type

                    files[file_type] = os.path.join(project_dir, file)
    except (OSError, FileNotFoundError):
        # Directory doesn't exist or can't be accessed - return what we have
        pass

    return files


def load_project_json(project_path: str) -> dict[str, Any] | None:
    """Load and parse a KiCad project file.

    Args:
        project_path: Path to the .kicad_pro file

    Returns:
        Parsed JSON data or None if parsing failed
    """
    try:
        with open(project_path) as f:
            return json.load(f)
    except Exception:
        return None
