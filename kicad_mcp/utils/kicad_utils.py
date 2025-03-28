"""
KiCad-specific utility functions.
"""
import os
import subprocess
from typing import Dict, List, Any

from kicad_mcp.config import KICAD_USER_DIR, KICAD_APP_PATH, KICAD_EXTENSIONS, ADDITIONAL_SEARCH_PATHS

def find_kicad_projects() -> List[Dict[str, Any]]:
    """Find KiCad projects in the user's directory.
    
    Returns:
        List of dictionaries with project information
    """
    projects = []

    # Search directories to look for KiCad projects
    search_dirs = [KICAD_USER_DIR] + ADDITIONAL_SEARCH_PATHS

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            print(f"Search directory does not exist: {search_dir}")
            continue
        
        print(f"Scanning directory: {search_dir}")
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith(KICAD_EXTENSIONS["project"]):
                    project_path = os.path.join(root, file)
                    rel_path = os.path.relpath(project_path, search_dir)
                    project_name = get_project_name_from_path(project_path)

                    print(f"Found KiCad project: {project_path}")
                    projects.append({
                        "name": project_name,
                        "path": project_path,
                        "relative_path": rel_path,
                        "modified": os.path.getmtime(project_path)
                    })
    
    print(f"Found {len(projects)} KiCad projects")
    return projects

def get_project_name_from_path(project_path: str) -> str:
    """Extract the project name from a .kicad_pro file path.
    
    Args:
        project_path: Path to the .kicad_pro file
        
    Returns:
        Project name without extension
    """
    basename = os.path.basename(project_path)
    return basename[:-len(KICAD_EXTENSIONS["project"])]


def open_kicad_project(project_path: str) -> Dict[str, Any]:
    """Open a KiCad project using the KiCad application.
    
    Args:
        project_path: Path to the .kicad_pro file
        
    Returns:
        Dictionary with result information
    """
    if not os.path.exists(project_path):
        return {"success": False, "error": f"Project not found: {project_path}"}
    
    try:
        # On MacOS, use the 'open' command to open the project in KiCad
        cmd = ["open", "-a", KICAD_APP_PATH, project_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "success": result.returncode == 0,
            "command": " ".join(cmd),
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}
