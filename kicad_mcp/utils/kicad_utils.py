"""
KiCad-specific utility functions.
"""

import logging  # Import logging
import os
import sys  # Add sys import
from typing import Any

from kicad_mcp.config import (
    ADDITIONAL_SEARCH_PATHS,
    KICAD_APP_PATH,
    KICAD_EXTENSIONS,
    KICAD_USER_DIR,
    TIMEOUT_CONSTANTS,
)

from .path_validator import PathValidationError, validate_directory, validate_kicad_file
from .secure_subprocess import SecureSubprocessError, SecureSubprocessRunner

# Get PID for logging - Removed, handled by logging config
# _PID = os.getpid()


def find_kicad_projects() -> list[dict[str, Any]]:
    """Find KiCad projects in the user's directory.

    Returns:
        List of dictionaries with project information
    """
    projects = []
    logging.info("Attempting to find KiCad projects...")  # Log start
    # Search directories to look for KiCad projects
    raw_search_dirs = [KICAD_USER_DIR] + ADDITIONAL_SEARCH_PATHS
    logging.info(f"Raw KICAD_USER_DIR: '{KICAD_USER_DIR}'")
    logging.info(f"Raw ADDITIONAL_SEARCH_PATHS: {ADDITIONAL_SEARCH_PATHS}")
    logging.info(f"Raw search list before expansion: {raw_search_dirs}")

    expanded_search_dirs = []
    for raw_dir in raw_search_dirs:
        expanded_dir = os.path.expanduser(raw_dir)  # Expand ~ and ~user
        if expanded_dir not in expanded_search_dirs:
            expanded_search_dirs.append(expanded_dir)
        else:
            logging.info(f"Skipping duplicate expanded path: {expanded_dir}")

    logging.info(f"Expanded search directories: {expanded_search_dirs}")

    for search_dir in expanded_search_dirs:
        try:
            # Validate the search directory
            validated_dir = validate_directory(search_dir, must_exist=False)
            if not os.path.exists(validated_dir):
                logging.warning(f"Search directory does not exist: {search_dir}")
                continue

            logging.info(f"Scanning validated directory: {validated_dir}")
            # Use followlinks=True to follow symlinks if needed
            for root, _, files in os.walk(validated_dir, followlinks=True):
                for file in files:
                    if file.endswith(KICAD_EXTENSIONS["project"]):
                        project_path = os.path.join(root, file)
                        # Check if it's a real file and not a broken symlink
                        if not os.path.isfile(project_path):
                            logging.info(f"Skipping non-file/broken symlink: {project_path}")
                            continue

                        try:
                            # Validate the project file with path validation
                            validated_project = validate_kicad_file(
                                project_path, "project", must_exist=True
                            )

                            # Get modification time to ensure file is accessible
                            mod_time = os.path.getmtime(validated_project)
                            rel_path = os.path.relpath(validated_project, validated_dir)
                            project_name = get_project_name_from_path(validated_project)

                            logging.info(f"Found accessible KiCad project: {validated_project}")
                            projects.append(
                                {
                                    "name": project_name,
                                    "path": validated_project,
                                    "relative_path": rel_path,
                                    "modified": mod_time,
                                }
                            )
                        except (OSError, PathValidationError) as e:
                            logging.error(
                                f"Error accessing/validating project file {project_path}: {e}"
                            )
                            continue  # Skip if we can't access or validate it
        except PathValidationError as e:
            logging.warning(f"Invalid search directory {search_dir}: {e}")
            continue

    logging.info(f"Found {len(projects)} KiCad projects after scanning.")
    return projects


def find_kicad_projects_in_dirs(search_directories: list[str]) -> list[dict[str, Any]]:
    """Find KiCad projects in specific directories.

    Args:
        search_directories: List of directories to search

    Returns:
        List of dictionaries with project information
    """
    projects = []
    logging.info(f"Searching KiCad projects in specified directories: {search_directories}")

    for search_dir in search_directories:
        try:
            # Validate the search directory
            validated_dir = validate_directory(search_dir, must_exist=True)
            logging.info(f"Scanning validated directory: {validated_dir}")

            for root, _, files in os.walk(validated_dir, followlinks=True):
                for file in files:
                    if file.endswith(KICAD_EXTENSIONS["project"]):
                        project_path = os.path.join(root, file)
                        if not os.path.isfile(project_path):
                            continue

                        try:
                            # Validate the project file
                            validated_project = validate_kicad_file(
                                project_path, "project", must_exist=True
                            )

                            project_info = {
                                "name": get_project_name_from_path(validated_project),
                                "path": validated_project,
                                "directory": os.path.dirname(validated_project),
                            }
                            projects.append(project_info)
                            logging.info(f"Found KiCad project: {validated_project}")
                        except (PathValidationError, Exception) as e:
                            logging.error(
                                f"Error processing/validating project {project_path}: {str(e)}"
                            )
                            continue
        except PathValidationError as e:
            logging.warning(f"Invalid search directory {search_dir}: {e}")
            continue

    logging.info(f"Found {len(projects)} KiCad projects in specified directories")
    return projects


def get_project_name_from_path(project_path: str) -> str:
    """Extract the project name from a .kicad_pro file path.

    Args:
        project_path: Path to the .kicad_pro file

    Returns:
        Project name without extension
    """
    basename = os.path.basename(project_path)
    return basename[: -len(KICAD_EXTENSIONS["project"])]


def open_kicad_project(project_path: str) -> dict[str, Any]:
    """Open a KiCad project using the KiCad application.

    Args:
        project_path: Path to the .kicad_pro file

    Returns:
        Dictionary with result information
    """
    try:
        # Validate and sanitize the project path
        validated_project_path = validate_kicad_file(project_path, "project", must_exist=True)

        # Create secure subprocess runner
        subprocess_runner = SecureSubprocessRunner()

        # Determine command based on platform
        cmd = []
        allowed_commands = []

        if sys.platform == "darwin":  # macOS
            # On MacOS, use the 'open' command to open the project in KiCad
            cmd = ["open", "-a", KICAD_APP_PATH, validated_project_path]
            allowed_commands = ["open"]
        elif sys.platform == "linux":  # Linux
            # On Linux, use 'xdg-open'
            cmd = ["xdg-open", validated_project_path]
            allowed_commands = ["xdg-open"]
        else:
            # Fallback or error for unsupported OS
            return {"success": False, "error": f"Unsupported operating system: {sys.platform}"}

        # Execute command using secure subprocess runner
        result = subprocess_runner.run_safe_command(
            cmd, allowed_commands=allowed_commands, timeout=TIMEOUT_CONSTANTS["application_open"]
        )

        return {
            "success": result.returncode == 0,
            "command": " ".join(cmd),
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
        }

    except PathValidationError as e:
        return {"success": False, "error": f"Invalid project path: {e}"}
    except SecureSubprocessError as e:
        return {"success": False, "error": f"Failed to open project: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
