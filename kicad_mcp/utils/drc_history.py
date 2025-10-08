"""
Utilities for tracking DRC history for KiCad projects.

This will allow users to compare DRC results over time.
"""

import os
import json
import platform
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Directory for storing DRC history
if platform.system() == "Windows":
    # Windows: Use APPDATA or LocalAppData
    DRC_HISTORY_DIR = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")), "kicad_mcp", "drc_history"
    )
else:
    # macOS/Linux: Use ~/.kicad_mcp/drc_history
    DRC_HISTORY_DIR = os.path.expanduser("~/.kicad_mcp/drc_history")


def ensure_history_dir() -> None:
    """Ensure the DRC history directory exists."""
    os.makedirs(DRC_HISTORY_DIR, exist_ok=True)


def get_project_history_path(project_path: str) -> str:
    """Get the path to the DRC history file for a project.

    Args:
        project_path: Path to the KiCad project file

    Returns:
        Path to the project's DRC history file
    """
    # Create a safe filename from the project path
    project_hash = hash(project_path) & 0xFFFFFFFF  # Ensure positive hash
    basename = os.path.basename(project_path)
    history_filename = f"{basename}_{project_hash}_drc_history.json"

    return os.path.join(DRC_HISTORY_DIR, history_filename)


def save_drc_result(project_path: str, drc_result: Dict[str, Any]) -> None:
    """Save a DRC result to the project's history.

    Args:
        project_path: Path to the KiCad project file
        drc_result: DRC result dictionary
    """
    ensure_history_dir()
    history_path = get_project_history_path(project_path)

    # Create a history entry
    timestamp = time.time()
    formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    history_entry = {
        "timestamp": timestamp,
        "datetime": formatted_time,
        "total_violations": drc_result.get("total_violations", 0),
        "violation_categories": drc_result.get("violation_categories", {}),
    }

    # Load existing history or create new
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading DRC history: {str(e)}")
            history = {"project_path": project_path, "entries": []}
    else:
        history = {"project_path": project_path, "entries": []}

    # Add new entry and save
    history["entries"].append(history_entry)

    # Keep only the last 10 entries to avoid excessive storage
    if len(history["entries"]) > 10:
        history["entries"] = sorted(history["entries"], key=lambda x: x["timestamp"], reverse=True)[
            :10
        ]

    try:
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved DRC history entry to {history_path}")
    except IOError as e:
        print(f"Error saving DRC history: {str(e)}")


def get_drc_history(project_path: str) -> List[Dict[str, Any]]:
    """Get the DRC history for a project.

    Args:
        project_path: Path to the KiCad project file

    Returns:
        List of DRC history entries, sorted by timestamp (newest first)
    """
    history_path = get_project_history_path(project_path)

    if not os.path.exists(history_path):
        print(f"No DRC history found for {project_path}")
        return []

    try:
        with open(history_path, "r") as f:
            history = json.load(f)

        # Sort entries by timestamp (newest first)
        entries = sorted(
            history.get("entries", []), key=lambda x: x.get("timestamp", 0), reverse=True
        )

        return entries
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading DRC history: {str(e)}")
        return []


def compare_with_previous(
    project_path: str, current_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Compare current DRC result with the previous one.

    Args:
        project_path: Path to the KiCad project file
        current_result: Current DRC result dictionary

    Returns:
        Comparison dictionary or None if no history exists
    """
    history = get_drc_history(project_path)

    if not history or len(history) < 2:  # Need at least one previous entry
        return None

    previous = history[0]  # Most recent entry
    current_violations = current_result.get("total_violations", 0)
    previous_violations = previous.get("total_violations", 0)

    # Compare violation categories
    current_categories = current_result.get("violation_categories", {})
    previous_categories = previous.get("violation_categories", {})

    # Find new categories
    new_categories = {}
    for category, count in current_categories.items():
        if category not in previous_categories:
            new_categories[category] = count

    # Find resolved categories
    resolved_categories = {}
    for category, count in previous_categories.items():
        if category not in current_categories:
            resolved_categories[category] = count

    # Find changed categories
    changed_categories = {}
    for category, count in current_categories.items():
        if category in previous_categories and count != previous_categories[category]:
            changed_categories[category] = {
                "current": count,
                "previous": previous_categories[category],
                "change": count - previous_categories[category],
            }

    comparison = {
        "current_violations": current_violations,
        "previous_violations": previous_violations,
        "change": current_violations - previous_violations,
        "previous_datetime": previous.get("datetime", "unknown"),
        "new_categories": new_categories,
        "resolved_categories": resolved_categories,
        "changed_categories": changed_categories,
    }

    return comparison
