"""
Boundary validation system for KiCad circuit generation.

Provides comprehensive validation for component positioning, boundary checking,
and validation report generation to prevent out-of-bounds placement issues.
"""

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

from kicad_mcp.utils.component_layout import ComponentLayoutManager, SchematicBounds
from kicad_mcp.utils.coordinate_converter import CoordinateConverter, validate_position


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during boundary checking."""

    severity: ValidationSeverity
    component_ref: str
    message: str
    position: tuple[float, float]
    suggested_position: tuple[float, float] | None = None
    component_type: str = "default"


@dataclass
class ValidationReport:
    """Comprehensive validation report for circuit positioning."""

    success: bool
    issues: list[ValidationIssue]
    total_components: int
    validated_components: int
    out_of_bounds_count: int
    corrected_positions: dict[str, tuple[float, float]]

    def has_errors(self) -> bool:
        """Check if report contains any error-level issues."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if report contains any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> list[ValidationIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]


class BoundaryValidator:
    """
    Comprehensive boundary validation system for KiCad circuit generation.

    Features:
    - Pre-generation coordinate validation
    - Automatic position correction
    - Detailed validation reports
    - Integration with circuit generation pipeline
    """

    def __init__(self, bounds: SchematicBounds | None = None):
        """
        Initialize the boundary validator.

        Args:
            bounds: Schematic boundaries (defaults to A4)
        """
        self.bounds = bounds or SchematicBounds()
        self.converter = CoordinateConverter()
        self.layout_manager = ComponentLayoutManager(self.bounds)

    def validate_component_position(
        self, component_ref: str, x: float, y: float, component_type: str = "default"
    ) -> ValidationIssue:
        """
        Validate a single component position.

        Args:
            component_ref: Component reference (e.g., "R1")
            x: X coordinate in mm
            y: Y coordinate in mm
            component_type: Type of component

        Returns:
            ValidationIssue describing the validation result
        """
        # Check if position is within A4 bounds
        if not validate_position(x, y, use_margins=True):
            # Find a corrected position
            corrected_x, corrected_y = self.layout_manager.find_valid_position(
                component_ref, component_type, x, y
            )

            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                component_ref=component_ref,
                message=f"Component {component_ref} at ({x:.2f}, {y:.2f}) is outside A4 bounds",
                position=(x, y),
                suggested_position=(corrected_x, corrected_y),
                component_type=component_type,
            )

        # Check if position is within usable area (with margins)
        if not validate_position(x, y, use_margins=False):
            # Position is within absolute bounds but outside usable area
            return ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component_ref=component_ref,
                message=f"Component {component_ref} at ({x:.2f}, {y:.2f}) is outside usable area (margins)",
                position=(x, y),
                component_type=component_type,
            )

        # Position is valid
        return ValidationIssue(
            severity=ValidationSeverity.INFO,
            component_ref=component_ref,
            message=f"Component {component_ref} position is valid",
            position=(x, y),
            component_type=component_type,
        )

    def validate_circuit_components(self, components: list[dict[str, Any]]) -> ValidationReport:
        """
        Validate positioning for all components in a circuit.

        Args:
            components: List of component dictionaries with position information

        Returns:
            ValidationReport with comprehensive validation results
        """
        issues = []
        corrected_positions = {}
        out_of_bounds_count = 0

        # Reset layout manager for this validation
        self.layout_manager.clear_layout()

        for component in components:
            component_ref = component.get("reference", "Unknown")
            component_type = component.get("component_type", "default")

            # Extract position - handle different formats
            position = component.get("position")
            if position is None:
                # No position specified - this is an info issue
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        component_ref=component_ref,
                        message=f"Component {component_ref} has no position specified",
                        position=(0, 0),
                        component_type=component_type,
                    )
                )
                continue

            # Handle position as tuple or list
            if isinstance(position, list | tuple) and len(position) >= 2:
                x, y = float(position[0]), float(position[1])
            else:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        component_ref=component_ref,
                        message=f"Component {component_ref} has invalid position format: {position}",
                        position=(0, 0),
                        component_type=component_type,
                    )
                )
                continue

            # Validate the position
            validation_issue = self.validate_component_position(component_ref, x, y, component_type)
            issues.append(validation_issue)

            # Track out of bounds components
            if validation_issue.severity == ValidationSeverity.ERROR:
                out_of_bounds_count += 1
                if validation_issue.suggested_position:
                    corrected_positions[component_ref] = validation_issue.suggested_position

        # Generate report
        report = ValidationReport(
            success=out_of_bounds_count == 0,
            issues=issues,
            total_components=len(components),
            validated_components=len([c for c in components if c.get("position") is not None]),
            out_of_bounds_count=out_of_bounds_count,
            corrected_positions=corrected_positions,
        )

        return report

    def validate_wire_connection(
        self, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> list[ValidationIssue]:
        """
        Validate wire connection endpoints.

        Args:
            start_x: Starting X coordinate in mm
            start_y: Starting Y coordinate in mm
            end_x: Ending X coordinate in mm
            end_y: Ending Y coordinate in mm

        Returns:
            List of validation issues for wire endpoints
        """
        issues = []

        # Validate start point
        if not validate_position(start_x, start_y, use_margins=True):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    component_ref="WIRE_START",
                    message=f"Wire start point ({start_x:.2f}, {start_y:.2f}) is outside bounds",
                    position=(start_x, start_y),
                )
            )

        # Validate end point
        if not validate_position(end_x, end_y, use_margins=True):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    component_ref="WIRE_END",
                    message=f"Wire end point ({end_x:.2f}, {end_y:.2f}) is outside bounds",
                    position=(end_x, end_y),
                )
            )

        return issues

    def auto_correct_positions(
        self, components: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], ValidationReport]:
        """
        Automatically correct out-of-bounds component positions.

        Args:
            components: List of component dictionaries

        Returns:
            Tuple of (corrected_components, validation_report)
        """
        # First validate to get correction suggestions
        validation_report = self.validate_circuit_components(components)

        # Apply corrections
        corrected_components = []
        for component in components:
            component_ref = component.get("reference", "Unknown")

            if component_ref in validation_report.corrected_positions:
                # Apply correction
                corrected_component = component.copy()
                corrected_component["position"] = validation_report.corrected_positions[
                    component_ref
                ]
                corrected_components.append(corrected_component)
            else:
                corrected_components.append(component)

        return corrected_components, validation_report

    def generate_validation_report_text(self, report: ValidationReport) -> str:
        """
        Generate a human-readable validation report.

        Args:
            report: ValidationReport to format

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BOUNDARY VALIDATION REPORT")
        lines.append("=" * 60)

        # Summary
        lines.append(f"Status: {'PASS' if report.success else 'FAIL'}")
        lines.append(f"Total Components: {report.total_components}")
        lines.append(f"Validated Components: {report.validated_components}")
        lines.append(f"Out of Bounds: {report.out_of_bounds_count}")
        lines.append(f"Corrected Positions: {len(report.corrected_positions)}")
        lines.append("")

        # Issues by severity
        errors = report.get_issues_by_severity(ValidationSeverity.ERROR)
        warnings = report.get_issues_by_severity(ValidationSeverity.WARNING)
        info = report.get_issues_by_severity(ValidationSeverity.INFO)

        if errors:
            lines.append("ERRORS:")
            for issue in errors:
                lines.append(f"  ❌ {issue.message}")
                if issue.suggested_position:
                    lines.append(f"     → Suggested: {issue.suggested_position}")
            lines.append("")

        if warnings:
            lines.append("WARNINGS:")
            for issue in warnings:
                lines.append(f"  ⚠️  {issue.message}")
            lines.append("")

        if info:
            lines.append("INFO:")
            for issue in info:
                lines.append(f"  ℹ️  {issue.message}")
            lines.append("")

        # Corrected positions
        if report.corrected_positions:
            lines.append("CORRECTED POSITIONS:")
            for component_ref, (x, y) in report.corrected_positions.items():
                lines.append(f"  {component_ref}: ({x:.2f}, {y:.2f})")

        return "\n".join(lines)

    def export_validation_report(self, report: ValidationReport, filepath: str) -> None:
        """
        Export validation report to JSON file.

        Args:
            report: ValidationReport to export
            filepath: Path to output file
        """
        # Convert report to serializable format
        export_data = {
            "success": report.success,
            "total_components": report.total_components,
            "validated_components": report.validated_components,
            "out_of_bounds_count": report.out_of_bounds_count,
            "corrected_positions": report.corrected_positions,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "component_ref": issue.component_ref,
                    "message": issue.message,
                    "position": issue.position,
                    "suggested_position": issue.suggested_position,
                    "component_type": issue.component_type,
                }
                for issue in report.issues
            ],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
