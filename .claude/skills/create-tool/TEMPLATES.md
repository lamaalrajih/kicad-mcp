# Tool Templates

Skeleton templates for kicad-mcp components. See `/.local/STYLE_GUIDE.md` for detailed patterns and common mistakes.

## Tool Skeleton (Async with Progress)

Use this for tools that need progress reporting:

```python
"""Brief description of this tool category."""
from typing import Any

from mcp.server.fastmcp import FastMCP, Context


def register_<category>_tools(mcp: FastMCP) -> None:
    """Register <category> tools with MCP server."""

    @mcp.tool()
    async def <tool_name>(
        <required_param>: str,
        <optional_param>: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """<Tool description>.

        Args:
            <required_param>: Description of required param
            <optional_param>: Description of optional param
            ctx: MCP context for progress reporting

        Returns:
            Dictionary with results
        """
        print(f"Processing: {<required_param>}")

        if ctx:
            await ctx.report_progress(10, 100)

        # Validate inputs
        if not <required_param>:
            return {"success": False, "error": "Missing required parameter"}

        if ctx:
            await ctx.report_progress(30, 100)

        # Implementation here
        result = do_something(<required_param>)

        if ctx:
            await ctx.report_progress(90, 100)

        return {
            "success": True,
            "data": result,
            "message": "Operation completed successfully",
        }
```

## Tool Skeleton (Sync - No Progress)

Use this for simple tools that don't need progress reporting:

```python
@mcp.tool()
def <tool_name>(<param>: str) -> dict[str, Any]:
    """<Tool description>."""
    print(f"Processing: {<param>}")

    if not <param>:
        return {"success": False, "error": "Missing parameter"}

    result = do_something(<param>)
    return {"success": True, "data": result}
```

## Test Skeleton

```python
"""Tests for <category> tools."""
from unittest.mock import MagicMock, patch

import pytest


class Test<ToolName>:
    """Tests for <tool_name> tool."""

    def test_success(self):
        """Test successful operation."""
        result = <tool_name>("valid_input")
        assert result["success"] is True
        assert "data" in result

    def test_missing_input(self):
        """Test error handling for missing input."""
        result = <tool_name>("")
        assert result["success"] is False
        assert "error" in result

    @patch("kicad_mcp.tools.<category>_tools.<dependency>")
    def test_with_mock(self, mock_dep):
        """Test with mocked dependency."""
        mock_dep.return_value = "mocked_value"
        result = <tool_name>("input")
        assert result["success"] is True
        mock_dep.assert_called_once()

    def test_invalid_path(self):
        """Test error handling for invalid file path."""
        result = <tool_name>("/nonexistent/path")
        assert result["success"] is False
        assert "error" in result
```

## Resource Skeleton (Optional)

Use if the tool should expose data via `kicad://` URI:

```python
@mcp.resource("kicad://<resource-type>/{parameter}")
def get_<resource>(parameter: str) -> str:
    """<Resource description>.

    Args:
        parameter: The resource parameter

    Returns:
        Formatted markdown string for LLM consumption
    """
    data = fetch_data(parameter)

    # Format as markdown for LLM
    output = f"# <Resource Title>\n\n"
    output += f"**Parameter**: {parameter}\n\n"
    output += f"## Data\n\n{data}\n"

    return output
```

## Prompt Skeleton (Optional)

Use if the tool benefits from conversation templates:

```python
@mcp.prompt()
def <prompt_name>() -> str:
    """<Prompt description>."""
    return """
    <Multi-line prompt text that serves as a conversation starter.>

    ## Context
    <What information the user should provide>

    ## Expected Output
    <What the tool will produce>
    """
```

## Return Value Patterns

```python
# Success with data
{"success": True, "data": result, "count": len(items)}

# Success with message
{"success": True, "message": "Operation completed"}

# Error with details
{"success": False, "error": "Description of what went wrong"}

# Error with suggestions
{"success": False, "error": "File not found", "suggestions": ["Check path", "Run list_projects first"]}

# Status-based (for caches)
{"success": False, "cache_status": "stale", "message": "Run rebuild_index first"}
```
