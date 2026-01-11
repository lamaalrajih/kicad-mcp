# Tool Style Guide

Patterns for `kicad_mcp/tools/*.py`. Run `uv run ruff check . && uv run ruff format .` to catch most issues.

---

## Tool Template

```python
"""Brief description."""
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP, Context
from kicad_mcp.config import ...

# Pre-compile regex at module level for performance
_PATTERN = re.compile(r'...')


def _helper() -> str:
    """Helpers above register function, underscore-prefixed."""
    pass


def register_category_tools(mcp: FastMCP) -> None:
    """Register tools with MCP server."""

    @mcp.tool()
    def sync_tool(param: str) -> dict[str, Any]:
        """Sync - no progress reporting."""
        print(f"Processing: {param}")
        return {"success": True, "data": param}

    @mcp.tool()
    async def async_tool(param: str, ctx: Context | None = None) -> dict[str, Any]:
        """Async - use for progress reporting."""
        print(f"Processing: {param}")
        if ctx:
            await ctx.report_progress(50, 100)  # AWAIT this
            ctx.info("Status message")           # NO await
        return {"success": True}
```

---

## Quick Rules

| Pattern | Correct | Wrong |
|---------|---------|-------|
| Import | `from mcp.server.fastmcp import ...` | `from fastmcp import ...` |
| Types | `dict[str, Any]`, `str \| None` | `Dict`, `Optional` |
| Context param | `ctx: Context \| None = None` | `ctx: Context \| None` (no default) |
| Progress | `await ctx.report_progress(...)` | `await ctx.info(...)` |
| File open | `open(f, encoding="utf-8")` | `open(f, "r", ...)` |

---

## Return Values

```python
# Success
{"success": True, "data": result, "count": len(items)}

# Error with recovery hints
{"success": False, "error": "Description", "suggestions": ["alt1", "alt2"]}

# Status-based (for caches)
{"success": False, "cache_status": "stale", "message": "Run rebuild_index."}
```

---

## Logging

**Tools**: Use `print()` for visibility
```python
print(f"Processing: {path}")
print(f"Found {len(items)} items")
```

**Utilities** (`utils/*.py`): Use logging module
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing...")
logger.error(f"Failed: {e}")
```

---

## Utility Patterns (`utils/*.py`)

### Parallel I/O
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_file(path: str) -> list[dict]:
    """Worker must handle its own errors."""
    try:
        with open(path, encoding="utf-8") as f:
            return parse(f.read())
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return []  # Don't crash the batch

async def build_index(files: list[str], ctx=None) -> dict[str, Any]:
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            results.extend(future.result())
            if ctx and i % 20 == 0:
                await ctx.report_progress(int(i / len(files) * 100), 100)
    return {"items": results}
```

### File Locking (for shared caches)
```python
from filelock import FileLock, Timeout

def save_cache(data: dict) -> bool:
    lock = FileLock(cache_path + ".lock", timeout=300)
    try:
        with lock:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        return True
    except Timeout:
        logger.error("Timeout waiting for lock")
        return False
```

---

## Testing

Create tests in `tests/unit/tools/test_<tool>.py`:

```python
from unittest.mock import patch, MagicMock
import pytest

class TestMyFunction:
    def test_success(self):
        result = my_function("valid")
        assert result["success"] is True

    @patch("kicad_mcp.tools.my_tools.get_path")
    def test_with_mock(self, mock_path):
        mock_path.return_value = "/fake/path"
        result = my_function("input")
        assert result["success"] is True

    def test_error(self):
        result = my_function("")
        assert result["success"] is False
        assert "error" in result
```

---

## Registration Checklist

1. **Import** in `server.py`:
   ```python
   from kicad_mcp.tools.category_tools import register_category_tools
   ```

2. **Register** in `create_server()`:
   ```python
   register_category_tools(mcp)
   ```

3. **Dependencies** in `pyproject.toml` (if needed):
   ```toml
   "filelock>=3.0.0",
   ```

4. **Tests** in `tests/unit/tools/test_category_tools.py`

5. **Verify**:
   ```bash
   uv run ruff check . && uv run ruff format . && uv run mypy kicad_mcp/ && uv run pytest tests/unit/ -v
   ```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `from fastmcp import ...` | `from mcp.server.fastmcp import ...` |
| `Dict[str, Any]` | `dict[str, Any]` |
| `Optional[str]` | `str \| None` |
| `ctx: Context \| None` (no default) | `ctx: Context \| None = None` |
| `await ctx.info(...)` | `ctx.info(...)` (no await) |
| `open(f, "r", ...)` | `open(f, ...)` (omit "r") |
| Missing `encoding="utf-8"` | Always include |
| Forgot server.py registration | Add import + register call |
| Helper inside register function | Move outside, prefix `_` |
| Missing `success` in return | Always include |
| No try/except in thread workers | Wrap, return `[]` on error |
| Regex compiled in loop | Move to module-level `_PATTERN` |

---

*Updated: 2025-01-10*
