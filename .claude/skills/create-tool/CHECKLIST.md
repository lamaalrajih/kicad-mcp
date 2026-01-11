# Tool Registration Checklist

Follow this checklist after generating tool code to ensure proper integration.

## Registration Steps

### 1. Import in `kicad_mcp/server.py`

Add the import at the top with other tool imports:

```python
from kicad_mcp.tools.<category>_tools import register_<category>_tools
```

### 2. Register in `create_server()`

Add the registration call inside the `create_server()` function:

```python
def create_server() -> FastMCP:
    mcp = FastMCP("KiCad", lifespan=lifespan_factory)

    # ... other registrations ...

    # Register your new tools
    register_<category>_tools(mcp)

    return mcp
```

### 3. Add Dependencies (if needed)

If your tool requires new packages, add them to `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "new-package>=1.0.0",
]
```

Then run:
```bash
uv sync
```

### 4. Create Tests

Create test file at `tests/unit/tools/test_<category>_tools.py` with:
- Success case tests
- Error handling tests
- Edge case tests

## Validation Commands

Run these commands to verify your tool:

```bash
# Lint and format (fixes issues automatically)
uv run ruff check . && uv run ruff format .

# Type check
uv run mypy kicad_mcp/

# Run unit tests
uv run pytest tests/unit/ -v

# Run all checks at once
uv run ruff check . && uv run ruff format . && uv run mypy kicad_mcp/ && uv run pytest tests/unit/ -v
```

## Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong import | `ModuleNotFoundError` | Use `from mcp.server.fastmcp import ...` |
| Old type hints | `TypeError` or mypy error | Use `dict[str, Any]` not `Dict`, `str \| None` not `Optional` |
| Missing ctx default | `TypeError: missing argument` | Ensure `ctx: Context \| None = None` has `= None` |
| Await on ctx.info | Runtime warning | `ctx.info()` needs NO await, `ctx.report_progress()` NEEDS await |
| Missing encoding | `UnicodeDecodeError` | Add `encoding="utf-8"` to all `open()` calls |
| Missing registration | Tool not appearing | Add both import AND register call in server.py |
| Helper inside register | Closure issues | Move helper functions outside `register_*`, prefix with `_` |
| Missing success key | Inconsistent API | Always include `"success": True/False` in return dict |
| Regex in loop | Performance issue | Move `re.compile()` to module level as `_PATTERN` |

## File Locations Reference

| Component | Location |
|-----------|----------|
| Tools | `kicad_mcp/tools/<category>_tools.py` |
| Resources | `kicad_mcp/resources/<category>.py` |
| Prompts | `kicad_mcp/prompts/<category>_prompts.py` |
| Utilities | `kicad_mcp/utils/<helper>.py` |
| Tests | `tests/unit/tools/test_<category>_tools.py` |
| Server | `kicad_mcp/server.py` |
| Config | `kicad_mcp/config.py` |
| Dependencies | `pyproject.toml` |

## Quick Verification

After completing all steps, verify:

1. [ ] Tool appears in MCP tool list
2. [ ] Tool executes without errors
3. [ ] All tests pass
4. [ ] No linting errors
5. [ ] No type errors
