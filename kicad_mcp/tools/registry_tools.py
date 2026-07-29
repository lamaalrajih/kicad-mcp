"""
Parts registry tools — search and fetch verified KiCad parts from an open,
no-login registry (default: PartReel, https://partreel.com).

Instead of hand-drawing a footprint (or hallucinating one), an assistant can
search the registry, inspect a part's provenance and field reports, and
download the .kicad_mod / .kicad_sym / STEP files straight into the project.
"""

import asyncio
import os
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from kicad_mcp.utils.parts_registry import (
    DEFAULT_REGISTRY_URL,
    FORMAT_EXTENSIONS,
    RegistryClient,
    RegistryError,
)
from kicad_mcp.utils.path_validator import PathValidator, PathValidationError


def register_registry_tools(mcp: FastMCP) -> None:
    """Register parts registry tools with the MCP server.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.tool()
    async def search_parts_registry(
        query: str, limit: int = 20, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Search an open registry of verified KiCad parts (no login needed).

        Matches part id, name, family, category, manufacturer and keywords.
        Example queries: "usb c connector", "0402 capacitor", "RKJXM".

        Args:
            query: Search terms (all terms must match; '*' wildcards allowed)
            limit: Maximum number of results (default 20)
            ctx: MCP context for progress reporting

        Returns:
            Dictionary with matching parts (id, name, family, page URL)
        """
        client = RegistryClient()
        try:
            hits = await asyncio.to_thread(client.search, query, limit)
        except RegistryError as exc:
            return {"success": False, "error": str(exc)}
        if ctx:
            ctx.info(f"Registry search '{query}': {len(hits)} hits")
        return {
            "success": True,
            "registry": DEFAULT_REGISTRY_URL,
            "count": len(hits),
            "parts": [
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "family": p.get("family"),
                    "category": p.get("category"),
                    "manufacturer": p.get("manufacturer"),
                    "page": p.get("page"),
                }
                for p in hits
            ],
        }

    @mcp.tool()
    async def get_registry_part(
        part_id: str, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Get the full record for one registry part.

        Includes download URLs for the footprint/symbol/3D files, the
        datasheet link, dimension provenance (which datasheet figure each
        dimension came from) and field reports from users who built real
        boards with the part.

        Args:
            part_id: Registry part id (from search_parts_registry)
            ctx: MCP context for progress reporting

        Returns:
            Dictionary with the part record, or an error
        """
        client = RegistryClient()
        try:
            part = await asyncio.to_thread(client.get_part, part_id)
        except RegistryError as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "part": part}

    @mcp.tool()
    async def download_registry_part(
        part_id: str,
        save_dir: str,
        formats: list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Download a registry part's files into a local directory.

        Downloads are restricted to the registry's own hosts, capped at 50 MB
        per file, and saved with extension-checked filenames.

        Args:
            part_id: Registry part id (from search_parts_registry)
            save_dir: Existing local directory to save into (e.g. the
                project's footprint library folder)
            formats: Which files to fetch — any of "footprint", "symbol",
                "step", "preview" (default: footprint + symbol)
            ctx: MCP context for progress reporting

        Returns:
            Dictionary with the saved file paths
        """
        formats = formats or ["footprint", "symbol"]
        unknown = [f for f in formats if f not in FORMAT_EXTENSIONS]
        if unknown:
            return {"success": False, "error": f"unknown formats: {unknown}"}

        # Writes are confined to the working dir or the user's home tree
        # (extra roots via PARTS_REGISTRY_SAVE_ROOTS, comma-separated).
        roots = {os.getcwd(), os.path.expanduser("~")}
        roots.update(
            r.strip()
            for r in os.environ.get("PARTS_REGISTRY_SAVE_ROOTS", "").split(",")
            if r.strip()
        )
        try:
            dest = PathValidator(trusted_roots=roots).validate_directory(save_dir)
        except PathValidationError as exc:
            return {"success": False, "error": f"invalid save_dir: {exc}"}

        client = RegistryClient()
        try:
            part = await asyncio.to_thread(client.get_part, part_id)
        except RegistryError as exc:
            return {"success": False, "error": str(exc)}

        files = part.get("files") or {}
        saved, missing = [], []
        for fmt in formats:
            url = files.get(fmt)
            if not url:
                missing.append(fmt)
                continue
            if ctx:
                ctx.info(f"Downloading {fmt} for {part_id}")
            try:
                path = await asyncio.to_thread(
                    client.download_asset, url, dest, part_id, fmt
                )
                saved.append(path)
            except RegistryError as exc:
                return {"success": False, "error": str(exc), "saved": saved}
        return {
            "success": True,
            "saved": saved,
            "not_available": missing,
            "license": part.get("license"),
            "attribution": part.get("provenance", {}).get("attribution")
            if isinstance(part.get("provenance"), dict)
            else None,
        }
