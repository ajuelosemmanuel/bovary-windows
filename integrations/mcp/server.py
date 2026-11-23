"""Serve local Gemma CPU and optional Inkling CUDA inference over MCP stdio."""

from pathlib import Path
import sys
from typing import Annotated, Literal

import anyio
from bovary.mini import MiniAgent
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field


server = FastMCP("Bovary")
agent = MiniAgent(preset="cpu-only")
inference_lock = anyio.Lock()


def ask_inkling(prompt: str, max_tokens: int) -> str:
    # The same scripts run beside the Windows launcher in the standalone ZIP.
    preview = str(Path(__file__).resolve().parents[2] / "previews" / "inkling")
    if preview not in sys.path:
        sys.path.insert(0, preview)
    from ask import ask

    return ask(prompt, max_tokens)


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def ask_bovary(
    prompt: Annotated[str, Field(min_length=1, max_length=8192)],
    max_tokens: Annotated[int, Field(ge=1, le=256)] = 96,
    model: Literal["gemma", "inkling"] = "gemma",
) -> str:
    """Ask Gemma 26B (CPU, default) or Inkling 2 Mini (CUDA preview).

    Input is limited to 8192 UTF-8 bytes. The reply is model-generated text.
    """
    async with inference_lock:
        if model == "inkling":
            return await anyio.to_thread.run_sync(ask_inkling, prompt, max_tokens)
        result = await anyio.to_thread.run_sync(lambda: agent.run(prompt, tokens=max_tokens))
    if result["status"] != "complete":
        raise ValueError(f"Bovary returned {result['status']}. Try a shorter task.")
    return result["text"]


if __name__ == "__main__":
    server.run(transport="stdio")
