"""Run from the configured project root with the MCP environment's Python."""
import argparse
import asyncio
import time
import tomllib
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def verify(config_path):
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    server = config["mcp_servers"]["bovary"]
    params = StdioServerParameters(command=server["command"], args=server["args"], env=server["env"])
    async with stdio_client(params) as (reader, writer), ClientSession(reader, writer) as client:
        await client.initialize()
        tools = await client.list_tools()
        if not any(tool.name == "ask_bovary" for tool in tools.tools):
            raise RuntimeError("The configured server does not expose ask_bovary.")
        invalid = await client.call_tool("ask_bovary", {"prompt": "Hello", "max_tokens": 257})
        if not invalid.isError:
            raise RuntimeError("The server accepted an out-of-range token limit.")
        invalid = await client.call_tool("ask_bovary", {"prompt": "Hello", "model": "unknown"})
        if not invalid.isError:
            raise RuntimeError("The server accepted an unknown model.")
        models = []
        if server["env"].get("BOVARY_MODEL"):
            models.append("gemma")
        if server["env"].get("BOVARY_INKLING_RECEIPT"):
            models.append("inkling")
        if not models:
            raise ValueError("Configure a Gemma model or Inkling receipt before the smoke test.")
        for model in models:
            started = time.monotonic()
            args = {"prompt": "What is 24 plus 37? Reply with just the number.", "max_tokens": 24}
            if model == "inkling":
                args["model"] = model
            result = await client.call_tool("ask_bovary", args)
            print(model, result.content, flush=True)
            if result.isError or not any(getattr(item, "text", "").strip() == "61" for item in result.content):
                raise RuntimeError(f"{model} failed the arithmetic check. See its response above.")
            print(f"Verified {model} in {time.monotonic() - started:.1f} seconds.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".codex/config.toml"))
    asyncio.run(asyncio.wait_for(verify(parser.parse_args().config), timeout=420))
