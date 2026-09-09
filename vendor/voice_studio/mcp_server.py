"""
Model Context Protocol (MCP) server for VoiceStudio.

Exposes exactly two tools:
- create_profile(audio_path: str, profile_name: str) -> str
- generate_audio(text: str, profile_name: str, output_filename: str) -> str
"""

from __future__ import annotations

import json
import logging
import sys

import core

logger = logging.getLogger("voicestudio.mcp")

TOOLS = [
    {
        "name": "create_profile",
        "description": "Extracts and saves a target voice profile locally.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": "File path to the reference audio file",
                },
                "profile_name": {
                    "type": "string",
                    "description": "Name identifier of the voice profile to create",
                },
            },
            "required": ["audio_path", "profile_name"],
        },
    },
    {
        "name": "generate_audio",
        "description": "Synthesizes speech to an audio file using a saved voice profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text string to synthesize into speech",
                },
                "profile_name": {
                    "type": "string",
                    "description": "Name of the saved voice profile to clone",
                },
                "output_filename": {
                    "type": "string",
                    "description": "Output audio filename (saved in ./output/)",
                },
            },
            "required": ["text", "profile_name", "output_filename"],
        },
    },
]


def create_mcp_server():
    """Build and return the FastMCP server instance if mcp SDK is installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None

    mcp = FastMCP(
        "VoiceStudio",
        instructions=(
            "Minimal headless VoiceStudio MCP server for voice cloning and speech synthesis. "
            "Exposes two tools: create_profile and generate_audio."
        ),
    )

    @mcp.tool()
    def create_profile(audio_path: str, profile_name: str) -> str:
        """Extracts and saves a target voice profile locally.

        Args:
            audio_path: Path to the reference audio file.
            profile_name: Name of the voice profile to create.

        Returns:
            Absolute path to the saved voice profile.
        """
        return core.create_profile(audio_path=audio_path, profile_name=profile_name)

    @mcp.tool()
    def generate_audio(text: str, profile_name: str, output_filename: str) -> str:
        """Synthesizes speech to an audio file using a saved voice profile.

        Args:
            text: Text to synthesize into speech.
            profile_name: Name of the saved voice profile to clone.
            output_filename: Filename for the output audio (saved in ./output/).

        Returns:
            Absolute path to the generated audio file.
        """
        return core.generate_audio(
            text=text, profile_name=profile_name, output_filename=output_filename
        )

    return mcp


def run_stdio_mcp_fallback():
    """Standard JSON-RPC 2.0 stdio MCP loop for environments without mcp SDK."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "VoiceStudio", "version": "0.5.2"},
                },
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments") or {}
            try:
                if tool_name == "create_profile":
                    out = core.create_profile(
                        audio_path=args["audio_path"],
                        profile_name=args["profile_name"],
                    )
                elif tool_name == "generate_audio":
                    out = core.generate_audio(
                        text=args["text"],
                        profile_name=args["profile_name"],
                        output_filename=args["output_filename"],
                    )
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(out)}],
                        "isError": False,
                    },
                }
            except Exception as err:
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {err}"}],
                        "isError": True,
                    },
                }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        elif req_id is not None:
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"},
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()


def main():
    mcp_app = create_mcp_server()
    if mcp_app is not None:
        mcp_app.run()
    else:
        run_stdio_mcp_fallback()


if __name__ == "__main__":
    main()
