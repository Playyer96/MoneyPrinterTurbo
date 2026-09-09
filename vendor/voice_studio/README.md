# VoiceStudio (Headless CLI + MCP Server)

A minimal, headless voice cloning and speech synthesis Python package powered by OmniVoice, exposing a clean Command Line Interface (CLI) and Model Context Protocol (MCP) server for AI agents.

## Features
- **Headless & Minimal**: Zero Web UI, frontend dependencies, or visual previews.
- **Dual Tool Interface**: Exactly two operations:
  - `create_profile(audio_path: str, profile_name: str) -> str`: Extracts and saves target voice profile locally.
  - `generate_audio(text: str, profile_name: str, output_filename: str) -> str`: Synthesizes speech to an audio file.
- **Local Storage**:
  - Saved voice profiles: `./voice_profiles/`
  - Generated audio outputs: `./output/` (returns absolute path on completion)

---

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Command Line Interface (CLI)

### 1. Create a Voice Profile
Extract and save a target voice profile locally:
```bash
python cli.py create --audio <path_to_audio> --name <profile_name>
```

Example:
```bash
python cli.py create --audio samples/speaker.wav --name narrator
```
The voice profile is stored in `./voice_profiles/narrator.pt` (and `./voice_profiles/narrator.json`).

### 2. Generate Audio
Synthesize speech audio from text using a saved voice profile:
```bash
python cli.py generate --text "<string>" --profile <profile_name> --out <filename>
```

Example:
```bash
python cli.py generate --text "Welcome to the minimal headless VoiceStudio." --profile narrator --out welcome.wav
```
The synthesized audio file will be saved in `./output/welcome.wav` and the absolute path will be printed.

---

## Model Context Protocol (MCP) Server

The package provides a standard Model Context Protocol (MCP) server exposing the two tools directly to AI agents (Claude Desktop, Cursor, Antigravity, etc.).

### Run Standalone
```bash
python mcp_server.py
```

### Claude Desktop Configuration
Add the server configuration to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "voicestudio": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/voice_studio"
    }
  }
}
```

### Exposed MCP Tools
1. **`create_profile`**:
   - `audio_path` (string, required): Path to reference audio file.
   - `profile_name` (string, required): Profile name identifier.
   - Returns: Absolute path to saved profile.

2. **`generate_audio`**:
   - `text` (string, required): Text string to synthesize.
   - `profile_name` (string, required): Name of saved voice profile.
   - `output_filename` (string, required): Output audio filename (saved under `./output/`).
   - Returns: Absolute path to generated audio file.

---

## Python API Usage

You can also import and use `core.py` directly in Python applications:

```python
import core

# Create a voice profile
profile_path = core.create_profile(audio_path="sample.wav", profile_name="alice")
print(f"Profile saved to: {profile_path}")

# Generate audio
audio_path = core.generate_audio(
    text="Hello from the Python API!",
    profile_name="alice",
    output_filename="alice_output.wav"
)
print(f"Audio generated at: {audio_path}")
```

---

## Running Tests
```bash
python -m unittest tests/test_headless.py
```
