#!/usr/bin/env bash
# VoiceStudio host-side runner. The LaunchAgent launches this, not python
# directly, so a stale venv (e.g. transformers bumped and tokenizers left
# behind) is repaired before omnivoice imports and crashes again.
#
# Keep this thin: the LaunchAgent's KeepAlive handles crash loops, and
# mac-setup / make up handles venv creation. This script only fixes the
# one class of failure we have actually hit -- transformers/tokenizers
# version drift -- so a freshly cloned repo on macOS does not need manual
# `uv pip install` runs after every dependency bump.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv/bin/python"

if [[ ! -x "$VENV" ]]; then
    echo "voicestudio-runner: $VENV missing -- run \`make mac-setup\` first" >&2
    exit 1
fi

# omnivoice imports transformers at module load, which checks tokenizers.
# If the check fails, install the version transformers actually wants and
# retry. One round is enough; if it still fails, surface the real error.
if ! "$VENV" -c "import transformers, tokenizers; \
        v = tuple(int(x) for x in tokenizers.__version__.split('.')[:2]); \
        assert (0, 23) <= v < (0, 24)" 2>/dev/null; then
    echo "voicestudio-runner: tokenizers out of range, repairing..." >&2
    uv pip install --python "$VENV" --quiet 'tokenizers>=0.23.1,<0.24.0' >&2
fi

exec "$VENV" "$REPO/vendor/voice_studio/server.py"
