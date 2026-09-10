# VoiceStudio headless TTS server. Lives in its own image so MoneyPrinterTurbo
# doesn't pull torch/transformers/accelerate (~5GB) into the app image.
#
# Build from the repo root with the vendor/voice_studio/ directory as the
# context, e.g. in docker-compose.yml:
#     build:
#       context: ./vendor/voice_studio
#
# CPU-only is the default. For NVIDIA hosts, set PYTORCH_INDEX_URL to
# the appropriate CUDA wheel index and add the
# `deploy.resources.reservations.devices` block in compose.

FROM python:3.11-slim-bookworm

# Keep the working directory stable so voice_profiles/ and output/ land at
# /voicestudio/{voice_profiles,output} and the named volume mount works as
# expected. server.py and core.py resolve PROFILES_DIR from BASE_DIR.
WORKDIR /voicestudio

# ffmpeg is required by core.py's audio decoding paths (pydub/soundfile).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps. requirements.txt includes fastapi + uvicorn so the
# HTTP server is reachable inside the container without an extra layer.
COPY requirements.txt ./
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 3 --timeout 120 \
    torch torchaudio --index-url ${PYTORCH_INDEX_URL}
RUN pip install --no-cache-dir --retries 3 --timeout 120 -r requirements.txt
RUN HF_HUB_OFFLINE=1 python -c "from omnivoice.models.omnivoice import OmniVoice"

# Copy the rest of the VoiceStudio package. Build context is
# vendor/voice_studio/, so .dockerignore keeps profiles/output/.venv out of
# the image and the heavy .pt model files do not bloat the layer.
COPY . ./
RUN HF_HUB_OFFLINE=1 python -c "from server import VOICE_PRESETS; from omnivoice.models.omnivoice import _resolve_instruct; [_resolve_instruct(value) for value in VOICE_PRESETS.values()]"

# Persist generated voice profiles + audio outputs across container restarts
# via the named volumes declared in docker-compose.yml.
RUN mkdir -p /voicestudio/voice_profiles /voicestudio/output

# Bind to all interfaces inside the container so Docker port mapping works.
# Port is overridable at runtime via VOICESTUDIO_PORT.
ENV VOICESTUDIO_HOST=0.0.0.0
ENV VOICESTUDIO_PORT=8780

EXPOSE 8780

# A small Python healthcheck matches /health in server.py. The compose
# service waits for this to flip healthy before starting MoneyPrinterTurbo
# so the first WebUI TTS request never hits a still-booting model server.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8780/health', timeout=2).status == 200 else 1)"

CMD ["python", "server.py"]
