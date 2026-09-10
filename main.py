import uvicorn
from loguru import logger

from app.config import config
from app.services import voice

if __name__ == "__main__":
    # Auto-launch the bundled VoiceStudio (OmniVoice) server when its port is
    # unreachable, so API users get the same out-of-the-box experience as the
    # WebUI. The helper is idempotent: externally-managed servers stay untouched.
    voice.ensure_voicestudio_server_running()
    logger.info(
        "start server, docs: http://127.0.0.1:" + str(config.listen_port) + "/docs"
    )
    # ffmpeg probing now lives in the shared task pipeline at app/services/task.py,
    # so the API, CLI, and WebUI all exercise the same path; no separate check here.
    uvicorn.run(
        app="app.asgi:app",
        host=config.listen_host,
        port=config.listen_port,
        reload=config.reload_debug,
        log_level="warning",
    )
