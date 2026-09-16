from __future__ import annotations

from pathlib import Path

from app.config import config
from app.services.voice import omnivoice


class _Response:
    status_code = 200
    content = b"R" * 1024
    text = ""


class _AudioClip:
    duration = 1.5

    def __init__(self, _path: str) -> None:
        pass

    def close(self) -> None:
        pass


def test_omnivoice_client_sends_style_and_speed(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.setattr(config, "omnivoice", {"base_url": "http://omnivoice", "style": "energetic"})
    monkeypatch.setattr(omnivoice, "AudioFileClip", _AudioClip)
    monkeypatch.setattr(omnivoice, "_apply_volume", lambda *_args: None)

    def post(url, *, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr(omnivoice.requests, "post", post)
    output = tmp_path / "voice.wav"

    result = omnivoice.omnivoice_tts(
        "A real sentence.", "narrator", str(output), voice_rate=1.2
    )

    assert result is not None
    assert output.read_bytes() == _Response.content
    assert captured == {
        "url": "http://omnivoice/generate",
        "payload": {
            "text": "A real sentence.",
            "voice": "narrator",
            "style": "energetic",
            "speed": 1.2,
        },
        "timeout": 1800,
    }
