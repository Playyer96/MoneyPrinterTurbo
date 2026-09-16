from pathlib import Path
from unittest.mock import patch

from app.config import config
from app.models.llm_provider import get_llm_provider
from app.services import voice


def test_azure_has_no_runtime_provider_or_dependency():
    root = Path(__file__).resolve().parents[2]

    assert get_llm_provider("azure") is None
    assert not hasattr(voice, "azure_tts_v1")
    assert not hasattr(voice, "azure_tts_v2")
    assert "azure-cognitiveservices-speech" not in (root / "pyproject.toml").read_text()


def test_legacy_azure_configuration_is_discarded(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[app]\nazure_api_key = "secret"\n\n[azure]\nspeech_key = "secret"\n'
    )

    with patch.object(config, "config_file", str(config_file)):
        loaded = config.load_config()

    assert "azure" not in loaded
    assert "azure_api_key" not in loaded["app"]


def test_unprefixed_voices_use_edge_tts():
    sentinel = object()

    with patch.object(voice, "edge_tts_synthesize", return_value=sentinel):
        result = voice._single_tts("hello", "en-US-AnaNeural-Female", 1.0, "out.mp3")

    assert result is sentinel
