"""Configuration controls for self-hosted TTS providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from app.config import config
from app.services import voice


def render_settings(
    *,
    tts_mode_enabled: bool,
    selected_tts_server: str,
    voice_name: str,
    tr: Callable[[str], str],
    parse_voices: Callable[[Any], list[str]],
    set_runtime_config: Callable[[str, str, Any], None],
    chatterbox_base_url: str,
    chatterbox_model: str,
    chatterbox_voices: list[str],
    kokoro_base_url: str,
    kokoro_model: str,
    kokoro_voices: list[str],
) -> None:
    """Render Chatterbox and Kokoro settings when either provider is active."""
    if tts_mode_enabled and (
        selected_tts_server == "chatterbox"
        or (voice_name and voice.is_chatterbox_voice(voice_name))
    ):
        _render_provider(
            section="chatterbox",
            label="Chatterbox",
            base_url_default=chatterbox_base_url,
            model_default=chatterbox_model,
            voices_default=chatterbox_voices,
            tr=tr,
            parse_voices=parse_voices,
            set_runtime_config=set_runtime_config,
        )
    if tts_mode_enabled and (
        selected_tts_server == "kokoro"
        or (voice_name and voice.is_kokoro_voice(voice_name))
    ):
        _render_provider(
            section="kokoro",
            label="Kokoro",
            base_url_default=kokoro_base_url,
            model_default=kokoro_model,
            voices_default=kokoro_voices,
            tr=tr,
            parse_voices=parse_voices,
            set_runtime_config=set_runtime_config,
        )


def _render_provider(
    *,
    section: str,
    label: str,
    base_url_default: str,
    model_default: str,
    voices_default: list[str],
    tr: Callable[[str], str],
    parse_voices: Callable[[Any], list[str]],
    set_runtime_config: Callable[[str, str, Any], None],
) -> None:
    """Render the shared base URL, key, model, and optional voices fields."""
    provider_config = getattr(config, section)
    base_url = st.text_input(
        tr(f"{label} Base URL"),
        value=provider_config.get("base_url") or base_url_default,
        key=f"{section}_base_url_input",
        placeholder=tr(f"{label} Base URL Placeholder"),
    )
    set_runtime_config(section, "base_url", (base_url or "").strip())
    api_key = st.text_input(
        tr(f"{label} API Key"),
        value=provider_config.get("api_key", ""),
        type="password",
        key=f"{section}_api_key_input",
    )
    set_runtime_config(section, "api_key", api_key)
    model = st.text_input(
        tr(f"{label} Model"),
        value=provider_config.get("model_id") or model_default,
        key=f"{section}_model_input",
    )
    set_runtime_config(section, "model_id", (model or model_default).strip())

    saved_voices = parse_voices(provider_config.get("voices")) or voices_default
    voices = st.text_input(
        tr(f"{label} Voices"),
        value=", ".join(saved_voices),
        key=f"{section}_voices_input",
        placeholder=tr(f"{label} Voices Placeholder"),
    )
    set_runtime_config(section, "voices", parse_voices(voices))
