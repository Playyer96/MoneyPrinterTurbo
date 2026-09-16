"""OmniVoice settings and clone-profile controls for the WebUI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from app.config import config
from app.services import voice


def render_omnivoice_settings(
    *,
    tts_mode_enabled: bool,
    selected_tts_server: str,
    voice_name: str,
    tr: Callable[[str], str],
    set_runtime_config: Callable[[str, str, Any], None],
    default_base_url: str,
) -> None:
    """Render OmniVoice configuration and cloned-profile management controls."""
    if not tts_mode_enabled or (
        selected_tts_server != "omnivoice"
        and not (voice_name and voice.is_omnivoice_voice(voice_name))
    ):
        return

    base_url = st.text_input(
        tr("OmniVoice Base URL"),
        value=config.omnivoice.get("base_url") or default_base_url,
        key="omnivoice_base_url_input",
        placeholder="http://127.0.0.1:8780",
    )
    set_runtime_config("omnivoice", "base_url", (base_url or "").strip())
    style = st.text_input(
        tr("OmniVoice Style Instruction"),
        value=config.omnivoice.get("style", ""),
        key="omnivoice_style_input",
        placeholder="energetic and playful; laugh warmly; angry and forceful",
        help=(
            "Applies to the bundled narrator preset. Cloned profiles preserve "
            "their recorded voice; use an OmniVoice tag such as [laughter] in "
            "the script for a non-verbal reaction."
        ),
    )
    set_runtime_config("omnivoice", "style", (style or "").strip()[:240])

    profile_sample = st.file_uploader(
        tr("OmniVoice Voice Sample"),
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="omnivoice_profile_sample_upload",
    )
    profile_name = st.text_input(
        tr("OmniVoice Profile Name"),
        value="",
        key="omnivoice_profile_name_input",
    )
    if st.button(
        tr("Create OmniVoice Profile"),
        key="omnivoice_create_profile_button",
    ):
        ok, message = voice.create_omnivoice_profile(
            profile_name=profile_name,
            audio_bytes=profile_sample.getvalue() if profile_sample else b"",
            original_filename=profile_sample.name if profile_sample else "",
        )
        if ok:
            st.success(f"{tr('OmniVoice Profile Created')}: {profile_name.strip()}")
            st.session_state.pop("omnivoice_voice_catalog", None)
        else:
            st.error(f"{tr('OmniVoice Profile Create Failed')}: {message}")

    profiles = voice.get_omnivoice_profiles()
    if not profiles:
        return
    profile_to_remove = st.selectbox(
        tr("OmniVoice Profile to Remove"),
        options=profiles,
        key="omnivoice_profile_remove_select",
    )
    if st.button(
        tr("Remove OmniVoice Profile"),
        key="omnivoice_remove_profile_button",
    ):
        ok, message = voice.delete_omnivoice_profile(profile_to_remove)
        if ok:
            st.success(f"{tr('OmniVoice Profile Removed')}: {profile_to_remove}")
            st.session_state.pop("omnivoice_voice_catalog", None)
        else:
            st.error(f"{tr('OmniVoice Profile Remove Failed')}: {message}")
