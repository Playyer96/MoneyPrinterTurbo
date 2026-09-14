"""Provider-specific controls for the background-music panel."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from app.config import config
from app.services import bgm as bgm_service
from app.services import elevenlabs_music as elevenlabs_music_service
from app.services import sonilo as sonilo_service


def render_api_key_control(
    bgm_type: str,
    *,
    tr: Callable[[str], str],
    set_runtime_config: Callable[[str, str, Any], None],
    render_elevenlabs_api_key_input: Callable[..., str],
    elevenlabs_api_key_rendered: bool,
) -> None:
    """Render a provider key input when the selected music source needs one."""
    if bgm_type == "sonilo":
        configured_key = str(config.app.get("sonilo_api_key", "") or "").strip()
        effective_key = configured_key or os.getenv("SONILO_API_KEY", "").strip()
        entered_key = st.text_input(
            tr("Sonilo API Key"),
            value=effective_key,
            type="password",
            key="sonilo_api_key_input",
        ).strip()
        if configured_key or entered_key != effective_key:
            set_runtime_config("app", "sonilo_api_key", entered_key)
    elif bgm_type == "elevenlabs":
        if elevenlabs_api_key_rendered:
            st.caption(tr("ElevenLabs API Key Help"))
        else:
            render_elevenlabs_api_key_input(
                "ElevenLabs Music API Key",
                tr=tr,
                sync_elevenlabs_api_key_input=lambda: None,
            )


def render_music_prompt(
    params: Any,
    previous_bgm_type: str | None,
    bgm_enabled: bool,
    *,
    tr: Callable[[str], str],
    saved_ui_text: Callable[..., str],
    set_runtime_config: Callable[[str, str, Any], None],
) -> None:
    """Render provider music prompts, connection checks, and key warnings."""
    if params.bgm_type == "sonilo":
        if previous_bgm_type != "sonilo":
            st.session_state["sonilo_bgm_prompt_input"] = saved_ui_text(
                "sonilo_bgm_prompt",
                max_length=sonilo_service.MAX_PROMPT_LENGTH,
            )
        params.video_music_prompt = st.text_input(
            tr("Sonilo Music Prompt"),
            key="sonilo_bgm_prompt_input",
            max_chars=sonilo_service.MAX_PROMPT_LENGTH,
            help=tr("Sonilo Music Prompt Help"),
        ).strip()
        set_runtime_config("ui", "sonilo_bgm_prompt", params.video_music_prompt)
        if params.video_count > 1:
            st.warning(tr("Sonilo Multiple Videos Warning"))
        if st.button(
            tr("Test Sonilo Connection"),
            key="test_sonilo_connection_button",
            use_container_width=True,
        ):
            try:
                sonilo_service.test_connection()
            except sonilo_service.SoniloError as exc:
                logger.warning(f"Sonilo connection test failed: {exc}")
                st.error(tr("Sonilo Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("Sonilo Connection Test Succeeded"))
    elif params.bgm_type == "elevenlabs":
        if previous_bgm_type != "elevenlabs":
            st.session_state["elevenlabs_music_prompt_input"] = saved_ui_text(
                "elevenlabs_music_prompt",
                max_length=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            )
        params.video_music_prompt = st.text_input(
            tr("ElevenLabs Music Prompt"),
            key="elevenlabs_music_prompt_input",
            max_chars=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            help=tr("ElevenLabs Music Prompt Help"),
        ).strip()
        set_runtime_config("ui", "elevenlabs_music_prompt", params.video_music_prompt)
        if params.video_count > 1:
            st.warning(tr("ElevenLabs Multiple Videos Warning"))
        if st.button(
            tr("Test ElevenLabs Connection"),
            key="test_elevenlabs_music_connection_button",
            use_container_width=True,
        ):
            try:
                elevenlabs_music_service.test_connection()
            except elevenlabs_music_service.ElevenLabsPaidPlanRequiredError:
                st.error(tr("ElevenLabs Paid Plan Required"))
            except elevenlabs_music_service.ElevenLabsMusicError as exc:
                logger.warning(f"ElevenLabs connection test failed: {exc}")
                st.error(
                    tr("ElevenLabs Connection Test Failed").format(error=str(exc))
                )
            else:
                st.success(tr("ElevenLabs Connection Test Succeeded"))

    if params.bgm_type == "sonilo" and bgm_enabled and not sonilo_service.is_enabled():
        st.warning(tr("Sonilo API Key Required"))
    elif (
        params.bgm_type == "elevenlabs"
        and bgm_enabled
        and not elevenlabs_music_service.is_enabled()
    ):
        st.warning(tr("ElevenLabs API Key Required"))



def render_elevenlabs_api_key_input(
    label_key: str,
    *,
    tr: Callable[[str], str],
    sync_elevenlabs_api_key_input: Callable[[], None],
) -> str:
    """Render the unique ElevenLabs API key input shared with TTS and soundtrack."""
    sync_elevenlabs_api_key_input()
    return st.text_input(
        tr(label_key),
        type="password",
        key="elevenlabs_api_key_input",
    ).strip()


def render_background_music_settings(
params: Any,
elevenlabs_api_key_rendered: bool = False,
*,
tr: Callable[[str], str],
set_runtime_config: Callable[[str, str, Any], None],
saved_ui_choice: Callable[..., Any],
saved_ui_text: Callable[..., str],
stable_selectbox: Callable[..., Any],
localized_widget_key: Callable[[str], str],
render_elevenlabs_api_key_input: Callable[[str], str],
) -> Any:
    """Render the background music source and volume settings, and return the uploaded file to be saved this time."""
    uploaded_bgm_file = None
    previous_bgm_type = st.session_state.get("last_rendered_bgm_type")
    st.divider()
    bgm_options = [
        (tr("No Background Music"), ""),
        (tr("Random Background Music"), "random"),
        (tr("Preset Song"), "preset"),
        (tr("Custom Background Music"), "custom"),
        (tr("Sonilo Background Music"), "sonilo"),
        (tr("ElevenLabs Background Music"), "elevenlabs"),
    ]
    selected_bgm_type = stable_selectbox(
        tr("Background Music Source"),
        options=[value for _, value in bgm_options],
        default_value=saved_ui_choice(
            "bgm_type",
            [value for _, value in bgm_options],
            "random",
        ),
        key="bgm_type_select",
        format_func=lambda value: dict((v, label) for label, v in bgm_options)[value],
    )
    params.bgm_type = selected_bgm_type
    set_runtime_config("ui", "bgm_type", params.bgm_type)
    render_api_key_control(
        params.bgm_type,
        tr=tr,
        set_runtime_config=set_runtime_config,
        render_elevenlabs_api_key_input=render_elevenlabs_api_key_input,
        elevenlabs_api_key_rendered=elevenlabs_api_key_rendered,
    )

    bgm_volume_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params.bgm_volume = stable_selectbox(
        tr("Background Music Volume"),
        options=bgm_volume_options,
        default_value=saved_ui_choice("bgm_volume", bgm_volume_options, 0.2),
        key="bgm_volume_select",
        format_func=lambda value: f"{int(value * 100)}%",
        disabled=not params.bgm_type,
    )
    set_runtime_config("ui", "bgm_volume", params.bgm_volume)
    bgm_enabled = bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)

    if params.bgm_type == "custom":
        uploaded_bgm_file = st.file_uploader(
            tr("Upload Background Music"),
            type=[
                extension.removeprefix(".")
                for extension in bgm_service.SUPPORTED_BGM_EXTENSIONS
            ],
            accept_multiple_files=False,
            key="custom_bgm_uploader",
            help=tr("Upload Background Music Help"),
        )
        if uploaded_bgm_file is not None and bgm_enabled:
            try:
                safe_name = bgm_service.sanitize_upload_filename(uploaded_bgm_file.name)
                # Streamlit will re-execute the page after adjusting any controls such as volume. Use content hashing
                # Differentiate uploaded files and cache the complete decoding results in the current session. You can neither rely on the same name,
                # Misuse of old results for files of the same size also avoids calling FFmpeg repeatedly for each rerun.
                validation_key = (
                    safe_name,
                    uploaded_bgm_file.size,
                    hashlib.sha256(uploaded_bgm_file.getbuffer()).hexdigest(),
                )
                cached_validation = st.session_state.get("custom_bgm_validation")
                if (
                    not cached_validation
                    or cached_validation.get("key") != validation_key
                ):
                    try:
                        bgm_service.validate_bgm_upload(
                            uploaded_bgm_file.name, uploaded_bgm_file
                        )
                    except bgm_service.BgmUploadError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "upload",
                        }
                        # The failed results of the same file fingerprint will be entered into the session cache, so here only
                        # Record it once when the verification is actually executed for the first time to avoid rerun of ordinary controls and refresh the screen.
                        logger.warning(
                            "WebUI background music validation rejected: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    except bgm_service.BgmServiceError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "service",
                        }
                        logger.error(
                            "WebUI background music validation failed: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    else:
                        cached_validation = {
                            "key": validation_key,
                            "error": "",
                            "error_type": "",
                        }
                    st.session_state["custom_bgm_validation"] = cached_validation

                if cached_validation.get("error"):
                    if cached_validation.get("error_type") == "service":
                        raise bgm_service.BgmServiceError(cached_validation["error"])
                    raise bgm_service.BgmUploadError(cached_validation["error"])
            except bgm_service.BgmUploadError:
                # Illegal files cannot inherit the name of the last valid upload, otherwise the task parameters may still point to
                # Historical BGM. Keep the UploadedFile return value so that it will still be finalized when the user clicks Generate
                # The server verifies the interception instead of silently generating a video without background music.
                params.bgm_file = ""
                st.error(tr("Invalid Background Music"))
            except bgm_service.BgmServiceError:
                params.bgm_file = ""
                st.error(tr("Background Music Validation Failed"))
            else:
                # The player and "Ready" will be displayed only after the complete decoding verification is passed. Files are still only clicking
                # Persisted on build, user merely previewing or subsequently removing files does not pollute storage/bgm.
                uploaded_mime_type = str(getattr(uploaded_bgm_file, "type", "") or "")
                preview_mime_type = (
                    uploaded_mime_type
                    if uploaded_mime_type.startswith("audio/")
                    else mimetypes.guess_type(safe_name)[0] or "audio/mpeg"
                )
                st.audio(uploaded_bgm_file, format=preview_mime_type)
                st.info(f"{tr('Background Music Ready')}: {safe_name}")
                params.bgm_file = safe_name

        # Streamlit cleans up the widget state of a conditional widget when it is temporarily not rendering.
        # Use the persisted value to restore when switching back from other BGM sources; under the same source
        # The previous_bgm_type does not change when the user actively clears it, so it will not be bounced by the old value.
        if previous_bgm_type != "custom":
            st.session_state["custom_bgm_file_input"] = saved_ui_text(
                "custom_bgm_file"
            )
        custom_bgm_file = st.text_input(
            tr("Custom Background Music File"),
            key="custom_bgm_file_input",
            disabled=uploaded_bgm_file is not None,
        )
        set_runtime_config(
            "ui", "custom_bgm_file", custom_bgm_file.strip()
        )
        if uploaded_bgm_file is None and custom_bgm_file and bgm_enabled:
            # The file name is mapped to storage/bgm or resource/songs by the service layer and then verified.
            # The UI does not accept any paths outside of the two whitelisted directories.
            params.bgm_file = custom_bgm_file.strip()
        elif not bgm_enabled:
            # The upload control continues to retain the files selected by the user, and the next rerun after turning up the volume will automatically
            # Complete verification; the current task parameters must be cleared to prevent the 0 volume task from saving or parsing the file.
            params.bgm_file = ""

    if params.bgm_type == "preset":
        # The service layer has uniformly completed extension, temporary file and symbolic link verification. Directly reuse it here
        # As a result, the UI is prevented from maintaining a second set of enumeration rules, and differences will not occur when subsequent formats are added.
        available_song_paths = bgm_service.list_builtin_bgm_files()
        songs_by_name = {
            os.path.basename(song_path): song_path for song_path in available_song_paths
        }
        available_songs = list(songs_by_name)
        if not available_songs:
            st.warning(tr("No Background Music Available"))
            params.bgm_file = ""
        else:
            default_preset_song = saved_ui_text("preset_song", available_songs[0])
            requested_preset_song = st.session_state.get(
                localized_widget_key("preset_song_select"), default_preset_song
            )
            if requested_preset_song not in available_songs:
                # Settings exported from historical missions or other versions may reference songs that do not exist in the current installation.
                # After a clear prompt, stable_selectbox will revert to the first song to avoid silently changing songs.
                st.warning(tr("Selected Background Music Unavailable"))
            selected_song = stable_selectbox(
                tr("Preset Song"),
                options=available_songs,
                default_value=(
                    default_preset_song
                    if default_preset_song in available_songs
                    else available_songs[0]
                ),
                key="preset_song_select",
            )
            set_runtime_config("ui", "preset_song", selected_song)
            # Online listening is provided immediately after the user selects the song. The player reads the data just passed through the service layer
            # The real path obtained by whitelist verification does not accept any file path entered on the page.
            selected_song_path = songs_by_name[selected_song]
            preview_mime_type = (
                mimetypes.guess_type(selected_song_path)[0] or "audio/mpeg"
            )
            preview_available = True
            try:
                # When Streamlit fails to read the path, it will wrap OSError into an internal exception, resulting in the following
                # Unable to handle by file error. Read the bytes yourself first, which not only maintains the player behavior, but also allows
                # Docker mounts that temporarily fail, permissions change, and other situations fall steadily into controllable branches.
                selected_song_bytes = Path(selected_song_path).read_bytes()
            except OSError as exc:
                preview_available = False
                # Files may be deleted by other processes after enumeration. If the audition fails, the page or video cannot be interrupted.
                # Parameter editing, but logs need to be kept to locate running environment and mounting problems.
                logger.warning(
                    "failed to preview preset background music: "
                    f"name={selected_song}, error={str(exc)}"
                )
                st.warning(tr("Background Music Preview Failed"))
            else:
                st.audio(selected_song_bytes, format=preview_mime_type)
            if bgm_enabled and preview_available:
                params.bgm_file = selected_song
            else:
                params.bgm_file = ""

    render_music_prompt(
        params,
        previous_bgm_type,
        bgm_enabled,
        tr=tr,
        saved_ui_text=saved_ui_text,
        set_runtime_config=set_runtime_config,
    )
    st.session_state["last_rendered_bgm_type"] = params.bgm_type
    return uploaded_bgm_file



