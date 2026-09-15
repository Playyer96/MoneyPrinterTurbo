"""Intro / outro blurred overlay + per-paragraph delivery cue controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st

from app.config import config


def _default_option(options: list[tuple[str, str]], saved: str) -> str:
    """Return the saved option when available, otherwise the first option."""
    values = {value for _, value in options}
    return saved if saved in values else options[0][1]


def _default_int(saved, default: float) -> float:
    try:
        return float(saved)
    except (TypeError, ValueError):
        return default


def render_intro_outro_settings(
    panel: Any,
    params: Any,
    *,
    tr: Callable[[str], str],
    saved_ui_bool: Callable[[str, bool], bool],
    stable_selectbox: Callable[..., Any],
    set_runtime_config: Callable[[str, str, Any], None],
    defaults: Mapping[str, Any],
) -> None:
    """Render intro / outro blurred overlay and delivery-cue controls."""
    with panel:
        with st.container(border=True):
            st.write(tr("Intro / Outro Overlay"))

            # ---- Intro ----
            st.session_state.setdefault(
                "intro_enabled_checkbox",
                saved_ui_bool("intro_enabled", defaults["intro_enabled"]),
            )
            params.intro_enabled = st.checkbox(
                tr("Enable Intro Overlay"),
                key="intro_enabled_checkbox",
            )
            set_runtime_config("ui", "intro_enabled", params.intro_enabled)
            intro_disabled = not params.intro_enabled

            saved_intro_text = config.ui.get("intro_text", defaults["intro_text"])
            st.session_state.setdefault(
                "intro_text_area",
                str(saved_intro_text),
            )
            params.intro_text = st.text_area(
                tr("Intro Text (one line per overlay line)"),
                placeholder=tr("Intro Text Help"),
                key="intro_text_area",
                height=80,
                disabled=intro_disabled,
            )
            set_runtime_config("ui", "intro_text", params.intro_text)

            intro_row = st.columns([0.5, 0.5])
            with intro_row[0]:
                params.intro_duration = float(
                    st.number_input(
                        tr("Intro Duration (seconds)"),
                        min_value=0.5,
                        max_value=15.0,
                        value=_default_int(
                            config.ui.get("intro_duration", defaults["intro_duration"]),
                            3.0,
                        ),
                        step=0.5,
                        key="intro_duration_input",
                        disabled=intro_disabled,
                    )
                )
                set_runtime_config("ui", "intro_duration", params.intro_duration)

            with intro_row[1]:
                params.intro_blur_strength = int(
                    st.slider(
                        tr("Intro Blur Strength"),
                        min_value=5,
                        max_value=80,
                        value=int(
                            _default_int(
                                config.ui.get(
                                    "intro_blur_strength",
                                    defaults["intro_blur_strength"],
                                ),
                                35,
                            )
                        ),
                        step=5,
                        key="intro_blur_strength_slider",
                        disabled=intro_disabled,
                    )
                )
                set_runtime_config(
                    "ui", "intro_blur_strength", params.intro_blur_strength
                )

            intro_anim_row = st.columns([0.5, 0.5])
            with intro_anim_row[0]:
                intro_animations = [
                    (tr("Fade In + Out"), "fade"),
                    (tr("Fade In Only"), "fade_in"),
                    (tr("Fade Out Only"), "fade_out"),
                    (tr("None"), "none"),
                ]
                params.intro_animation = stable_selectbox(
                    tr("Intro Animation"),
                    options=[value for _, value in intro_animations],
                    default_value=_default_option(
                        intro_animations,
                        config.ui.get(
                            "intro_animation", defaults.get("intro_animation", "fade")
                        ),
                    ),
                    key="intro_animation_select",
                    format_func=lambda value: dict(intro_animations[::-1]).get(
                        value, value
                    ),
                    disabled=intro_disabled,
                )
                set_runtime_config("ui", "intro_animation", params.intro_animation)

            with intro_anim_row[1]:
                params.intro_tts_enabled = st.checkbox(
                    tr("Narrate Intro With TTS"),
                    key="intro_tts_enabled_checkbox",
                    disabled=intro_disabled,
                    help=tr(
                        "When enabled, the intro text is read aloud using the project's voice and prepended to the audio track."
                    ),
                )
                set_runtime_config("ui", "intro_tts_enabled", params.intro_tts_enabled)

            st.divider()

            # ---- Outro ----
            st.session_state.setdefault(
                "outro_enabled_checkbox",
                saved_ui_bool("outro_enabled", defaults["outro_enabled"]),
            )
            params.outro_enabled = st.checkbox(
                tr("Enable Outro Overlay"),
                key="outro_enabled_checkbox",
            )
            set_runtime_config("ui", "outro_enabled", params.outro_enabled)
            outro_disabled = not params.outro_enabled

            saved_outro_text = config.ui.get("outro_text", defaults["outro_text"])
            st.session_state.setdefault(
                "outro_text_area",
                str(saved_outro_text),
            )
            params.outro_text = st.text_area(
                tr("Outro Text (leave empty for default)"),
                placeholder=tr("Outro Text Help"),
                key="outro_text_area",
                height=80,
                disabled=outro_disabled,
            )
            set_runtime_config("ui", "outro_text", params.outro_text)

            outro_row = st.columns([0.5, 0.5])
            with outro_row[0]:
                params.outro_duration = float(
                    st.number_input(
                        tr("Outro Duration (seconds)"),
                        min_value=0.5,
                        max_value=15.0,
                        value=_default_int(
                            config.ui.get("outro_duration", defaults["outro_duration"]),
                            4.0,
                        ),
                        step=0.5,
                        key="outro_duration_input",
                        disabled=outro_disabled,
                    )
                )
                set_runtime_config("ui", "outro_duration", params.outro_duration)

            with outro_row[1]:
                params.outro_blur_strength = int(
                    st.slider(
                        tr("Outro Blur Strength"),
                        min_value=5,
                        max_value=80,
                        value=int(
                            _default_int(
                                config.ui.get(
                                    "outro_blur_strength",
                                    defaults["outro_blur_strength"],
                                ),
                                35,
                            )
                        ),
                        step=5,
                        key="outro_blur_strength_slider",
                        disabled=outro_disabled,
                    )
                )
                set_runtime_config(
                    "ui", "outro_blur_strength", params.outro_blur_strength
                )

            outro_anim_row = st.columns([0.5, 0.5])
            with outro_anim_row[0]:
                outro_animations = [
                    (tr("Fade In + Out"), "fade"),
                    (tr("Fade In Only"), "fade_in"),
                    (tr("Fade Out Only"), "fade_out"),
                    (tr("None"), "none"),
                ]
                params.outro_animation = stable_selectbox(
                    tr("Outro Animation"),
                    options=[value for _, value in outro_animations],
                    default_value=_default_option(
                        outro_animations,
                        config.ui.get(
                            "outro_animation", defaults.get("outro_animation", "fade")
                        ),
                    ),
                    key="outro_animation_select",
                    format_func=lambda value: dict(outro_animations[::-1]).get(
                        value, value
                    ),
                    disabled=outro_disabled,
                )
                set_runtime_config("ui", "outro_animation", params.outro_animation)

            with outro_anim_row[1]:
                params.outro_tts_enabled = st.checkbox(
                    tr("Narrate Outro With TTS"),
                    key="outro_tts_enabled_checkbox",
                    disabled=outro_disabled,
                    help=tr(
                        "When enabled, the outro text is read aloud using the project's voice and appended to the audio track."
                    ),
                )
                set_runtime_config("ui", "outro_tts_enabled", params.outro_tts_enabled)

            st.divider()

            # ---- Per-paragraph delivery cues ----
            st.session_state.setdefault(
                "delivery_cues_enabled_checkbox",
                saved_ui_bool(
                    "delivery_cues_enabled", defaults["delivery_cues_enabled"]
                ),
            )
            params.delivery_cues_enabled = st.checkbox(
                tr("Generate Per-Paragraph Delivery Cues"),
                key="delivery_cues_enabled_checkbox",
                help=tr("Delivery Cues Help"),
            )
            set_runtime_config(
                "ui", "delivery_cues_enabled", params.delivery_cues_enabled
            )


def render_intro_outro_default_render(
    panel: Any,
    *,
    tr: Callable[[str], str],
    set_runtime_config: Callable[[str, str, Any], None],
    defaults: Mapping[str, Any],
) -> None:
    """Render the in-panel default-reset controls for intro / outro / cues."""
    with panel:
        with st.container(border=True):
            if st.button(
                tr("Restore Intro / Outro Defaults"),
                key="restore_intro_outro_defaults_button",
                use_container_width=True,
            ):
                for key in (
                    "intro_enabled",
                    "intro_text",
                    "intro_duration",
                    "intro_blur_strength",
                    "outro_enabled",
                    "outro_text",
                    "outro_duration",
                    "outro_blur_strength",
                    "delivery_cues_enabled",
                ):
                    if key in defaults:
                        set_runtime_config("ui", key, defaults[key])
                st.toast(tr("Intro / Outro Defaults Restored"))