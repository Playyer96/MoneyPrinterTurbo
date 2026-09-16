"""Title-overlay controls for the WebUI generation form."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st

from app.config import config


def _default_option(options: list[tuple[str, str]], saved: str) -> str:
    """Return the saved option when available, otherwise the first option."""
    values = {value for _, value in options}
    return saved if saved in values else options[0][1]


def render_title_settings(
    panel: Any,
    params: Any,
    *,
    tr: Callable[[str], str],
    saved_ui_bool: Callable[[str, bool], bool],
    stable_selectbox: Callable[..., Any],
    set_runtime_config: Callable[[str, str, Any], None],
    defaults: Mapping[str, Any],
) -> None:
    """Render title overlay controls and write them to video parameters."""
    with panel:
        with st.container(border=True):
            st.write(tr("Video Title Settings"))
            st.session_state.setdefault(
                "title_enabled_checkbox",
                saved_ui_bool("title_enabled", defaults["title_enabled"]),
            )
            params.title_enabled = st.checkbox(
                tr("Enable Title / Hook Banner"),
                key="title_enabled_checkbox",
            )
            set_runtime_config("ui", "title_enabled", params.title_enabled)
            title_disabled = not params.title_enabled

            saved_title_text = config.ui.get("title_text", defaults["title_text"])
            st.session_state.setdefault("title_text_input", str(saved_title_text))
            params.title_text = st.text_input(
                tr("Title Text"),
                placeholder=tr("Title Text Help"),
                key="title_text_input",
                disabled=title_disabled,
            )
            set_runtime_config("ui", "title_text", params.title_text)

            title_styles = [
                (tr("TikTok Yellow Badge"), "tiktok_yellow"),
                (tr("Breaking Red Banner"), "red_banner"),
                (tr("CapCut Dark Pill"), "capcut_black"),
                (tr("Neon Cyber Glow"), "neon_cyan"),
                (tr("Minimalist Bold White"), "minimalist_white"),
                (tr("Golden Luxury Card"), "golden_luxury"),
                (tr("Comic Bang Punch"), "comic_punch"),
            ]
            params.title_style = stable_selectbox(
                tr("Title Style"),
                options=[value for _, value in title_styles],
                default_value=_default_option(
                    title_styles,
                    config.ui.get("title_style", defaults["title_style"]),
                ),
                key="title_style_select",
                format_func=lambda value: dict(title_styles[::-1]).get(value, value),
                disabled=title_disabled,
            )
            set_runtime_config("ui", "title_style", params.title_style)

            title_row = st.columns([0.5, 0.5])
            with title_row[0]:
                title_positions = [
                    (tr("Top"), "top"),
                    (tr("Center"), "center"),
                    (tr("Bottom"), "bottom"),
                ]
                params.title_position = stable_selectbox(
                    tr("Title Position"),
                    options=[value for _, value in title_positions],
                    default_value=_default_option(
                        title_positions,
                        config.ui.get("title_position", defaults["title_position"]),
                    ),
                    key="title_position_select",
                    format_func=lambda value: dict(title_positions[::-1]).get(
                        value, value
                    ),
                    disabled=title_disabled,
                )
                set_runtime_config("ui", "title_position", params.title_position)

            with title_row[1]:
                title_durations = [
                    (tr("Intro (First 4 Seconds)"), "intro"),
                    (tr("Full Video"), "full"),
                ]
                params.title_duration = stable_selectbox(
                    tr("Title Duration"),
                    options=[value for _, value in title_durations],
                    default_value=_default_option(
                        title_durations,
                        config.ui.get("title_duration", defaults["title_duration"]),
                    ),
                    key="title_duration_select",
                    format_func=lambda value: dict(title_durations[::-1]).get(
                        value, value
                    ),
                    disabled=title_disabled,
                )
                set_runtime_config("ui", "title_duration", params.title_duration)

            title_animations = [
                (tr("Pop Up (Spring)"), "pop_spring"),
                (tr("Scale Up (Punch)"), "scale_up"),
                (tr("Smooth Fade"), "fade"),
                (tr("Slide Up"), "slide_up"),
                (tr("Shake (Impact)"), "shake"),
                (tr("None"), "none"),
            ]
            params.title_animation = stable_selectbox(
                tr("Title Animation"),
                options=[value for _, value in title_animations],
                default_value=_default_option(
                    title_animations,
                    config.ui.get("title_animation", defaults["title_animation"]),
                ),
                key="title_animation_select",
                format_func=lambda value: dict(title_animations[::-1]).get(
                    value, value
                ),
                disabled=title_disabled,
            )
            set_runtime_config("ui", "title_animation", params.title_animation)
