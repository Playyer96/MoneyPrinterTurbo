"""Small, independent settings tabs for the WebUI."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import streamlit as st

from app.config import config
from app.services import (
    metaso_minimax,
    ofox,
    upload_post as upload_post_service,
    volcengine_seedance,
)


def render_publish_settings(
    panel: Any,
    *,
    tr: Callable[[str], str],
    set_runtime_config: Callable[[str, str, Any], None],
    api_keys_url: str,
    manage_users_url: str,
) -> None:
    """Render the independent Upload-Post settings tab."""
    with panel:
        st.write(tr("Automatically publish generated videos to social media using upload-post.com"))
        st.info(
            tr("Upload-Post Setup Guide").format(
                api_keys_url=api_keys_url,
                manage_users_url=manage_users_url,
            )
        )

        is_enabled = config.app.get("upload_post_enabled", False)
        is_auto = config.app.get("upload_post_auto_upload", False)
        upload_post_enabled = st.checkbox(
            tr("Enable Upload-Post Integration"),
            value=is_enabled,
            key="upload_post_enabled_checkbox",
        )
        if upload_post_enabled != is_enabled:
            set_runtime_config("app", "upload_post_enabled", upload_post_enabled)

        upload_post_auto_upload = st.checkbox(
            tr("Enable Auto-Publish"),
            value=is_auto,
            key="upload_post_auto_upload_checkbox",
        )
        if upload_post_auto_upload != is_auto:
            set_runtime_config("app", "upload_post_auto_upload", upload_post_auto_upload)

        upload_post_api_key = st.text_input(
            tr("Upload-Post API Key"),
            value=config.app.get("upload_post_api_key", ""),
            type="password",
            help=tr("Upload-Post API Key Help").format(api_keys_url=api_keys_url),
            key="upload_post_api_key_input",
        )
        if upload_post_api_key != config.app.get("upload_post_api_key", ""):
            set_runtime_config("app", "upload_post_api_key", upload_post_api_key)

        upload_post_username = st.text_input(
            tr("Upload-Post Profile Username"),
            value=config.app.get("upload_post_username", ""),
            help=tr("Upload-Post Profile Username Help").format(
                manage_users_url=manage_users_url
            ),
            key="upload_post_username_input",
        )
        if upload_post_username != config.app.get("upload_post_username", ""):
            set_runtime_config("app", "upload_post_username", upload_post_username)

        upload_post_platforms = st.multiselect(
            tr("Platforms"),
            options=["tiktok", "instagram", "youtube"],
            default=config.app.get("upload_post_platforms", ["tiktok", "instagram"]),
            help="Select platforms to publish to",
            key="upload_post_platforms_multiselect",
        )
        if upload_post_platforms != config.app.get(
            "upload_post_platforms", ["tiktok", "instagram"]
        ):
            set_runtime_config("app", "upload_post_platforms", upload_post_platforms)

        if "youtube" in upload_post_platforms:
            yt_status_options = ["public", "private", "unlisted"]
            yt_saved = config.app.get("upload_post_youtube_privacy_status", "public")
            if yt_saved not in yt_status_options:
                yt_saved = "public"
            privacy_status = st.selectbox(
                tr("YouTube Privacy Status"),
                options=yt_status_options,
                index=yt_status_options.index(yt_saved),
                key="upload_post_youtube_privacy_status_selectbox",
            )
            if privacy_status != config.app.get(
                "upload_post_youtube_privacy_status", "public"
            ):
                set_runtime_config(
                    "app", "upload_post_youtube_privacy_status", privacy_status
                )

        if st.button(
            tr("Test Connection"),
            key="upload_post_test_connection_button",
            use_container_width=True,
            help=tr("Test Connection Help"),
        ):
            test_result = upload_post_service.upload_post_service.test_connection()
            if not test_result.get("configured"):
                st.warning(tr("Test Connection Not Configured"))
            elif test_result.get("valid"):
                st.success(tr("Test Connection Success"))
            else:
                st.error(tr("Test Connection Failed"))



def render_material_settings(
    panel: Any,
    *,
    tr: Callable[[str], str],
    set_runtime_config: Callable[[str, str, Any], None],
    get_material_api_keys: Callable[[str], str],
    save_material_api_keys: Callable[[str, str], None],
    stable_selectbox: Callable[..., Any],
) -> None:
    """Render the Material API settings tab."""
    with panel:
            # Material Provider Click "Search stock materials/AI generated videos/AI generated pictures"
            # Grouping to avoid all fields being mixed in a long list as the number of Providers increases.
            # Grouping only adjusts the display level and does not change existing configuration keys. After upgrading, old users
            # The original config.toml value will continue to be read.
            with st.container(border=True):
                st.markdown(f"#### {tr('Stock Video APIs')}")
                st.caption(tr("Stock Video APIs Help"))

                pexels_api_key = get_material_api_keys("pexels_api_keys")
                pixabay_api_key = get_material_api_keys("pixabay_api_keys")
                coverr_api_key = get_material_api_keys("coverr_api_keys")
                pexels_api_key = st.text_input(
                    tr("Pexels API Key"),
                    value=pexels_api_key,
                    type="password",
                    key="pexels_api_keys_input",
                )
                save_material_api_keys("pexels_api_keys", pexels_api_key)

                pixabay_api_key = st.text_input(
                    tr("Pixabay API Key"),
                    value=pixabay_api_key,
                    type="password",
                    key="pixabay_api_keys_input",
                )
                save_material_api_keys("pixabay_api_keys", pixabay_api_key)

                coverr_api_key = st.text_input(
                    tr("Coverr API Key"),
                    value=coverr_api_key,
                    type="password",
                    key="coverr_api_keys_input",
                )
                save_material_api_keys("coverr_api_keys", coverr_api_key)

            with st.container(border=True):
                st.markdown(f"#### {tr('AI Video Generation APIs')}")
                st.caption(tr("AI Video Generation APIs Help"))

                # Video generation provider displays first by sponsor, in order within the sponsor
                # In line with business agreements: Secret Tower, Odds Cloud, and Volcano Engine.
                st.markdown(f"**{tr('Metaso MiniMax H3')}**")
                metaso_api_key = st.text_input(
                    tr("Metaso MiniMax API Key"),
                    value=str(
                        config.app.get("metaso_minimax_api_key", "") or ""
                    ).strip(),
                    type="password",
                    help=tr("Metaso MiniMax API Key Help"),
                    key="metaso_minimax_api_key_input",
                )
                set_runtime_config(
                    "app", "metaso_minimax_api_key", metaso_api_key.strip()
                )
                configured_metaso_base_url = str(
                    config.app.get(
                        "metaso_minimax_base_url",
                        metaso_minimax.DEFAULT_BASE_URL,
                    )
                    or metaso_minimax.DEFAULT_BASE_URL
                ).strip()
                metaso_base_url = st.text_input(
                    tr("Metaso MiniMax Base URL"),
                    value=(
                        ""
                        if configured_metaso_base_url == metaso_minimax.DEFAULT_BASE_URL
                        else configured_metaso_base_url
                    ),
                    placeholder=metaso_minimax.DEFAULT_BASE_URL,
                    key="metaso_minimax_base_url_input",
                )
                set_runtime_config(
                    "app",
                    "metaso_minimax_base_url",
                    metaso_base_url.strip() or metaso_minimax.DEFAULT_BASE_URL,
                )
                configured_metaso_resolution = (
                    str(
                        config.app.get(
                            "metaso_minimax_resolution",
                            metaso_minimax.DEFAULT_RESOLUTION,
                        )
                    )
                    .strip()
                    .upper()
                )
                metaso_resolution_options = sorted(
                    metaso_minimax.SUPPORTED_RESOLUTIONS,
                    key=lambda value: value != metaso_minimax.DEFAULT_RESOLUTION,
                )
                resolution_is_valid = (
                    configured_metaso_resolution
                    in metaso_minimax.SUPPORTED_RESOLUTIONS
                )
                if not resolution_is_valid:
                    # Resolution directly affects billing. In case of manual configuration error, retain the original value and ask the user
                    # It is an active choice and cannot be silently changed to the more expensive 2K when the settings pop-up window is opened.
                    st.error(
                        tr("Metaso MiniMax Invalid Resolution").format(
                            value=configured_metaso_resolution,
                            supported=", ".join(metaso_resolution_options),
                        )
                    )
                metaso_resolution = st.selectbox(
                    tr("Metaso MiniMax Resolution"),
                    options=metaso_resolution_options,
                    index=(
                        metaso_resolution_options.index(configured_metaso_resolution)
                        if resolution_is_valid
                        else None
                    ),
                    key="metaso_minimax_resolution_input",
                    help=tr("Metaso MiniMax Resolution Help"),
                    placeholder=tr("Select Metaso MiniMax Resolution"),
                )
                if metaso_resolution is not None:
                    set_runtime_config(
                        "app", "metaso_minimax_resolution", metaso_resolution
                    )

                st.divider()
                st.markdown(f"**{tr('Shengsuan Cloud AI Video')}**")
                app_config_snapshot = config.snapshot_config_with_pending(config.app)
                if (
                    str(app_config_snapshot.get("llm_provider", "") or "").lower()
                    == "shengsuanyun"
                ):
                    # When the large model Provider has been selected to win the cloud, the video generation is directly reused.
                    # For the same key, an independent input box that is prone to ambiguity is no longer displayed.
                    st.caption(tr("Shengsuan Cloud API Key Reused"))
                else:
                    configured_loomloom_token = str(
                        app_config_snapshot.get("loomloom_api_token", "") or ""
                    ).strip()
                    loomloom_api_token = st.text_input(
                        tr("Shengsuan Cloud API Key"),
                        value=configured_loomloom_token,
                        type="password",
                        key="loomloom_api_token_input",
                        help=tr("Shengsuan Cloud API Key Help"),
                        placeholder=tr("Shengsuan Cloud API Key Placeholder"),
                    ).strip()
                    set_runtime_config(
                        "app", "loomloom_api_token", loomloom_api_token
                    )

                st.divider()
                seedance_api_key_value = str(
                    config.app.get("volcengine_seedance_api_key", "") or ""
                ).strip()
                shared_ark_api_key = str(
                    config.app.get("volcengine_api_key", "") or ""
                ).strip()
                environment_ark_api_key = os.getenv(
                    "VOLCENGINE_ARK_API_KEY", ""
                ).strip()
                seedance_reuses_llm_key = bool(
                    not seedance_api_key_value
                    and not environment_ark_api_key
                    and shared_ark_api_key
                )
                seedance_title = f"**{tr('Volcano Engine Seedance')}**"
                if seedance_reuses_llm_key:
                    # Only the reused large model key cannot be directly seen from the current input box, so keep this prompt.
                    # This can prevent users from mistakenly thinking that they must fill in the information repeatedly; the general configuration status will not be described again.
                    seedance_title += f" :blue[{tr('Reusing LLM API Key')}]"
                st.markdown(seedance_title)
                seedance_api_key = st.text_input(
                    tr("Volcano Engine Ark API Key"),
                    value=seedance_api_key_value,
                    type="password",
                    help=tr("Volcano Engine Ark API Key Help"),
                    key="volcengine_seedance_api_key_input",
                )
                set_runtime_config(
                    "app", "volcengine_seedance_api_key", seedance_api_key.strip()
                )
                configured_seedance_model = str(
                    config.app.get(
                        "volcengine_seedance_model",
                        volcengine_seedance.DEFAULT_MODEL_ID,
                    )
                    or volcengine_seedance.DEFAULT_MODEL_ID
                ).strip()
                seedance_model = st.text_input(
                    tr("Volcano Engine Seedance Model"),
                    # Built-in default values ​​are displayed through placeholders, user-defined
                    # Model or access point IDs are still displayed and saved as real values.
                    value=(
                        ""
                        if configured_seedance_model
                        == volcengine_seedance.DEFAULT_MODEL_ID
                        else configured_seedance_model
                    ),
                    placeholder=volcengine_seedance.DEFAULT_MODEL_ID,
                    key="volcengine_seedance_model_input",
                )
                set_runtime_config(
                    "app",
                    "volcengine_seedance_model",
                    seedance_model.strip() or volcengine_seedance.DEFAULT_MODEL_ID,
                )
                configured_seedance_base_url = str(
                    config.app.get(
                        "volcengine_seedance_base_url",
                        volcengine_seedance.DEFAULT_BASE_URL,
                    )
                    or volcengine_seedance.DEFAULT_BASE_URL
                ).strip()
                seedance_base_url = st.text_input(
                    tr("Volcano Engine Ark Base URL"),
                    value=(
                        ""
                        if configured_seedance_base_url
                        == volcengine_seedance.DEFAULT_BASE_URL
                        else configured_seedance_base_url
                    ),
                    placeholder=volcengine_seedance.DEFAULT_BASE_URL,
                    key="volcengine_seedance_base_url_input",
                )
                set_runtime_config(
                    "app",
                    "volcengine_seedance_base_url",
                    seedance_base_url.strip() or volcengine_seedance.DEFAULT_BASE_URL,
                )

                st.divider()
                wavespeed_api_key = get_material_api_keys("wavespeed_api_keys")
                st.markdown("**WaveSpeed**")
                wavespeed_api_key = st.text_input(
                    tr("WaveSpeed API Key"),
                    value=wavespeed_api_key,
                    type="password",
                    key="wavespeed_api_keys_input",
                )
                save_material_api_keys("wavespeed_api_keys", wavespeed_api_key)

                st.divider()
                st.markdown("**OFox**")
                ofox_api_key = st.text_input(
                    tr("OFox API Key"),
                    value=str(config.app.get("ofox_api_key", "") or ""),
                    type="password",
                    key="ofox_api_key_input",
                )
                set_runtime_config("app", "ofox_api_key", ofox_api_key.strip())
                ofox_model = st.text_input(
                    tr("OFox Text-to-Video Model"),
                    value=str(
                        config.app.get(
                            "ofox_text_to_video_model",
                            ofox.DEFAULT_MODEL_ID,
                        )
                        or ofox.DEFAULT_MODEL_ID
                    ),
                    key="ofox_text_to_video_model_input",
                )
                set_runtime_config(
                    "app", "ofox_text_to_video_model", ofox_model.strip()
                )
                configured_ofox_base_url = str(
                    config.app.get("ofox_base_url", ofox.DEFAULT_BASE_URL)
                    or ofox.DEFAULT_BASE_URL
                ).strip()
                ofox_base_url = st.text_input(
                    tr("OFox Base URL"),
                    value=(
                        ""
                        if configured_ofox_base_url == ofox.DEFAULT_BASE_URL
                        else configured_ofox_base_url
                    ),
                    placeholder=ofox.DEFAULT_BASE_URL,
                    key="ofox_base_url_input",
                )
                set_runtime_config(
                    "app",
                    "ofox_base_url",
                    ofox_base_url.strip() or ofox.DEFAULT_BASE_URL,
                )
                ofox_vendor_options = [
                    (tr("OFox Vendor BytePlus"), "byteplus"),
                    (tr("OFox Vendor Volcengine"), "volcengine"),
                    (tr("OFox Vendor Auto"), ""),
                ]
                configured_ofox_vendor = str(
                    config.app.get("ofox_provider", ofox.DEFAULT_PROVIDER_TYPE)
                    or ""
                ).strip()
                if configured_ofox_vendor not in {
                    value for _, value in ofox_vendor_options
                }:
                    # Keep this selection when the user manually pinned other vendor names in config.toml.
                    # Avoid being overwritten back to the default value by the drop-down box when opening the settings page.
                    ofox_vendor_options.append(
                        (configured_ofox_vendor, configured_ofox_vendor)
                    )
                selected_ofox_vendor = stable_selectbox(
                    tr("OFox Upstream Vendor"),
                    options=[value for _, value in ofox_vendor_options],
                    default_value=configured_ofox_vendor,
                    key="ofox_provider_select",
                    format_func=lambda value: dict(
                        (v, label) for label, v in ofox_vendor_options
                    )[value],
                    help=tr("OFox Upstream Vendor Help"),
                )
                set_runtime_config("app", "ofox_provider", selected_ofox_vendor)

            with st.container(border=True):
                st.markdown(f"#### {tr('AI Image Generation APIs')}")
                st.caption(tr("AI Image Generation APIs Help"))
                st.markdown(f"**{tr('OpenAI Compatible Text-to-Image')}**")

                openai_image_base_url = st.text_input(
                    tr("OpenAI Image Base URL"),
                    value=str(config.app.get("openai_image_base_url", "") or ""),
                    placeholder="https://api.openai.com/v1",
                    key="openai_image_base_url_input",
                )
                set_runtime_config(
                    "app", "openai_image_base_url", openai_image_base_url.strip()
                )

                openai_image_api_key = get_material_api_keys(
                    "openai_image_api_keys"
                )
                openai_image_api_key = st.text_input(
                    tr("OpenAI Image API Key"),
                    value=openai_image_api_key,
                    type="password",
                    help=tr("OpenAI Image API Key Help"),
                    key="openai_image_api_keys_input",
                )
                save_material_api_keys(
                    "openai_image_api_keys", openai_image_api_key
                )

                openai_image_model = st.text_input(
                    tr("OpenAI Image Model"),
                    value=str(config.app.get("openai_image_model", "") or ""),
                    placeholder="gpt-image-2",
                    key="openai_image_model_input",
                )
                set_runtime_config(
                    "app", "openai_image_model", openai_image_model.strip()
                )
                # Only reference values ​​are shown and OpenAI official endpoints are not written as the default configuration.
                # There is no uniform value for the Base URL and model ID of compatible services; leaving blank will not
                # If the user mistakenly connects to the official payment interface without knowing it, the old configuration will not be overwritten.
                st.caption(tr("OpenAI Image Configuration Example"))

                with st.expander(
                    tr("OpenAI Image Advanced Settings"), expanded=False
                ):
                    openai_image_size = st.text_input(
                        tr("OpenAI Image Size"),
                        value=str(config.app.get("openai_image_size", "") or ""),
                        placeholder="1024x1536",
                        help=tr("OpenAI Image Size Help"),
                        key="openai_image_size_input",
                    )
                    set_runtime_config(
                        "app", "openai_image_size", openai_image_size.strip()
                    )

                    openai_image_prompt_template = st.text_input(
                        tr("OpenAI Image Prompt Template"),
                        value=str(
                            config.app.get("openai_image_prompt_template", "") or ""
                        ),
                        placeholder="cinematic photo of {term}, photorealistic",
                        help=tr("OpenAI Image Prompt Template Help"),
                        key="openai_image_prompt_template_input",
                    )
                    set_runtime_config(
                        "app",
                        "openai_image_prompt_template",
                        openai_image_prompt_template.strip(),
                    )

