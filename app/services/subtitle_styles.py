"""
Subtitle and video title style presets inspired by trending viral short-form content
(TikTok, YouTube Shorts, Instagram Reels, CapCut, Alex Hormozi, MrBeast).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Curated viral subtitle presets
SUBTITLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "custom": {
        "id": "custom",
        "name": "Custom",
        "description": "Manual custom settings",
        "font_name": "MicrosoftYaHeiBold.ttc",
        "text_fore_color": "#FFFFFF",
        "font_size": 60,
        "stroke_color": "#000000",
        "stroke_width": 1.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "none",
        "subtitle_casing": "as_is",
    },
    "tiktok_yellow": {
        "id": "tiktok_yellow",
        "name": "TikTok Viral Yellow",
        "description": "Punchy bright yellow with heavy black outline and bounce",
        "font_name": "Anton-Regular.ttf",
        "text_fore_color": "#FFE814",
        "font_size": 65,
        "stroke_color": "#000000",
        "stroke_width": 4.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "uppercase",
    },
    "hormozi": {
        "id": "hormozi",
        "name": "Alex Hormozi Punch",
        "description": "Ultra-bold energetic yellow captions with punchy scale effect",
        "font_name": "Anton-Regular.ttf",
        "text_fore_color": "#FFF200",
        "font_size": 70,
        "stroke_color": "#000000",
        "stroke_width": 5.0,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "scale_up",
        "subtitle_casing": "uppercase",
    },
    "mrbeast": {
        "id": "mrbeast",
        "name": "MrBeast Bold",
        "description": "Ultra-bold white captions with heavy black stroke and spring animation",
        "font_name": "Montserrat-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 65,
        "stroke_color": "#000000",
        "stroke_width": 5.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "uppercase",
    },
    "capcut_box": {
        "id": "capcut_box",
        "name": "CapCut Dark Pill",
        "description": "Clean white text inside a rounded semi-transparent dark plate",
        "font_name": "Montserrat-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 55,
        "stroke_color": "#000000",
        "stroke_width": 0.0,
        "subtitle_background_enabled": True,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": True,
        "subtitle_animation": "fade",
        "subtitle_casing": "as_is",
    },
    "minimalist_clean": {
        "id": "minimalist_clean",
        "name": "Minimalist Clean",
        "description": "Elegant ivory text with subtle outline and smooth slide up",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#F8F9FA",
        "font_size": 52,
        "stroke_color": "#000000",
        "stroke_width": 1.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "slide_up",
        "subtitle_casing": "as_is",
    },
    "cyber_neon": {
        "id": "cyber_neon",
        "name": "Cyber Neon Aqua",
        "description": "Electric cyan glowing text with black outline and energy shake",
        "font_name": "Anton-Regular.ttf",
        "text_fore_color": "#00F0FF",
        "font_size": 65,
        "stroke_color": "#000000",
        "stroke_width": 4.0,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#0A0A1A",
        "rounded_subtitle_background": False,
        "subtitle_animation": "shake",
        "subtitle_casing": "uppercase",
    },
    "fire_alert": {
        "id": "fire_alert",
        "name": "Fire & Alert Red",
        "description": "Vibrant fire-red bold text with heavy stroke for urgent hooks",
        "font_name": "Anton-Regular.ttf",
        "text_fore_color": "#FF3B30",
        "font_size": 65,
        "stroke_color": "#000000",
        "stroke_width": 4.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "uppercase",
    },
    "golden_luxury": {
        "id": "golden_luxury",
        "name": "Golden Luxury",
        "description": "Warm champagne gold text on sleek rounded dark plate",
        "font_name": "BebasNeue-Regular.ttf",
        "text_fore_color": "#FFD700",
        "font_size": 68,
        "stroke_color": "#000000",
        "stroke_width": 2.0,
        "subtitle_background_enabled": True,
        "subtitle_background_color": "#111111",
        "rounded_subtitle_background": True,
        "subtitle_animation": "slide_up",
        "subtitle_casing": "uppercase",
    },
    "comic_pop": {
        "id": "comic_pop",
        "name": "Comic Pop Humor",
        "description": "Playful comic font in sun yellow with thick outline",
        "font_name": "Bangers-Regular.ttf",
        "text_fore_color": "#FFDE59",
        "font_size": 70,
        "stroke_color": "#000000",
        "stroke_width": 5.0,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "as_is",
    },
    "barbie_pink": {
        "id": "barbie_pink",
        "name": "Viral Barbie Pink",
        "description": "Vivid hot pink text with dark stroke and bouncy spring",
        "font_name": "Montserrat-Bold.ttf",
        "text_fore_color": "#FF2A85",
        "font_size": 62,
        "stroke_color": "#000000",
        "stroke_width": 3.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "uppercase",
    },
    "vintage_film": {
        "id": "vintage_film",
        "name": "Vintage Cinema Warm",
        "description": "Warm retro cream text with gentle fade",
        "font_name": "Charm-Bold.ttf",
        "text_fore_color": "#FFF3B0",
        "font_size": 60,
        "stroke_color": "#2C1D11",
        "stroke_width": 2.0,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "fade",
        "subtitle_casing": "as_is",
    },
}


# Curated viral video title / hook header styles
TITLE_STYLES: Dict[str, Dict[str, Any]] = {
    "tiktok_yellow": {
        "id": "tiktok_yellow",
        "name": "TikTok Yellow Badge",
        "description": "Black text on vibrant yellow rounded pill badge",
        "font_name": "Anton-Regular.ttf",
        "text_color": "#000000",
        "bg_color": "#FFE600",
        "stroke_color": None,
        "stroke_width": 0.0,
        "rounded": True,
        "casing": "uppercase",
    },
    "red_banner": {
        "id": "red_banner",
        "name": "Breaking Red Banner",
        "description": "White text on bold high-impact red banner",
        "font_name": "Anton-Regular.ttf",
        "text_color": "#FFFFFF",
        "bg_color": "#E50914",
        "stroke_color": None,
        "stroke_width": 0.0,
        "rounded": True,
        "casing": "uppercase",
    },
    "capcut_black": {
        "id": "capcut_black",
        "name": "CapCut Dark Pill",
        "description": "Crisp white text on sleek translucent dark pill badge",
        "font_name": "Montserrat-Bold.ttf",
        "text_color": "#FFFFFF",
        "bg_color": "#111111",
        "stroke_color": None,
        "stroke_width": 0.0,
        "rounded": True,
        "casing": "as_is",
    },
    "neon_cyan": {
        "id": "neon_cyan",
        "name": "Neon Cyber Glow",
        "description": "Glowing electric cyan text on deep dark badge",
        "font_name": "Anton-Regular.ttf",
        "text_color": "#00F0FF",
        "bg_color": "#050B14",
        "stroke_color": "#00F0FF",
        "stroke_width": 1.5,
        "rounded": True,
        "casing": "uppercase",
    },
    "minimalist_white": {
        "id": "minimalist_white",
        "name": "Minimalist Bold White",
        "description": "Clean white text with bold black outline",
        "font_name": "Montserrat-Bold.ttf",
        "text_color": "#FFFFFF",
        "bg_color": None,
        "stroke_color": "#000000",
        "stroke_width": 4.0,
        "rounded": False,
        "casing": "as_is",
    },
    "golden_luxury": {
        "id": "golden_luxury",
        "name": "Golden Luxury Card",
        "description": "Metallic champagne gold text on obsidian card",
        "font_name": "BebasNeue-Regular.ttf",
        "text_color": "#FFD700",
        "bg_color": "#1A1608",
        "stroke_color": "#D4AF37",
        "stroke_width": 1.0,
        "rounded": True,
        "casing": "uppercase",
    },
    "comic_punch": {
        "id": "comic_punch",
        "name": "Comic Bang Punch",
        "description": "Punchy yellow comic font with dark stroke",
        "font_name": "Bangers-Regular.ttf",
        "text_color": "#FFDE59",
        "bg_color": "#111111",
        "stroke_color": "#000000",
        "stroke_width": 3.0,
        "rounded": True,
        "casing": "uppercase",
    },
}

SUPPORTED_SUBTITLE_ANIMATIONS = (
    "none",
    "pop_spring",
    "scale_up",
    "fade",
    "slide_up",
    "shake",
)

SUPPORTED_TITLE_POSITIONS = ("top", "center", "bottom", "custom")
SUPPORTED_TITLE_DURATIONS = ("intro", "full")


def get_subtitle_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    """Return preset configuration by ID, or None if unknown."""
    return SUBTITLE_PRESETS.get(preset_id)


def get_title_style(style_id: str) -> Optional[Dict[str, Any]]:
    """Return video title style configuration by ID, or None if unknown."""
    return TITLE_STYLES.get(style_id)


def apply_text_casing(text: str, casing: Optional[str]) -> str:
    """Format subtitle or title string with requested casing."""
    if not text or not casing or casing == "as_is":
        return text
    if casing == "uppercase":
        return text.upper()
    if casing == "lowercase":
        return text.lower()
    if casing == "capitalize":
        return text.title()
    return text
