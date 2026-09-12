"""
Subtitle and video title style presets inspired by trending viral short-form content
(TikTok, YouTube Shorts, Instagram Reels, CapCut, Alex Hormozi, MrBeast).
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, Iterable, Optional


HIGHLIGHT_OPEN = "{{active}}"
HIGHLIGHT_CLOSE = "{{/active}}"
SUPPORTED_SUBTITLE_DISPLAY_MODES = (
    "sentence",
    "word_by_word",
    "two_words",
    "three_words",
    "progressive",
    "karaoke",
)


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
        "subtitle_display_mode": "sentence",
        "subtitle_position": "bottom",
        "highlight_color": "#FFE600",
    },
    "binance_karaoke": {
        "id": "binance_karaoke",
        "name": "Binance Gold Karaoke",
        "description": "Clean white captions with a Binance-gold active word",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 64,
        "stroke_color": "#050505",
        "stroke_width": 4.0,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "scale_up",
        "subtitle_casing": "as_is",
        "subtitle_display_mode": "karaoke",
        "subtitle_position": "two_thirds_bottom",
        "highlight_color": "#F0B90B",
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
        "subtitle_display_mode": "two_words",
        "subtitle_position": "two_thirds_bottom",
        "highlight_color": "#FFE814",
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
        "subtitle_display_mode": "three_words",
        "subtitle_position": "center",
        "highlight_color": "#FFF200",
    },
    "mrbeast": {
        "id": "mrbeast",
        "name": "MrBeast Bold",
        "description": "Ultra-bold white captions with heavy black stroke and spring animation",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 65,
        "stroke_color": "#000000",
        "stroke_width": 5.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "uppercase",
        "subtitle_display_mode": "two_words",
        "subtitle_position": "center",
        "highlight_color": "#7DFF3A",
    },
    "capcut_box": {
        "id": "capcut_box",
        "name": "CapCut Dark Pill",
        "description": "Clean white text inside a rounded semi-transparent dark plate",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 55,
        "stroke_color": "#000000",
        "stroke_width": 0.0,
        "subtitle_background_enabled": True,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": True,
        "subtitle_animation": "fade",
        "subtitle_casing": "as_is",
        "subtitle_display_mode": "progressive",
        "subtitle_position": "bottom",
        "highlight_color": "#FFFFFF",
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
        "subtitle_display_mode": "sentence",
        "subtitle_position": "bottom",
        "highlight_color": "#F8F9FA",
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
        "subtitle_display_mode": "word_by_word",
        "subtitle_position": "center",
        "highlight_color": "#00F0FF",
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
        "subtitle_display_mode": "two_words",
        "subtitle_position": "center",
        "highlight_color": "#FF3B30",
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
        "subtitle_display_mode": "sentence",
        "subtitle_position": "bottom",
        "highlight_color": "#FFD700",
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
        "subtitle_display_mode": "word_by_word",
        "subtitle_position": "center",
        "highlight_color": "#FFDE59",
    },
    "barbie_pink": {
        "id": "barbie_pink",
        "name": "Viral Barbie Pink",
        "description": "Vivid hot pink text with dark stroke and bouncy spring",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FF2A85",
        "font_size": 62,
        "stroke_color": "#000000",
        "stroke_width": 3.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "uppercase",
        "subtitle_display_mode": "two_words",
        "subtitle_position": "center",
        "highlight_color": "#FF2A85",
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
        "subtitle_display_mode": "sentence",
        "subtitle_position": "bottom",
        "highlight_color": "#FFF3B0",
    },
    "tiktok_karaoke": {
        "id": "tiktok_karaoke",
        "name": "TikTok Lime Karaoke",
        "description": "White creator captions with a vivid lime active word",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 64,
        "stroke_color": "#000000",
        "stroke_width": 4.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "pop_spring",
        "subtitle_casing": "as_is",
        "subtitle_display_mode": "karaoke",
        "subtitle_position": "center",
        "highlight_color": "#B8FF20",
    },
    "podcast_karaoke": {
        "id": "podcast_karaoke",
        "name": "Podcast Blue Focus",
        "description": "Readable podcast captions with a crisp blue active word",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 58,
        "stroke_color": "#111827",
        "stroke_width": 3.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "none",
        "subtitle_casing": "as_is",
        "subtitle_display_mode": "karaoke",
        "subtitle_position": "bottom",
        "highlight_color": "#42A5FF",
    },
    "news_reel": {
        "id": "news_reel",
        "name": "News Reel Stack",
        "description": "Fast three-word headlines on a compact dark plate",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 60,
        "stroke_color": "#000000",
        "stroke_width": 0.0,
        "subtitle_background_enabled": True,
        "subtitle_background_color": "#101114",
        "rounded_subtitle_background": True,
        "subtitle_animation": "slide_up",
        "subtitle_casing": "uppercase",
        "subtitle_display_mode": "three_words",
        "subtitle_position": "two_thirds_bottom",
        "highlight_color": "#FFDF3E",
    },
    "creator_clean": {
        "id": "creator_clean",
        "name": "Creator Clean Reveal",
        "description": "Soft white captions that build with the speaker",
        "font_name": "BeVietnamPro-Bold.ttf",
        "text_fore_color": "#FFFFFF",
        "font_size": 56,
        "stroke_color": "#171717",
        "stroke_width": 2.5,
        "subtitle_background_enabled": False,
        "subtitle_background_color": "#000000",
        "rounded_subtitle_background": False,
        "subtitle_animation": "fade",
        "subtitle_casing": "as_is",
        "subtitle_display_mode": "progressive",
        "subtitle_position": "bottom",
        "highlight_color": "#FFFFFF",
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
    protected = text.replace(HIGHLIGHT_OPEN, "\x00").replace(HIGHLIGHT_CLOSE, "\x01")
    if casing == "uppercase":
        protected = protected.upper()
    elif casing == "lowercase":
        protected = protected.lower()
    elif casing == "capitalize":
        protected = protected.title()
    return protected.replace("\x00", HIGHLIGHT_OPEN).replace("\x01", HIGHLIGHT_CLOSE)


def _ends_phrase(text: str) -> bool:
    """Return whether a word token ends with sentence punctuation."""
    return bool(text) and unicodedata.category(text.rstrip()[-1]).startswith("P")


def _needs_space(previous: str, current: str) -> bool:
    """Keep Latin words readable without inserting spaces into CJK text."""
    if not previous or not current:
        return False
    left, right = previous[-1], current[0]
    if unicodedata.category(right).startswith("P"):
        return False
    if unicodedata.category(left) in {"Ps", "Pi"}:
        return False
    return not (
        unicodedata.east_asian_width(left) in {"W", "F"}
        or unicodedata.east_asian_width(right) in {"W", "F"}
    )


def _join_words(words: Iterable[str]) -> str:
    joined = ""
    for word in words:
        word = str(word).strip()
        if not word:
            continue
        if _needs_space(joined, word):
            joined += " "
        joined += word
    return joined


def _join_marked_words(words: list[str], active_index: int) -> str:
    joined = ""
    previous = ""
    for index, word in enumerate(words):
        if _needs_space(previous, word):
            joined += " "
        joined += (
            f"{HIGHLIGHT_OPEN}{word}{HIGHLIGHT_CLOSE}"
            if index == active_index
            else word
        )
        previous = word
    return joined


def _group_word_cues(cues: list[tuple], size: int) -> list[list[tuple]]:
    groups: list[list[tuple]] = []
    current: list[tuple] = []
    for cue in cues:
        current.append(cue)
        if len(current) >= size or _ends_phrase(str(cue[1])):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def build_display_cues(cues: Iterable[tuple], mode: str) -> list[tuple]:
    """Build the visual timing pattern for word-level subtitle cues."""
    items = list(cues)
    if mode in {"sentence", "word_by_word"}:
        return items

    group_size = 2 if mode == "two_words" else 3 if mode == "three_words" else 4
    groups = _group_word_cues(items, group_size)
    if mode in {"two_words", "three_words"}:
        return [
            ((group[0][0][0], group[-1][0][1]), _join_words(cue[1] for cue in group))
            for group in groups
        ]

    rendered: list[tuple] = []
    for group in groups:
        words = [str(cue[1]).strip() for cue in group]
        for index, cue in enumerate(group):
            visible_words = words[: index + 1] if mode == "progressive" else words.copy()
            if mode == "karaoke":
                phrase = _join_marked_words(visible_words, index)
            else:
                phrase = _join_words(visible_words)
            rendered.append((cue[0], phrase))
    return rendered
