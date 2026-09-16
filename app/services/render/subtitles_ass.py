"""Build the ASS document that ffmpeg's ``ass`` filter burns into the video.

One builder feeds both the on-disk ``subtitle.ass`` written at the subtitle
stage and the in-memory document the render stage regenerates, so what the
user can edit on disk and what the burn-in path renders are the same thing.

Cues come from :mod:`app.services.render.subtitle_cues`: at most two lines of
32 characters, never split inside a word, held on screen for at least 0.7 s.
Every word-level display mode renders the active word in the preset's
highlight colour with one Dialogue event per word; sentence mode renders one
event per cue.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Optional

from loguru import logger

from app.services import subtitle_styles
from app.services.render import subtitle_cues
from app.utils import utils

_DEFAULT_SHADOW = 1
# Usable width for a subtitle line as a fraction of the canvas: the 9:16
# platform UIs cover roughly the outer 5 % on each side.
_USABLE_WIDTH_RATIO = 0.88


def ass_color(color: str, fallback: str) -> str:
    """Convert a web ``#RRGGBB`` colour to ASS ``&H00BBGGRR``."""
    value = (
        color
        if isinstance(color, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", color)
        else fallback
    )
    return f"&H00{value[5:7]}{value[3:5]}{value[1:3]}".upper()


def ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def escape_ass_text(text: str) -> str:
    """Escape user text so it cannot be interpreted as ASS override tags."""
    return (
        str(text or "")
        .replace("\\", "＼")
        .replace("{", "｛")
        .replace("}", "｝")
        .replace("\r\n", "\n")
        .replace("\n", r"\N")
    )


def resolve_position(
    position: str, custom_position: float, width: int, height: int
) -> tuple[int, float, float]:
    """Map ``subtitle_position`` to (alignment, x, y) inside the 9:16 safe zone.

    Vertical-video UIs (TikTok, Reels, Shorts) cover the top ~13 % and the
    bottom ~22 % of the frame, so ``bottom`` sits at 78 % and ``top`` at 13 %;
    ``two_thirds_bottom`` is centred on 68 % of the height.
    """
    if position == "bottom":
        return 2, width / 2, height * 0.78
    if position == "top":
        return 8, width / 2, height * 0.13
    if position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
        return 5, width / 2, height * 0.68
    if position == "custom":
        percent = max(0.0, min(100.0, float(custom_position or 70.0)))
        return 5, width / 2, height * percent / 100
    return 5, width / 2, height / 2


def resolve_animation(animation: str) -> str:
    """Translate ``subtitle_animation`` to an ASS ``\\t``-style override tag."""
    if animation in ("scale_up", "zoom_in", "punch"):
        return r"\fscx65\fscy65\t(0,180,0.5,\fscx100\fscy100)"
    if animation in ("pop_spring", "spring", "pop"):
        return (
            r"\fscx5\fscy5\t(0,100,0.5,\fscx135\fscy135)"
            r"\t(100,180,0.5,\fscx100\fscy100)"
        )
    if animation in ("fade", "fade_in"):
        return r"\fad(180,0)"
    return ""


def normalise_style_override(raw: str) -> str:
    """Strip a leading ``[V4+ Styles]`` header from a pasted style block."""
    text = (raw or "").strip()
    if not text:
        return ""
    header = "[V4+ Styles]"
    if text.lower().startswith(header.lower()):
        text = text[len(header):].lstrip("\r\n")
    return text


def font_family_name(font_path: str, fallback_basename: str) -> str:
    """Return the family name libass matches on (``Anton``, not ``Anton-Regular``).

    libass resolves ``Fontname`` through fontconfig by family; the file stem
    silently falls back to DejaVu, so read the real family from the font.
    """
    if font_path and os.path.isfile(font_path):
        try:
            from PIL import ImageFont

            family = ImageFont.truetype(font_path, 24).getname()[0]
            if family:
                return family
        except Exception as exc:  # pragma: no cover - depends on the font file
            logger.debug(f"could not read font family from {font_path}: {exc}")
    return os.path.splitext(os.path.basename(fallback_basename))[0]


def _pixel_fit_checker(font_path: str, font_size: int, usable_width: int) -> Optional[Callable[[str], bool]]:
    """Return a Pillow-backed ``fits(line)`` check, or None when the font is unreadable."""
    if not font_path or not os.path.isfile(font_path):
        return None
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(font_path, font_size)
    except Exception as exc:  # pragma: no cover - depends on the host font stack
        logger.debug(f"subtitle width measurement unavailable: {exc}")
        return None

    def fits(line: str) -> bool:
        try:
            return font.getlength(line) <= usable_width
        except Exception:
            return True

    return fits


def build_ass_document(
    *,
    timed_cues: Iterable[tuple[tuple[float, float], str]],
    params,
    font_path: str,
    width: int,
    height: int,
    max_duration: Optional[float] = None,
    words_json_path: Optional[str] = None,
) -> str:
    """Return the full ASS text for the cues, or ``""`` when there is nothing to show."""
    timed_cues = [
        ((float(start), float(end)), str(text))
        for (start, end), text in timed_cues
        if float(end) > float(start) and str(text).strip()
    ]
    if max_duration is not None:
        timed_cues = [cue for cue in timed_cues if cue[0][0] < max_duration]
    if not timed_cues:
        return ""

    preset = subtitle_styles.get_subtitle_preset(
        getattr(params, "subtitle_style_preset", "custom") or "custom"
    ) or {}
    normal_color = ass_color(getattr(params, "text_fore_color", "") or "", "#FFFFFF")
    stroke_color = ass_color(getattr(params, "stroke_color", "") or "", "#000000")
    highlight_color = ass_color(
        (preset.get("highlight_color") if isinstance(preset, dict) else "") or "",
        "#FFE600",
    )
    resolved_font = font_path if font_path and os.path.isfile(font_path) else os.path.join(
        utils.font_dir(), os.path.basename(getattr(params, "font_name", "") or "")
    )
    font_name = font_family_name(resolved_font, os.path.basename(font_path or getattr(params, "font_name", "") or "STHeitiMedium.ttc"))
    font_size = int(getattr(params, "font_size", 60) or 60)
    ass_font_size = max(1, int(round(font_size * 1.15)))
    stroke_width = max(0, int(round(float(getattr(params, "stroke_width", 0) or 0))))
    margin_x = max(10, int(width * 0.05))
    casing = getattr(params, "subtitle_casing", "as_is") or "as_is"

    background_color = (getattr(params, "subtitle_ass_background_color", "") or "") or (
        "#000000"
        if bool(getattr(params, "subtitle_background_enabled", False))
        or bool(getattr(params, "rounded_subtitle_background", False))
        else ""
    )
    back_color = ass_color(background_color, "#000000") if background_color else "&H00000000"

    custom_style = normalise_style_override(getattr(params, "subtitle_ass_style_override", "") or "")
    if custom_style:
        style_block = custom_style
    else:
        style_block = (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font_name},{ass_font_size},{normal_color},"
            f"{normal_color},{stroke_color},{back_color},-1,0,0,0,100,100,"
            f"0,0,1,{stroke_width},{_DEFAULT_SHADOW},5,{margin_x},{margin_x},0,1"
        )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        f"{style_block}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )

    alignment, x, y = resolve_position(
        getattr(params, "subtitle_position", "bottom") or "bottom",
        float(getattr(params, "custom_position", 70.0) or 70.0),
        width,
        height,
    )
    animation = getattr(params, "subtitle_animation", "none") or "none"
    animation_tag = resolve_animation(animation)
    is_slide = animation in ("slide_up", "rise")
    shift = max(15, int(round(font_size * 0.3)))
    if is_slide:
        position_tag = f"\\an{alignment}\\move({x:.0f},{y + shift:.0f},{x:.0f},{y:.0f},0,180)"
    else:
        position_tag = f"\\an{alignment}\\pos({x:.0f},{y:.0f})"

    static_overrides = ""
    if float(getattr(params, "subtitle_ass_shadow", 0) or 0) > 0:
        static_overrides += f"\\shad{int(float(getattr(params, 'subtitle_ass_shadow')))}"
    if float(getattr(params, "subtitle_ass_blur", 0) or 0) > 0:
        static_overrides += f"\\blur{int(float(getattr(params, 'subtitle_ass_blur')))}"
    if float(getattr(params, "subtitle_ass_rotation", 0) or 0) % 360 != 0:
        static_overrides += f"\\frz{int(float(getattr(params, 'subtitle_ass_rotation')) % 360)}"
    raw_overrides = (getattr(params, "subtitle_ass_event_overrides", "") or "").strip()

    # Word timing: aligned script words when the whisper stage produced them,
    # otherwise the SRT cues spread over their words.
    display_mode = getattr(params, "subtitle_display_mode", "sentence") or "sentence"
    karaoke = display_mode != "sentence"
    words = subtitle_cues.load_words_json(words_json_path) if words_json_path else None
    hard_boundaries: set[int] = set()
    if not words:
        words, hard_boundaries = subtitle_cues.words_from_timed_cues(timed_cues)
        if not karaoke:
            # Sentence mode shows the cues as authored in the SRT: every
            # source cue ends a display cue, even a one-word one.
            count = 0
            for _timing, text in timed_cues:
                count += len(subtitle_cues.script_tokens(text.replace("\n", " ")))
                if count:
                    hard_boundaries.add(count - 1)
    if max_duration is not None:
        words = [w for w in words if w.start < max_duration]
    if not words:
        return ""

    fits = _pixel_fit_checker(resolved_font, ass_font_size, int(width * _USABLE_WIDTH_RATIO))
    cues = subtitle_cues.build_cues(words, hard_boundaries=hard_boundaries, fits=fits)
    if not cues:
        return ""

    def render_lines(cue: subtitle_cues.Cue, active: Optional[int]) -> str:
        rendered_lines = []
        for line in cue.lines:
            pieces = []
            for word_index in line:
                text = escape_ass_text(
                    subtitle_styles.apply_text_casing(cue.words[word_index].text, casing)
                )
                if active is not None and word_index == active:
                    pieces.append(f"{{\\c{highlight_color}&}}{text}{{\\c{normal_color}&}}")
                else:
                    pieces.append(text)
            rendered_lines.append(" ".join(pieces))
        return r"\N".join(rendered_lines)

    def event(start: float, end: float, text: str, animated: bool) -> str:
        if max_duration is not None:
            end = min(end, max_duration)
        overrides = "".join(
            tag for tag in ((animation_tag if animated else ""), static_overrides, raw_overrides) if tag
        )
        return (
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,"
            f"{{{position_tag}{overrides}}}{text}"
        )

    events: list[str] = []
    for cue in cues:
        if not karaoke:
            if cue.end > cue.start:
                events.append(event(cue.start, cue.end, render_lines(cue, None), True))
            continue
        for order, (word_index, start, end) in enumerate(subtitle_cues.highlight_spans(cue)):
            if end <= start:
                continue
            events.append(event(start, end, render_lines(cue, word_index), order == 0))
    return header + "\n".join(events) + "\n" if events else ""
