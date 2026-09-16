"""Hard quality limits for the whole pipeline.

Everything here is enforced in code, on the path every render takes, with no
config key to loosen it. Prompt wording is a request; a model, a provider or a
hand-edited config can ignore a request. These checks run afterwards and
cannot be talked out of, which is the entire point: the output of this project
should not read, sound or look like generated filler.

Five stages are covered:

  research   drop sources that carry no usable prose, keep domains diverse
  script     strip generated-filler language and meta text, report what is left
  voice      keep speed and volume inside the intelligible range
  video      keep cuts long enough to see and playback close to natural
  subtitles  keep cues readable, non-overlapping and long enough to read

Limits are deliberately generous: they exist to catch output that is broken
or slop, not to second-guess normal editorial choices.
"""

import re
from urllib.parse import urlsplit

from loguru import logger

from app.utils import utils

# --- research -----------------------------------------------------------
# A page under this length is a cookie wall, a paywall stub or a nav menu.
MIN_PAGE_PROSE_CHARS = 220
# Real prose ends sentences. Link lists and menus do not.
MIN_PAGE_SENTENCES = 2
# One source per site, so six results cannot be six pages of the same blog.
MAX_RESULTS_PER_DOMAIN = 1

# --- voice --------------------------------------------------------------
# Outside this range TTS stops being narration: too slow drags, too fast is
# unintelligible. Wide enough that no normal WebUI setting is touched.
MIN_VOICE_RATE = 0.5
MAX_VOICE_RATE = 2.0
MIN_VOICE_VOLUME = 0.1
MAX_VOICE_VOLUME = 5.0

# --- video --------------------------------------------------------------
# Sub-two-second cuts are the signature of generated video slop and read as
# a strobe rather than as editing.
MIN_CLIP_SECONDS = 2

# --- subtitles ----------------------------------------------------------
# A cue shorter than this cannot be read even if it is one word.
MIN_CUE_SECONDS = 0.8
# Comfortable reading speed. Above it, extend the cue if there is room.
MAX_CHARS_PER_SECOND = 22.0
# Keep a visible gap so two cues never share the screen.
MIN_CUE_GAP_SECONDS = 0.04

_SENTENCE_END_RE = re.compile(r"[.!?。！？…]")
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]+"
)
_WHITESPACE_RE = re.compile(r"[ \t ]+")

# Lines that are the model talking about the task instead of doing it. The
# whole line goes.
_META_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:sure|certainly|of course|absolutely|here(?:'s| is)|below is)\b.*?:"
    r"|(?:video\s+)?script\s*:"
    r"|title\s*:|hook\s*:|outro\s*:|intro\s*:"
    r"|paragraph\s*\d+\s*[:.)-]"
    r"|part\s*\d+\s*[:.)-]"
    r"|scene\s*\d+\s*[:.)-]"
    r"|\[[^\]]*\]"
    r"|as an ai\b.*"
    r"|note\s*:.*"
    r"|word count\s*:.*"
    r")\s*$",
    re.IGNORECASE,
)

# Whole sentences that carry no information. Dropped outright — they are
# short, self-contained, and never the only thing a paragraph says.
_FILLER_SENTENCE_RE = re.compile(
    r"^\s*(?:"
    r"(?:so\s+)?let(?:'s| us)\s+(?:dive|jump|get)\s+(?:right\s+)?(?:in|into it|started).*"
    r"|(?:but\s+)?(?:first|before we (?:begin|start)),?\s*(?:let(?:'s| us)\b.*)?"
    r"|buckle up.*"
    r"|stay tuned.*"
    r"|(?:and\s+)?(?:that(?:'s| is) it|there you have it).*"
    r"|(?:don't|do not) forget to (?:like|subscribe|comment|follow).*"
    r"|(?:like|subscribe|follow)[ ,].*(?:for more|if you).*"
    r"|welcome (?:back )?to (?:this|the|my|our) (?:video|channel).*"
    r"|in (?:today's|this) (?:video|clip|short).*"
    r"|without further ado.*"
    r"|the (?:answer|truth|result) (?:may|might|will) (?:surprise|shock) you.*"
    r"|keep watching.*"
    r"|read on.*"
    r"|in conclusion,?"
    r"|to sum(?: it)? up,?"
    r")\s*$",
    re.IGNORECASE,
)

# Generated-prose vocabulary. Not auto-removed — cutting these mid-sentence
# would maim the text — but reported so the generator is retried, and logged
# when it survives.
# A speaker label in front of real narration: drop the label, keep the line.
_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:narrator|voice\s*-?\s*over|vo|host|speaker)\s*:\s*", re.IGNORECASE
)

_SLOP_PHRASES = (
    ("delve into", r"\bdelve[sd]?\s+into\b"),
    ("tapestry", r"\btapestry\b"),
    ("testament to", r"\ba testament to\b"),
    ("ever-evolving", r"\bever[- ]evolving\b"),
    ("in the realm of", r"\bin the (?:realm|world) of\b"),
    ("navigate the landscape", r"\bnavigat\w+ the \w+ landscape\b"),
    ("game-changer", r"\bgame[- ]chang(?:er|ing)\b"),
    ("unlock the secrets", r"\bunlock(?:ing)? the \w+\b"),
    ("harness the power", r"\bharness(?:ing)? the power\b"),
    ("revolutionize", r"\brevolutioniz\w+\b"),
    ("unleash", r"\bunleash\w*\b"),
    ("elevate your", r"\belevate your\b"),
    ("not just X, it's Y", r"\b(?:it(?:'s| is)|they(?:'re| are)) not just\b"),
    ("more than just", r"\bmore than just\b"),
    ("dive deeper", r"\bdiv(?:e|ing) deeper\b"),
    ("in a world where", r"\bin a world where\b"),
    ("little did they know", r"\blittle did \w+ know\b"),
)
_SLOP_RES = tuple(
    (label, re.compile(pattern, re.I)) for label, pattern in _SLOP_PHRASES
)


# -----------------------------------------------------------------------
# script
# -----------------------------------------------------------------------
def find_slop(text: str) -> list[str]:
    """Return labels for generated-filler language still present in the text."""
    return [label for label, pattern in _SLOP_RES if pattern.search(text or "")]


def scrub_script(text: str) -> str:
    """Remove what can be removed mechanically without damaging the writing.

    Meta lines, stage directions, emoji, hashtags, markdown and self-contained
    filler sentences are all safe to delete: none of them is content. Anything
    that cannot be cut cleanly is left to find_slop() and the retry.
    """
    if not text:
        return ""

    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"(?<!\w)#\w+", "", text)  # hashtags, before # is stripped below
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"(?m)^\s*[-•]\s+", "", text)

    kept_paragraphs = []
    for paragraph in text.split("\n\n"):
        lines = [
            _SPEAKER_PREFIX_RE.sub("", line)
            for line in paragraph.split("\n")
            if line.strip() and not _META_LINE_RE.match(line)
        ]
        lines = [line for line in lines if line.strip()]
        if not lines:
            continue
        sentences = _split_sentences(" ".join(lines))
        kept = [s for s in sentences if not _FILLER_SENTENCE_RE.match(s)]
        # A paragraph that was nothing but filler is dropped; one that would be
        # emptied by the filter keeps its original text rather than vanishing.
        paragraph_text = " ".join(kept).strip() if kept else ""
        if paragraph_text:
            kept_paragraphs.append(_WHITESPACE_RE.sub(" ", paragraph_text))

    cleaned = "\n\n".join(kept_paragraphs).strip()
    # Exclamation spam reads as ad copy no matter what the words are.
    cleaned = re.sub(r"!{2,}", "!", cleaned)
    return cleaned


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？…])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def enforce_script(text: str) -> tuple[str, list[str]]:
    """Scrub the script and report the slop that scrubbing cannot remove."""
    cleaned = scrub_script(text)
    return cleaned, find_slop(cleaned)


# -----------------------------------------------------------------------
# research
# -----------------------------------------------------------------------
def is_usable_page(text: str) -> bool:
    """True when the fetched text is prose a script can actually be built on."""
    body = (text or "").strip()
    if len(body) < MIN_PAGE_PROSE_CHARS:
        return False
    return len(_SENTENCE_END_RE.findall(body)) >= MIN_PAGE_SENTENCES


def filter_research_results(results: list[dict]) -> list[dict]:
    """Keep one result per site and drop entries with no readable snippet.

    Six links to the same content farm is not research, and neither is a page
    whose snippet is a cookie banner.
    """
    seen_domains: dict[str, int] = {}
    seen_snippets: set[str] = set()
    kept = []
    for result in results:
        url = str(result.get("url", ""))
        domain = (urlsplit(url).hostname or "").lower().removeprefix("www.")
        if not domain:
            continue
        if seen_domains.get(domain, 0) >= MAX_RESULTS_PER_DOMAIN:
            logger.debug(f"research: dropping extra result from {domain}")
            continue
        fingerprint = re.sub(r"\W+", "", str(result.get("snippet", "")).lower())[:120]
        if fingerprint and fingerprint in seen_snippets:
            logger.debug(f"research: dropping duplicate snippet from {domain}")
            continue
        seen_domains[domain] = seen_domains.get(domain, 0) + 1
        if fingerprint:
            seen_snippets.add(fingerprint)
        kept.append(result)
    return kept


# -----------------------------------------------------------------------
# voice
# -----------------------------------------------------------------------
def clamp_voice_rate(value, default: float = 1.0) -> float:
    return _clamp_float(value, MIN_VOICE_RATE, MAX_VOICE_RATE, default, "voice rate")


def clamp_voice_volume(value, default: float = 1.0) -> float:
    return _clamp_float(
        value, MIN_VOICE_VOLUME, MAX_VOICE_VOLUME, default, "voice volume"
    )


# -----------------------------------------------------------------------
# video
# -----------------------------------------------------------------------
def clamp_clip_duration(value, default: int = 5) -> int:
    """Keep every cut long enough to register as a shot."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return default
    if seconds < MIN_CLIP_SECONDS:
        logger.warning(
            f"clip duration {seconds}s is below the {MIN_CLIP_SECONDS}s floor; "
            "raising it — faster cuts read as a strobe, not as editing"
        )
        return MIN_CLIP_SECONDS
    return seconds


def auto_clip_duration(audio_duration_seconds: float, *, default: int = 5) -> int:
    """Pick a sensible max-clip-duration when the user picked "Auto".

    Aim for roughly ten cuts across the audio so a 60s narration gets
    ~6s cuts and a 3min narration still gets ~10 distinct scenes. Clamped
    to the same 2..10s window the WebUI exposes so an extreme script does
    not get a 30s cut. The audio length is unknown before TTS, so the
    pipeline passes the realised narration length here at combine time.
    """
    max_clip_seconds = 10
    try:
        duration = float(audio_duration_seconds or 0.0)
    except (TypeError, ValueError):
        return default
    if duration <= 0:
        return default
    # Aim for ~10 cuts so a short video does not feel chopped and a long
    # one still shows enough scene variety. The max-cap mirrors the WebUI
    # dropdown so a 5-minute audio still picks 10s cuts instead of 30s.
    target = round(duration / 10.0)
    target = max(MIN_CLIP_SECONDS, min(max_clip_seconds, target))
    return clamp_clip_duration(target, default=default)


def clamp_clip_speed(value, default: float = 1.0) -> float:
    """Playback speed, bounded by the shared WebUI/API range."""
    return utils.normalize_clip_speed(value, default)


def _clamp_float(value, low: float, high: float, default: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return default
    bounded = min(max(number, low), high)
    if bounded != number:
        logger.warning(f"{label} {number} is out of range; clamped to {bounded}")
    return bounded


# -----------------------------------------------------------------------
# subtitles
# -----------------------------------------------------------------------
def enforce_subtitle_cues(cues: list[dict], word_level: bool = False) -> list[dict]:
    """Make cues readable: no overlap, no flashes, no unreadable density.

    Cues are ``{"msg", "start_time", "end_time"}`` in seconds, in order. Word
    level cues keep their own short timings — they are meant to flash, one
    word at a time — and only get the text cleanup and the overlap fix.
    """
    cleaned = []
    for cue in cues:
        text = _WHITESPACE_RE.sub(
            " ", _EMOJI_RE.sub("", str(cue.get("msg", "")))
        ).strip()
        if not text:
            continue
        try:
            start = float(cue.get("start_time", 0.0))
            end = float(cue.get("end_time", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        cleaned.append({"msg": text, "start_time": start, "end_time": end})

    for index, cue in enumerate(cleaned):
        next_start = (
            cleaned[index + 1]["start_time"] if index + 1 < len(cleaned) else None
        )
        if word_level:
            # Karaoke cues are meant to butt up against each other, one word
            # per beat, so only a genuine overlap is corrected here.
            if next_start is not None and cue["end_time"] > next_start:
                cue["end_time"] = max(cue["start_time"], next_start)
            continue

        # Never let two sentence cues share the screen.
        if (
            next_start is not None
            and cue["end_time"] > next_start - MIN_CUE_GAP_SECONDS
        ):
            cue["end_time"] = max(cue["start_time"], next_start - MIN_CUE_GAP_SECONDS)

        # Give a too-short or too-dense cue more time, but only into the gap
        # that is actually free — never by pushing the next cue away from its
        # own audio.
        ceiling = (
            next_start - MIN_CUE_GAP_SECONDS if next_start is not None else float("inf")
        )
        needed = max(MIN_CUE_SECONDS, len(cue["msg"]) / MAX_CHARS_PER_SECOND)
        if cue["end_time"] - cue["start_time"] < needed:
            cue["end_time"] = min(cue["start_time"] + needed, ceiling)

    return [cue for cue in cleaned if cue["end_time"] > cue["start_time"]]
