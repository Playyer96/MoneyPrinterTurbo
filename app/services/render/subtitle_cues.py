"""Word timing, script alignment, and cue segmentation for burned subtitles.

The pipeline gets word timing from three places, in decreasing quality:

1. Whisper word timestamps aligned onto the script text (``align_script_words``),
2. Edge TTS word boundaries (already one word per SRT cue),
3. A per-sentence estimate from providers with no timing at all (OmniVoice,
   Gemini, ...), which ``words_from_timed_cues`` spreads over the words by
   character weight.

Whatever the source, ``build_cues`` turns the words into cues of at most
``max_lines`` lines of ``max_chars`` characters, broken at punctuation and
pauses and never inside a word, with a minimum and maximum on-screen time.
The ASS builder renders each cue with the active word highlighted.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

MAX_CHARS_PER_LINE = 32
MAX_LINES_PER_CUE = 2
MIN_CUE_SECONDS = 0.7
MAX_CUE_SECONDS = 6.0
# A silence longer than this between two words is a natural place to cut.
PAUSE_BREAK_SECONDS = 0.5
# Consecutive cues keep this much air between them so libass never shows two.
CUE_GAP_SECONDS = 0.04

_STRONG_PUNCTUATION = ".!?…:;。！？"
_SOFT_PUNCTUATION = ",，、"
_TOKEN_STRIP_RE = re.compile(r"[^0-9a-z]+")


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    words: list[Word]
    # Word indexes (into ``words``) per rendered line.
    lines: list[list[int]] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    def line_texts(self) -> list[str]:
        return [" ".join(self.words[i].text for i in line) for line in self.lines]

    @property
    def text(self) -> str:
        return "\n".join(self.line_texts())


def normalize_token(text: str) -> str:
    """Lowercase, strip accents and punctuation, so ``Qué,`` matches ``que``."""
    decomposed = unicodedata.normalize("NFD", str(text or "").lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _TOKEN_STRIP_RE.sub("", stripped)


def script_tokens(script: str) -> list[str]:
    """Split a script into spoken tokens, keeping punctuation attached."""
    return [token for token in re.split(r"\s+", str(script or "").strip()) if token]


def words_from_timed_cues(
    timed_cues: Iterable[tuple[tuple[float, float], str]],
) -> tuple[list[Word], set[int]]:
    """Explode SRT cues into words.

    A multi-word cue has no timing per word, so its span is distributed by
    character weight. Returns the words plus the set of word indexes after
    which a multi-word source cue ended: the segmenter never merges across
    those, because a sentence-level SRT already encodes the author's cuts.
    """
    words: list[Word] = []
    hard_boundaries: set[int] = set()
    for (start, end), text in timed_cues:
        tokens = script_tokens(str(text).replace("\n", " "))
        if not tokens:
            continue
        start = float(start)
        end = max(float(end), start)
        total = sum(len(token) for token in tokens) or 1
        cursor = start
        for index, token in enumerate(tokens):
            if index == len(tokens) - 1:
                word_end = end
            else:
                word_end = cursor + (end - start) * len(token) / total
            words.append(Word(token, cursor, max(word_end, cursor)))
            cursor = word_end
        if len(tokens) > 1:
            hard_boundaries.add(len(words) - 1)
    return words, hard_boundaries


def align_script_words(script: str, spoken: Sequence[Word]) -> list[Word]:
    """Give every script token a start/end taken from recognised speech.

    Tokens are matched by normalised text with ``difflib``; runs of script
    tokens that the recogniser missed (numbers read as words, mumbled
    syllables) are spread over the time between their matched neighbours by
    character weight, so the output always covers every script word in order.
    """
    tokens = script_tokens(script)
    if not tokens:
        return []
    if not spoken:
        return [Word(token, 0.0, 0.0) for token in tokens]

    script_keys = [normalize_token(token) for token in tokens]
    spoken_keys = [normalize_token(word.text) for word in spoken]
    matcher = difflib.SequenceMatcher(None, script_keys, spoken_keys, autojunk=False)
    timing: list[Optional[tuple[float, float]]] = [None] * len(tokens)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            word = spoken[block.b + offset]
            timing[block.a + offset] = (float(word.start), float(word.end))

    first_start = float(spoken[0].start)
    last_end = float(spoken[-1].end)
    aligned: list[Word] = []
    index = 0
    while index < len(tokens):
        if timing[index] is not None:
            start, end = timing[index]
            aligned.append(Word(tokens[index], start, end))
            index += 1
            continue
        # Unmatched run: interpolate between the previous end and next start.
        run_start = index
        while index < len(tokens) and timing[index] is None:
            index += 1
        previous_end = aligned[-1].end if aligned else first_start
        next_start = timing[index][0] if index < len(tokens) else last_end
        span_start = previous_end
        span_end = max(next_start, span_start)
        run_tokens = tokens[run_start:index]
        total = sum(len(token) for token in run_tokens) or 1
        cursor = span_start
        for position, token in enumerate(run_tokens):
            if position == len(run_tokens) - 1:
                word_end = span_end
            else:
                word_end = cursor + (span_end - span_start) * len(token) / total
            aligned.append(Word(token, cursor, word_end))
            cursor = word_end
    return aligned


def load_words_json(path: str) -> Optional[list[Word]]:
    """Read the ``.words.json`` sidecar written by the whisper stage."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    entries = payload.get("words") if isinstance(payload, dict) else None
    if not entries:
        return None
    words = [
        Word(str(entry.get("w", "")), float(entry.get("s", 0.0)), float(entry.get("e", 0.0)))
        for entry in entries
        if str(entry.get("w", "")).strip()
    ]
    return words or None


def write_aligned_words_json(path: str, script: str) -> bool:
    """Replace the recogniser's words in the sidecar with script tokens carrying
    the aligned timing, and flag the file so readers know it is script text."""
    spoken = load_words_json(path)
    if not spoken:
        return False
    aligned = align_script_words(script, spoken)
    if not aligned:
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        payload = {"version": 1}
    payload["words"] = [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3)} for w in aligned]
    payload["aligned_to_script"] = True
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return True


def _ends_with(text: str, chars: str) -> bool:
    stripped = text.rstrip("\"'”’)»]")
    return bool(stripped) and stripped[-1] in chars


def wrap_words(
    texts: Sequence[str],
    *,
    max_chars: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES_PER_CUE,
    fits: Optional[Callable[[str], bool]] = None,
) -> Optional[list[list[int]]]:
    """Split word indexes into balanced lines, or None when they cannot fit.

    ``fits`` is an optional pixel-width check (a Pillow measurement) that is
    applied on top of the character cap, so a wide font cannot overflow the
    canvas even when the character count says it fits.
    """
    if not texts:
        return None

    def line_ok(indexes: list[int]) -> bool:
        line = " ".join(texts[i] for i in indexes)
        if len(line) > max_chars:
            return False
        return fits(line) if fits is not None else True

    indexes = list(range(len(texts)))
    if line_ok(indexes):
        return [indexes]
    if max_lines < 2:
        return None
    # Try every split point for two lines and keep the most balanced one.
    best: Optional[list[list[int]]] = None
    best_score = None
    for cut in range(1, len(indexes)):
        first, second = indexes[:cut], indexes[cut:]
        if not (line_ok(first) and line_ok(second)):
            continue
        # Prefer a shorter second line (pyramid shape reads better) but keep
        # both lines close in length.
        len_first = len(" ".join(texts[i] for i in first))
        len_second = len(" ".join(texts[i] for i in second))
        score = abs(len_first - len_second) + (2 if len_second > len_first else 0)
        if best_score is None or score < best_score:
            best, best_score = [first, second], score
    return best


def build_cues(
    words: Sequence[Word],
    *,
    hard_boundaries: Optional[set[int]] = None,
    max_chars: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES_PER_CUE,
    min_duration: float = MIN_CUE_SECONDS,
    max_duration: float = MAX_CUE_SECONDS,
    pause_break: float = PAUSE_BREAK_SECONDS,
    fits: Optional[Callable[[str], bool]] = None,
) -> list[Cue]:
    """Group timed words into readable cues.

    A cue closes when the next word would not fit, when it would run past
    ``max_duration``, at a hard boundary, after a pause, or after
    punctuation once the cue is long enough to be readable on its own.
    """
    hard_boundaries = hard_boundaries or set()
    words = [w for w in words if w.text.strip()]
    cues: list[Cue] = []
    current: list[int] = []

    def close_current() -> None:
        if not current:
            return
        texts = [words[i].text for i in current]
        lines = wrap_words(texts, max_chars=max_chars, max_lines=max_lines, fits=fits)
        if lines is None:
            # ponytail: a single word longer than a line cannot be wrapped;
            # it goes out on its own line and libass squeezes it. Hyphenation
            # would be the upgrade if scripts ever contain such words.
            lines = [[i] for i in range(len(texts))][:max_lines] or [[0]]
        cue_words = [words[i] for i in current]
        cues.append(
            Cue(
                words=cue_words,
                lines=lines,
                start=cue_words[0].start,
                end=max(cue_words[-1].end, cue_words[0].start),
            )
        )
        current.clear()

    for index, word in enumerate(words):
        if current:
            candidate = [words[i].text for i in current] + [word.text]
            if wrap_words(candidate, max_chars=max_chars, max_lines=max_lines, fits=fits) is None:
                close_current()
            elif word.end - words[current[0]].start > max_duration:
                close_current()
        current.append(index)

        is_last = index == len(words) - 1
        if is_last:
            close_current()
            break
        duration = word.end - words[current[0]].start
        next_word = words[index + 1]
        gap = next_word.start - word.end
        if index in hard_boundaries or gap > pause_break:
            close_current()
        elif _ends_with(word.text, _STRONG_PUNCTUATION) and duration >= min_duration:
            close_current()
        elif _ends_with(word.text, _SOFT_PUNCTUATION) and duration >= min_duration:
            # Only cut on a comma when the cue is already reasonably full;
            # otherwise short clauses produce a flicker of tiny cues.
            chars = len(" ".join(words[i].text for i in current))
            if chars >= int(max_chars * max_lines * 0.45):
                close_current()

    # Timing pass: hold short cues on screen, never into the next cue.
    for position, cue in enumerate(cues):
        next_start = cues[position + 1].start if position + 1 < len(cues) else None
        ceiling = (next_start - CUE_GAP_SECONDS) if next_start is not None else float("inf")
        if cue.end - cue.start < min_duration:
            cue.end = min(cue.start + min_duration, max(ceiling, cue.start))
        if next_start is not None and cue.end > ceiling:
            cue.end = max(cue.start, ceiling)
    return [cue for cue in cues if cue.end > cue.start]


def highlight_spans(cue: Cue) -> list[tuple[int, float, float]]:
    """Per-word (index, start, end) windows that tile the cue without gaps.

    The highlight moves to a word when that word starts and stays until the
    next word starts, so silences inside the cue keep the previous word lit
    instead of flashing to nothing.
    """
    spans: list[tuple[int, float, float]] = []
    for index, word in enumerate(cue.words):
        start = cue.start if index == 0 else max(word.start, spans[-1][2])
        end = cue.words[index + 1].start if index + 1 < len(cue.words) else cue.end
        end = min(max(end, start), cue.end)
        spans.append((index, start, end))
    # Make sure the last span reaches the cue end even when word timing is short.
    if spans:
        index, start, _ = spans[-1]
        spans[-1] = (index, start, cue.end)
    return [span for span in spans if span[2] > span[1]]
