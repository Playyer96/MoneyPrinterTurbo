"""Cue segmentation and script alignment for burned subtitles.

Each test is the smallest check that fails if the rule regresses: two lines
of 32 characters, breaks only between words, a 0.7 s floor per cue, script
tokens getting real timing from whisper words, and the ASS builder emitting
one highlighted event per word inside the 9:16 safe zone.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.render import subtitle_cues as sc
from app.services.render import subtitles_ass


def _words(text: str, seconds_per_word: float = 0.3, start: float = 0.0) -> list[sc.Word]:
    words = []
    cursor = start
    for token in text.split():
        words.append(sc.Word(token, cursor, cursor + seconds_per_word))
        cursor += seconds_per_word
    return words


SCRIPT = (
    "Hacer caber el gigantesco mapa de Los Santos en apenas quinientos doce "
    "megabytes de memoria RAM parecía una misión imposible. Los ingenieros de "
    "Rockstar se enfrentaron a un hardware muy limitado, lanzado en dos mil cinco."
)


def test_cues_respect_line_and_char_limits_without_splitting_words():
    cues = sc.build_cues(_words(SCRIPT))
    assert cues
    script_tokens = set(SCRIPT.split())
    for cue in cues:
        assert len(cue.lines) <= 2
        for line in cue.line_texts():
            assert len(line) <= 32, line
            for token in line.split():
                assert token in script_tokens, token
    # Every script word appears exactly once, in order.
    rendered = " ".join(word.text for cue in cues for word in cue.words)
    assert rendered == SCRIPT


def test_cue_breaks_after_sentence_punctuation():
    cues = sc.build_cues(_words("Primera frase corta. Segunda frase corta.", seconds_per_word=0.4))
    assert [cue.text for cue in cues] == ["Primera frase corta.", "Segunda frase corta."]


def test_short_cue_is_held_for_minimum_duration_without_touching_next():
    words = [sc.Word("Sí.", 0.0, 0.2), sc.Word("Vamos", 2.0, 2.4), sc.Word("ya.", 2.4, 2.9)]
    cues = sc.build_cues(words)
    assert cues[0].end - cues[0].start >= 0.7
    assert cues[0].end < cues[1].start


def test_multi_word_srt_cues_are_never_merged_but_single_words_are():
    words, boundaries = sc.words_from_timed_cues(
        [((0.0, 1.0), "Una frase entera"), ((1.0, 2.0), "otra frase entera")]
    )
    assert boundaries == {2, 5}
    cues = sc.build_cues(words, hard_boundaries=boundaries)
    assert [cue.text for cue in cues] == ["Una frase entera", "otra frase entera"]

    single, boundaries = sc.words_from_timed_cues([((0.0, 0.3), "Una"), ((0.3, 0.6), "frase")])
    assert boundaries == set()
    assert [cue.text for cue in sc.build_cues(single, hard_boundaries=boundaries)] == ["Una frase"]


def test_align_script_words_uses_recognised_timing_and_interpolates_gaps():
    spoken = [
        sc.Word("hacer", 0.0, 0.3),
        sc.Word("caber", 0.3, 0.6),
        sc.Word("quinientos", 1.0, 1.4),   # script says "512"
        sc.Word("doce", 1.4, 1.7),
        sc.Word("megabytes", 1.7, 2.2),
    ]
    aligned = sc.align_script_words("Hacer caber 512 megabytes", spoken)
    assert [w.text for w in aligned] == ["Hacer", "caber", "512", "megabytes"]
    assert (aligned[0].start, aligned[0].end) == (0.0, 0.3)
    assert aligned[3].start == 1.7
    # The unmatched "512" is spread over the gap between its neighbours.
    assert 0.6 <= aligned[2].start < aligned[2].end <= 1.7


def test_ass_document_highlights_one_word_per_event_inside_safe_zone(tmp_path):
    params = SimpleNamespace(
        subtitle_style_preset="tiktok_yellow",
        text_fore_color="#FFFFFF",
        stroke_color="#000000",
        font_size=60,
        stroke_width=4.0,
        subtitle_position="two_thirds_bottom",
        custom_position=70.0,
        subtitle_animation="pop_spring",
        subtitle_display_mode="karaoke",
        subtitle_casing="as_is",
        font_name="Anton-Regular.ttf",
        subtitle_ass_style_override="",
        subtitle_ass_event_overrides="",
        subtitle_ass_shadow=0.0,
        subtitle_ass_blur=0.0,
        subtitle_ass_rotation=0.0,
        subtitle_ass_background_color="",
    )
    document = subtitles_ass.build_ass_document(
        timed_cues=[((0.0, 1.2), "Hola mundo cruel")],
        params=params,
        font_path="",
        width=1080,
        height=1920,
    )
    events = [line for line in document.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 3
    assert "PlayResX: 1080" in document and "PlayResY: 1920" in document
    assert r"\an5\pos(540,1306)" in events[0]
    assert events[0].count(r"\c&H0014E8FF&") == 1  # only the active word is highlighted
    assert "mundo" in events[1] and r"{\c&H0014E8FF&}mundo" in events[1]
    # Entry animation only on the first word so the line does not re-pop.
    assert r"\fscx5" in events[0] and r"\fscx5" not in events[1]
