"""Lock in the intro/outro text resolution introduced when ``intro_enabled``
and ``outro_enabled`` flipped to ON-by-default.

The user-facing complaint was that renders went out with no intro and no
outro even after repeated requests. These tests prove the default text
matches the per-series expectation so the next render carries the right
overlay copy without any UI trip.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.video import _resolve_intro_text, _resolve_outro_text


def _params(**overrides):
    base = dict(
        video_subject="how GTA V was developed",
        video_script="",
        series_enabled=True,
        series_parts=2,
        series_outline=[
            "El desafío técnico imposible de GTA V en PS3 y Xbox 360",
            "El legado histórico de GTA V en la séptima generación",
        ],
        intro_enabled=True,
        intro_text="",
        outro_enabled=True,
        outro_text="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_intro_picks_series_outline_when_no_explicit_text():
    assert (
        _resolve_intro_text(_params())
        == "El desafío técnico imposible de GTA V en PS3 y Xbox 360"
    )


def test_intro_prefers_user_text_when_provided():
    assert (
        _resolve_intro_text(_params(intro_text="Mi intro personalizado"))
        == "Mi intro personalizado"
    )


def test_intro_falls_back_to_subject_when_no_outline_no_user_text():
    assert (
        _resolve_intro_text(_params(series_outline=[], series_enabled=False))
        == "how GTA V was developed"
    )


def test_intro_uses_default_spanish_when_subject_missing():
    assert (
        _resolve_intro_text(
            _params(series_outline=[], series_enabled=False, video_subject="")
        )
        == "Lo que viene en este video"
    )


def test_outro_multi_part_points_to_next_part():
    text = _resolve_outro_text(_params(series_parts=3))
    assert "Continúa con la siguiente parte" in text
    assert "Gracias por ver" in text


def test_outro_single_video_thanks_the_viewer_without_next_part():
    text = _resolve_outro_text(_params(series_parts=0, series_enabled=False))
    assert "Gracias por ver" in text
    assert "Continúa" not in text
    assert "Síguenos" in text


def test_outro_user_text_wins_over_default():
    text = _resolve_outro_text(
        _params(series_parts=3, outro_text="Nos vemos en la parte dos")
    )
    assert text == "Nos vemos en la parte dos"


def test_outro_disabled_does_not_change_default_resolution_path():
    # When ``outro_enabled`` is False the renderer skips the overlay
    # entirely; the resolver still returns the multi-part text so a focus
    # test on text selection is independent of the toggle.
    text = _resolve_outro_text(_params(outro_enabled=False, series_parts=2))
    assert "Continúa" in text
