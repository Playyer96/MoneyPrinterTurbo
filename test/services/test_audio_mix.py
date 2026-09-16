"""Narration + BGM pre-mix: filter graph shape and the measure/apply contract.

The smallest checks that fail if the audio chain regresses: the ducking
filter only appears when BGM is present, the two-pass loudnorm measurement
JSON is parsed and fed into the apply pass, and a failed measurement pass
is reported as failure rather than silently skipping normalization.
"""

from __future__ import annotations

from unittest.mock import patch

from app.services.render import audio_mix as am


def test_mix_filter_includes_sidechain_duck_only_with_bgm():
    with_bgm = am._mix_filter(has_bgm=True, voice_volume=1.0, bgm_volume=0.2, duration=10.0, final_pass=None)
    without_bgm = am._mix_filter(has_bgm=False, voice_volume=1.0, bgm_volume=0.2, duration=10.0, final_pass=None)
    assert "sidechaincompress" in with_bgm
    assert "sidechaincompress" not in without_bgm
    assert "[aout]" in with_bgm and "[aout]" in without_bgm


def test_mix_filter_final_pass_uses_measured_values():
    measured = {"input_i": "-20.1", "input_tp": "-3.2", "input_lra": "5.0", "input_thresh": "-30.1"}
    graph = am._mix_filter(has_bgm=False, voice_volume=1.0, bgm_volume=0.0, duration=5.0, final_pass=measured)
    assert "measured_I=-20.1" in graph
    assert "measured_TP=-3.2" in graph
    assert "linear=true" in graph


def test_render_mixed_audio_parses_measurement_and_calls_apply_with_it(tmp_path):
    voice = tmp_path / "voice.mp3"
    voice.write_bytes(b"fake")
    output = tmp_path / "mixed.wav"

    measure_stderr = (
        '[Parsed_loudnorm_1 @ 0x0] \n{\n"input_i" : "-19.5",\n"input_tp" : "-2.1",\n'
        '"input_lra" : "6.0",\n"input_thresh" : "-29.6",\n"output_i" : "-14.0",\n'
        '"output_tp" : "-1.0",\n"output_lra" : "7.0",\n"output_thresh" : "-24.0",\n'
        '"normalization_type" : "dynamic",\n"target_offset" : "0.30"\n}\n'
    )

    calls = []

    def fake_run(command, capture_output, text, check, timeout):
        calls.append(command)
        from types import SimpleNamespace

        if "-f" in command and "null" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr=measure_stderr)
        # Apply pass: actually write the output file so the caller sees it exist.
        output.write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with (
        patch.object(am.subprocess, "run", side_effect=fake_run),
        patch.object(am, "probe_duration", return_value=8.5),
    ):
        ok = am.render_mixed_audio(
            voice_path=str(voice), bgm_path=None, voice_volume=1.0, bgm_volume=0.0, output_path=str(output)
        )

    assert ok is True
    assert output.exists()
    assert len(calls) == 2
    apply_filter = calls[1][calls[1].index("-filter_complex") + 1]
    assert "measured_I=-19.5" in apply_filter
    assert "measured_TP=-2.1" in apply_filter


def test_render_mixed_audio_fails_when_measurement_has_no_json(tmp_path):
    voice = tmp_path / "voice.mp3"
    voice.write_bytes(b"fake")

    def fake_run(command, capture_output, text, check, timeout):
        from types import SimpleNamespace

        return SimpleNamespace(returncode=0, stdout="", stderr="no loudnorm output here")

    with (
        patch.object(am.subprocess, "run", side_effect=fake_run),
        patch.object(am, "probe_duration", return_value=5.0),
    ):
        ok = am.render_mixed_audio(
            voice_path=str(voice), bgm_path=None, voice_volume=1.0, bgm_volume=0.0,
            output_path=str(tmp_path / "out.wav"),
        )
    assert ok is False
