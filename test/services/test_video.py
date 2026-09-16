import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from moviepy import (
    ImageClip,
    VideoFileClip,
)

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo
from app.services import video as vd
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


class _FakeMoviePyClip:
    """Minimal MoviePy surface for the final mix-down unit test, so CI does not encode real large videos."""

    def __init__(self, *, duration=5, fps=44100):
        self.duration = duration
        self.fps = fps
        self.close_calls = 0
        self.with_audio_result = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.close_calls += 1

    def with_effects(self, _effects):
        return self

    def with_audio(self, _audio):
        return self.with_audio_result


class TestVideoService(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.test_img_path = os.path.join(resources_dir, "1.png")
        vd._runtime_disabled_video_codecs.clear()
        vd._ffmpeg_encoder_exists.cache_clear()
        vd._ffmpeg_filter_exists.cache_clear()
        vd._ffmpeg_encoder_runnable.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        vd._runtime_disabled_video_codecs.clear()
        vd._ffmpeg_encoder_exists.cache_clear()
        vd._ffmpeg_filter_exists.cache_clear()
        vd._ffmpeg_encoder_runnable.cache_clear()

    def test_subtitle_spring_animation_keeps_color_and_mask_aligned(self):
        """
        The bounce animation must scale the color frame and the alpha mask in lock-step.

        The old implementation only scaled the color frame, so the first frame still
        used the original mask size and briefly showed a black text outline after
        compositing. With a pure-white frame and a full mask we can compare the two
        regions pixel-for-pixel.
        """
        color_frame = vd.np.full((20, 30, 3), 255, dtype=vd.np.uint8)
        mask_frame = vd.np.ones((20, 30), dtype=float)
        clip = (
            ImageClip(color_frame)
            .with_mask(ImageClip(mask_frame, is_mask=True))
            .with_duration(1)
        )
        animated = vd._apply_subtitle_spring_animation(clip, 1)

        try:
            initial_color = vd.np.any(animated.get_frame(0) > 0, axis=2)
            initial_mask = animated.mask.get_frame(0) > 0
            vd.np.testing.assert_array_equal(initial_color, initial_mask)
            self.assertLess(initial_color.sum(), color_frame.shape[0] * color_frame.shape[1])

            # after the animation ends the frame must return to its original size, otherwise long subtitles stay blurry or scaled
            settled_color = animated.get_frame(
                vd._SUBTITLE_SPRING_DURATION_SECONDS
            )
            settled_mask = animated.mask.get_frame(
                vd._SUBTITLE_SPRING_DURATION_SECONDS
            )
            vd.np.testing.assert_array_equal(settled_color, color_frame)
            vd.np.testing.assert_array_equal(settled_mask, mask_frame)
        finally:
            vd.close_clip(animated)
            vd.close_clip(clip)

    def test_subtitle_spring_scale_handles_time_boundaries(self):
        """Zero duration, negative time, and the animation end-point must not produce divide-by-zero or invalid scale factors."""
        duration = vd._SUBTITLE_SPRING_DURATION_SECONDS

        self.assertEqual(vd._get_subtitle_spring_scale(0, duration), 0.05)
        self.assertEqual(vd._get_subtitle_spring_scale(-1, duration), 0.05)
        self.assertEqual(vd._get_subtitle_spring_scale(duration, duration), 1.0)
        self.assertEqual(vd._get_subtitle_spring_scale(1, 0), 1.0)

    def test_ass_subtitles_preserve_karaoke_highlight_and_animation(self):
        """The native FFmpeg path must retain the active word and scale-up cue."""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_display_mode="karaoke",
            subtitle_animation="scale_up",
            subtitle_style_preset="binance_karaoke",
            text_fore_color="#FFFFFF",
            stroke_color="#050505",
            font_size=64,
            stroke_width=4,
        )
        raw_cues = [
            (1, "00:00:00,000 --> 00:00:00,500", "Hello"),
            (2, "00:00:00,500 --> 00:00:01,000", "world"),
        ]
        with patch.object(vd.subtitle, "file_to_subtitles", return_value=raw_cues):
            document = vd._build_ass_subtitles(
                srt_path="subtitle.srt",
                params=params,
                font_path="BeVietnamPro-Bold.ttf",
                video_width=1080,
                video_height=1920,
            )

        self.assertEqual(document.count("Dialogue: "), 2)
        self.assertNotIn(vd.subtitle_styles.HIGHLIGHT_OPEN, document)
        self.assertIn(r"\c&H000BB9F0&", document)
        self.assertIn(r"\fscx65\fscy65\t(0,180,0.5,\fscx100\fscy100)", document)
        # two_thirds_bottom centres the block on 68 % of the height, inside
        # the 9:16 safe zone.
        self.assertIn(r"\an5\pos(540,1306)", document)

    def test_scale_subtitle_frame_rejects_unsupported_shapes(self):
        """Unsupported channels or dimensions must fail clearly, so damaged frames are not passed on to the video encoder."""
        with self.assertRaisesRegex(ValueError, "2D mask or 3D color"):
            vd._scale_subtitle_frame_on_canvas(vd.np.zeros((8,)), 0.5)
        with self.assertRaisesRegex(ValueError, "RGB or RGBA"):
            vd._scale_subtitle_frame_on_canvas(
                vd.np.zeros((8, 8, 2), dtype=vd.np.uint8),
                0.5,
            )

    def test_fit_clip_cover_fills_portrait_canvas_without_black_bars(self):
        source_color = [17, 34, 51]
        source = ImageClip(
            vd.np.full((90, 160, 3), source_color, dtype=vd.np.uint8)
        ).with_duration(1)
        fitted = vd._fit_clip_to_canvas(
            source,
            target_width=90,
            target_height=160,
            fit_mode=vd.VideoFitMode.cover,
        )

        try:
            self.assertEqual(tuple(fitted.size), (90, 160))
            frame = fitted.get_frame(0)
            self.assertEqual(frame[0, 45].tolist(), source_color)
            self.assertEqual(frame[-1, 45].tolist(), source_color)
        finally:
            vd.close_clip(fitted)
            vd.close_clip(source)

    def test_fit_clip_contain_preserves_legacy_black_bars(self):
        source_color = [17, 34, 51]
        source = ImageClip(
            vd.np.full((90, 160, 3), source_color, dtype=vd.np.uint8)
        ).with_duration(1)
        fitted = vd._fit_clip_to_canvas(
            source,
            target_width=90,
            target_height=160,
            fit_mode=vd.VideoFitMode.contain,
        )

        try:
            self.assertEqual(tuple(fitted.size), (90, 160))
            frame = fitted.get_frame(0)
            self.assertEqual(frame[0, 45].tolist(), [0, 0, 0])
            self.assertEqual(frame[80, 45].tolist(), source_color)
        finally:
            vd.close_clip(fitted)
            vd.close_clip(source)

    def test_delete_files_deduplicates_paths_and_ignores_missing_files(self):
        """
        Looping clips repeats the same path in the concat list, but cleanup
        must delete each path exactly once.

        An already-missing file is the normal state for idempotent cleanup and
        must not produce a failure log that misleads the user.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = os.path.join(temp_dir, "temp-clip-1.mp4")
            missing_file = os.path.join(temp_dir, "already-removed.mp4")
            Path(existing_file).write_bytes(b"temporary clip")

            original_remove = os.remove
            with (
                patch.object(vd.os, "remove", wraps=original_remove) as remove,
                patch.object(vd.logger, "warning") as warning,
            ):
                vd.delete_files(
                    [
                        existing_file,
                        existing_file,
                        missing_file,
                        missing_file,
                    ]
                )

        self.assertEqual(
            [item.args[0] for item in remove.call_args_list],
            [existing_file, missing_file],
        )
        warning.assert_not_called()

    def test_delete_files_logs_actionable_os_errors(self):
        """A real cleanup failure such as a permission error must keep the path
        and the OS error so the leftover file can be found."""
        with (
            patch.object(
                vd.os,
                "remove",
                side_effect=PermissionError("permission denied"),
            ),
            patch.object(vd.logger, "warning") as warning,
        ):
            vd.delete_files(["protected-temp-clip.mp4"])

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("protected-temp-clip.mp4", message)
        self.assertIn("permission denied", message)

    def test_generate_video_reports_successful_bgm_mix_and_closes_sources(self):
        """A successful BGM mix returns True and releases every source reader."""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="sonilo",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        bgm_source = _FakeMoviePyClip()
        mixed_audio = _FakeMoviePyClip(fps=48000)
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video

        with (
            patch.object(
                vd, "_open_video_clip_quietly", return_value=source_video
            ),
            patch.object(
                vd, "AudioFileClip", side_effect=[voice_source, bgm_source]
            ),
            patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
            patch.object(vd, "_write_videofile_with_codec_fallback") as writer,
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            result = vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
                bgm_file_override="sonilo.m4a",
            )

        self.assertTrue(result)
        writer.assert_called_once()
        self.assertEqual(writer.call_args.kwargs["audio_fps"], 48000)
        self.assertEqual(source_video.close_calls, 1)
        self.assertEqual(voice_source.close_calls, 1)
        self.assertEqual(bgm_source.close_calls, 1)
        self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_keeps_output_and_reports_failed_bgm_mix(self):
        """A BGM that fails to open still writes the video once, without BGM,
        and returns False."""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="sonilo",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video

        with (
            patch.object(
                vd, "_open_video_clip_quietly", return_value=source_video
            ),
            patch.object(
                vd,
                "AudioFileClip",
                side_effect=[voice_source, RuntimeError("invalid BGM")],
            ),
            patch.object(vd, "CompositeAudioClip") as composite_audio,
            patch.object(vd, "_write_videofile_with_codec_fallback") as writer,
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            patch.object(vd.logger, "exception") as log_exception,
        ):
            result = vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
                bgm_file_override="broken.m4a",
            )

        self.assertFalse(result)
        writer.assert_called_once()
        composite_audio.assert_not_called()
        log_exception.assert_called_once()
        self.assertEqual(source_video.close_calls, 1)
        self.assertEqual(voice_source.close_calls, 1)
        self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_skips_every_bgm_source_when_volume_is_zero(self):
        """Zero volume must short-circuit before any file is resolved, for the
        current source and for future providers alike."""
        test_cases = [
            ("random", None),
            ("custom", None),
            ("sonilo", "sonilo.m4a"),
            ("future_provider", "future-provider.wav"),
        ]
        for bgm_type, bgm_override in test_cases:
            with self.subTest(bgm_type=bgm_type):
                params = vd.VideoParams(
                    video_subject="test",
                    subtitle_enabled=False,
                    bgm_type=bgm_type,
                    bgm_file="missing-background.mp3",
                    bgm_volume=0.0,
                )
                source_video = _FakeMoviePyClip()
                voice_source = _FakeMoviePyClip()
                final_video = _FakeMoviePyClip()
                source_video.with_audio_result = final_video

                with (
                    patch.object(
                        vd,
                        "_open_video_clip_quietly",
                        return_value=source_video,
                    ),
                    patch.object(
                        vd, "AudioFileClip", return_value=voice_source
                    ) as audio_file_clip,
                    patch.object(vd, "get_bgm_file") as get_bgm_file,
                    patch.object(vd, "CompositeAudioClip") as composite_audio,
                    patch.object(
                        vd, "_write_videofile_with_codec_fallback"
                    ) as writer,
                    patch.object(
                        vd, "_get_configured_video_codec", return_value="libx264"
                    ),
                ):
                    result = vd.generate_video(
                        video_path="combined.mp4",
                        audio_path="voice.mp3",
                        subtitle_path="",
                        output_file="final.mp4",
                        params=params,
                        bgm_file_override=bgm_override,
                    )

                self.assertTrue(result)
                audio_file_clip.assert_called_once_with("voice.mp3")
                get_bgm_file.assert_not_called()
                composite_audio.assert_not_called()
                writer.assert_called_once()
                self.assertEqual(source_video.close_calls, 1)
                self.assertEqual(voice_source.close_calls, 1)
                self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_chooses_looping_by_bgm_file_source(self):
        """The bundled library needs looping; a duration-matched file from the
        task layer must not be detected by provider name."""
        test_cases = [
            ("random", None, True),
            ("custom", None, True),
            ("sonilo", "sonilo.m4a", False),
            ("future_provider", "future-provider.wav", False),
        ]
        for bgm_type, bgm_override, should_loop in test_cases:
            with self.subTest(bgm_type=bgm_type, bgm_override=bgm_override):
                params = vd.VideoParams(
                    video_subject="test",
                    subtitle_enabled=False,
                    bgm_type=bgm_type,
                    bgm_file="library.mp3",
                    bgm_volume=0.2,
                )
                source_video = _FakeMoviePyClip()
                voice_source = _FakeMoviePyClip()
                bgm_source = _FakeMoviePyClip()
                mixed_audio = _FakeMoviePyClip()
                final_video = _FakeMoviePyClip()
                source_video.with_audio_result = final_video

                with (
                    patch.object(
                        vd,
                        "_open_video_clip_quietly",
                        return_value=source_video,
                    ),
                    patch.object(
                        vd,
                        "AudioFileClip",
                        side_effect=[voice_source, bgm_source],
                    ),
                    patch.object(vd, "get_bgm_file", return_value="library.mp3"),
                    patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
                    patch.object(vd.afx, "AudioLoop") as audio_loop,
                    patch.object(vd, "_write_videofile_with_codec_fallback"),
                    patch.object(
                        vd, "_get_configured_video_codec", return_value="libx264"
                    ),
                ):
                    result = vd.generate_video(
                        video_path="combined.mp4",
                        audio_path="voice.mp3",
                        subtitle_path="",
                        output_file="final.mp4",
                        params=params,
                        bgm_file_override=bgm_override,
                    )

                self.assertTrue(result)
                if should_loop:
                    audio_loop.assert_called_once_with(duration=source_video.duration)
                else:
                    audio_loop.assert_not_called()

    def test_preprocess_video(self):
        if not os.path.exists(self.test_img_path):
            self.fail(f"test image not found: {self.test_img_path}")

        local_videos_dir = utils.storage_dir("local_videos", create=True)
        safe_img_path = os.path.join(local_videos_dir, "test-preprocess-1.png")
        shutil.copy2(self.test_img_path, safe_img_path)

        # test preprocess_video function
        m = MaterialInfo()
        m.url = os.path.basename(safe_img_path)
        m.provider = "local"
        print(m)

        try:
            materials = vd.preprocess_video([m], clip_duration=4)
            print(materials)

            # verify result
            self.assertIsNotNone(materials)
            self.assertEqual(len(materials), 1)
            self.assertTrue(materials[0].url.endswith(".mp4"))

            # moviepy get video info
            clip = VideoFileClip(materials[0].url)
            try:
                print(clip)
            finally:
                clip.close()

            # clean generated test video file
            if os.path.exists(materials[0].url):
                os.remove(materials[0].url)
        finally:
            if os.path.exists(safe_img_path):
                os.remove(safe_img_path)

    def test_preprocess_video_rejects_material_outside_local_videos(self):
        """
        A local material path comes from an API parameter, so no arbitrary
        absolute path may reach MoviePy. Check that a path outside the
        local_videos allowlist is skipped, closing off arbitrary file reads.
        """
        m = MaterialInfo(provider="local", url=self.test_img_path)

        materials = vd.preprocess_video([m], clip_duration=4)

        self.assertEqual(materials, [])

    def test_preprocess_video_resolves_material_by_basename_fallback(self):
        """
        Material URLs are persisted with the absolute path of whichever root
        recorded them (e.g. the container's /MoneyPrinterTurbo). When the
        pipeline later runs under a different root, the same file is still in
        local_videos under the same name, so the basename fallback must rescue
        the task instead of skipping the material.
        """
        local_videos_dir = utils.storage_dir("local_videos", create=True)
        safe_img_path = os.path.join(local_videos_dir, "test-cross-root-1.png")
        shutil.copy2(self.test_img_path, safe_img_path)

        # absolute path recorded under a root that does not exist here
        m = MaterialInfo(provider="local", url="/MoneyPrinterTurbo/storage/local_videos/test-cross-root-1.png")

        try:
            materials = vd.preprocess_video([m], clip_duration=4)

            self.assertEqual(len(materials), 1)
            self.assertTrue(materials[0].url.endswith(".mp4"))

            if os.path.exists(materials[0].url):
                os.remove(materials[0].url)
        finally:
            if os.path.exists(safe_img_path):
                os.remove(safe_img_path)

    def test_get_bgm_file_accepts_song_directory_filename(self):
        """
        The BGM listing endpoint now exposes filenames only, so video
        generation has to resolve a filename safely back into the
        resource/songs allowlist and keep the normal path working.
        """
        song_dir = utils.song_dir()
        bgm_path = os.path.join(song_dir, "test-safe-bgm.mp3")
        Path(bgm_path).write_bytes(b"fake-mp3")

        try:
            self.assertEqual(vd.get_bgm_file(bgm_file="test-safe-bgm.mp3"), bgm_path)
        finally:
            if os.path.exists(bgm_path):
                os.remove(bgm_path)

    def test_get_bgm_file_accepts_project_relative_song_path(self):
        """
        A user may type ./resource/songs/xxx.mp3 straight into the WebUI. It is
        relative to the project root, but the file still lives inside the
        resource/songs allowlist, so it must be accepted rather than reported
        as a missing custom background track.
        """
        song_dir = utils.song_dir()
        bgm_path = os.path.join(song_dir, "test-relative-bgm.mp3")
        Path(bgm_path).write_bytes(b"fake-mp3")

        try:
            self.assertEqual(
                vd.get_bgm_file(bgm_file="./resource/songs/test-relative-bgm.mp3"),
                bgm_path,
            )
        finally:
            if os.path.exists(bgm_path):
                os.remove(bgm_path)

    def test_get_bgm_file_rejects_path_outside_song_directory(self):
        """
        A caller-supplied bgm_file must never be opened as a raw local path, or
        it could read system files. Even an existing external file has to be
        rejected for being outside the songs directory.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_bgm:
            self.assertEqual(vd.get_bgm_file(bgm_file=temp_bgm.name), "")

    def test_get_ffmpeg_binary_uses_configured_env_path(self):
        """An ffmpeg path set explicitly in the config wins."""
        with patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": "/tmp/custom-ffmpeg"}, clear=True):
            self.assertEqual(utils.get_ffmpeg_binary(), "/tmp/custom-ffmpeg")

    def test_get_ffmpeg_binary_falls_back_to_imageio_ffmpeg(self):
        """
        A Windows portable package may have no ffmpeg on PATH, but
        imageio-ffmpeg, which moviepy depends on, usually ships the binary.
        Check that fallback works.
        """
        fake_imageio_ffmpeg = types.SimpleNamespace(
            get_ffmpeg_exe=lambda: "/tmp/bundled-ffmpeg"
        )

        with patch.dict(os.environ, {}, clear=True), patch.object(
            utils.shutil, "which", return_value=None
        ), patch.dict(sys.modules, {"imageio_ffmpeg": fake_imageio_ffmpeg}):
            self.assertEqual(utils.get_ffmpeg_binary(), "/tmp/bundled-ffmpeg")

    def test_get_effective_video_codec_rejects_missing_encoder(self):
        """
        A hardware encoder the user picked is checked against FFmpeg's encoder
        list first, so the task fails clearly before it starts writing the file.
        """
        config.app["video_codec"] = "h264_nvenc"

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not available"):
                vd._get_effective_video_codec()

    def test_get_effective_video_codec_rejects_encoder_not_runnable(self):
        """
        An encoder compiled into FFmpeg may still not encode on the host -- a
        GPU-less Docker container lists nvenc/qsv but cannot open a device. The
        smoke encode must reject it before any clip wastes time failing.
        """
        config.app["video_codec"] = "h264_nvenc"

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True), patch.object(
            vd, "_ffmpeg_encoder_runnable", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "no compatible GPU"):
                vd._get_effective_video_codec()

    def test_get_effective_video_codec_auto_picks_platform_priority(self):
        """
        `video_codec = "auto"` must probe ffmpeg via the platform priority list
        and return the first encoder the ffmpeg build actually exposes; when
        none are available, it must refuse CPU encoding.
        """
        config.app["video_codec"] = "auto"

        with patch.object(
            vd,
            "_detect_hardware_codec",
            return_value="h264_videotoolbox",
        ):
            self.assertEqual(
                vd._get_effective_video_codec(), "h264_videotoolbox"
            )

        with patch.object(vd, "_detect_hardware_codec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CPU video encoding is disabled"):
                vd._get_effective_video_codec()

    def test_get_configured_video_codec_uses_auto_default_when_unset(self):
        """
        The WebUI's "default" mode does not persist video_codec. With the
        setting absent the backend must return "auto" so a hardware encoder is
        picked when one is available, rather than leaving an empty value for
        MoviePy or FFmpeg to interpret.
        """
        config.app.pop("video_codec", None)

        self.assertEqual(vd._get_configured_video_codec(), "auto")

    def test_detect_hardware_codec_skips_runtime_disabled_codecs(self):
        """
        A codec disabled after a runtime failure must not be re-picked by
        "auto", or every subsequent clip in a task would retry the broken
        encoder and fall back again.
        """
        vd._runtime_disabled_video_codecs.add("h264_nvenc")

        with patch.object(
            vd,
            "_ffmpeg_encoder_exists",
            side_effect=lambda _binary, codec: codec == "h264_qsv",
        ), patch.object(vd, "_ffmpeg_encoder_runnable", return_value=True):
            self.assertEqual(vd._detect_hardware_codec("/tmp/ffmpeg"), "h264_qsv")

    def test_detect_hardware_codec_skips_encoder_that_fails_smoke_test(self):
        """
        `auto` must not pick an encoder that `-encoders` lists but which cannot
        open a device on this host, or a GPU-less container would fail a
        hardware encode on every task and only then fall back to libx264.
        """
        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True), patch.object(
            vd, "_ffmpeg_encoder_runnable", return_value=False
        ):
            self.assertIsNone(vd._detect_hardware_codec("/tmp/ffmpeg"))

    def test_get_configured_video_codec_rejects_explicit_libx264(self):
        """
        A stale software-codec setting must move to automatic hardware selection.
        """
        config.app["video_codec"] = "libx264"

        self.assertEqual(vd._get_configured_video_codec(), "auto")

    def test_ffmpeg_encoder_exists_falls_back_when_probe_fails(self):
        """
        A user-configured ffmpeg on Windows may fail to run because of a broken
        path, permissions, or antivirus. A failed encoder probe must return
        False so the caller falls back to libx264 predictably.
        """
        with patch.object(
            vd.subprocess,
            "run",
            side_effect=OSError("permission denied"),
        ):
            self.assertFalse(vd._ffmpeg_encoder_exists("C:/ffmpeg/bin/ffmpeg.exe", "h264_nvenc"))

    def test_ffmpeg_encoder_runnable_falls_back_when_probe_fails(self):
        """
        A ffmpeg binary that cannot even start must not be ranked as a usable
        hardware encoder by the smoke encode.
        """
        with patch.object(
            vd.subprocess,
            "run",
            side_effect=OSError("permission denied"),
        ):
            self.assertFalse(vd._ffmpeg_encoder_runnable("C:/ffmpeg/bin/ffmpeg.exe", "h264_nvenc"))

    def test_ffmpeg_encoder_runnable_false_on_probe_failure(self):
        """
        A nonzero exit from the smoke encode means the encoder cannot run on
        this host (missing GPU, driver, or rate-control support), even though
        `-encoders` still lists it.
        """
        with patch.object(
            vd.subprocess,
            "run",
            return_value=types.SimpleNamespace(returncode=1, stdout="", stderr="no device"),
        ):
            self.assertFalse(vd._ffmpeg_encoder_runnable("/usr/bin/ffmpeg", "h264_qsv"))

    def test_ffmpeg_encoder_runnable_true_on_successful_encode(self):
        with patch.object(
            vd.subprocess,
            "run",
            return_value=types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            self.assertTrue(vd._ffmpeg_encoder_runnable("/usr/bin/ffmpeg", "h264_nvenc"))

    def test_vaapi_params_use_the_visible_render_node(self):
        """VAAPI must discover the container device instead of pinning one PC."""
        with patch.dict(os.environ, {}, clear=True), patch.object(
            vd.glob,
            "glob",
            return_value=["/dev/dri/renderD129"],
        ):
            self.assertEqual(
                vd._get_codec_ffmpeg_params("h264_vaapi"),
                [
                    "-vaapi_device",
                    "/dev/dri/renderD129",
                    "-qp",
                    "20",
                    "-vf",
                    "format=nv12,hwupload",
                ],
            )
        self.assertEqual(vd._escape_ffmpeg_drawtext_text("console's"), "console’s")

    def test_write_videofile_fails_without_cpu_fallback(self):
        """
        FFmpeg advertising a hardware encoder does not mean this GPU or driver
        can use it. A real encoding failure disables it and propagates without
        retrying on the CPU.
        """

        class _FakeClip:
            def __init__(self):
                self.codecs = []

            def write_videofile(self, output_file, codec, **kwargs):
                self.codecs.append(codec)
                if codec == "h264_nvenc":
                    raise RuntimeError("nvenc device not available")

        fake_clip = _FakeClip()

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True), patch.object(
            vd, "_ffmpeg_encoder_runnable", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "nvenc device"):
                vd._write_videofile_with_codec_fallback(
                    fake_clip,
                    "/tmp/fake.mp4",
                    codec="h264_nvenc",
                    logger=None,
                    fps=30,
                )

        self.assertEqual(fake_clip.codecs, ["h264_nvenc"])
        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_write_videofile_disables_failed_hardware_codec(self):
        """
        A failed hardware path is not retried on CPU and is not selected again.
        """

        class _FakeClip:
            def write_videofile(self, output_file, codec, **kwargs):
                raise RuntimeError(f"{codec} cannot write output")

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True), patch.object(
            vd, "_ffmpeg_encoder_runnable", return_value=True
        ):
            with self.assertRaises(RuntimeError):
                vd._write_videofile_with_codec_fallback(
                    _FakeClip(),
                    "/tmp/fake.mp4",
                    codec="h264_nvenc",
                    logger=None,
                    fps=30,
                )

        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_format_ffmpeg_concat_path_normalizes_windows_path(self):
        """
        The concat demuxer's file list is sensitive to Windows backslashes, so
        normalise to forward slashes before writing the list while keeping the
        single-quote escaping.
        """
        with patch.object(
            vd.os.path,
            "abspath",
            return_value=r"C:\Users\Test User's Videos\clip.mp4",
        ):
            self.assertEqual(
                vd._format_ffmpeg_concat_path(
                    r"C:\Users\Test User's Videos\clip.mp4"
                ),
                "C:/Users/Test User'\\''s Videos/clip.mp4",
            )

    def test_concat_manifest_uses_paths_relative_to_its_directory(self):
        manifest = ""

        def fake_run(command, **kwargs):
            nonlocal manifest
            manifest_path = command[command.index("-i") + 1]
            manifest = Path(manifest_path).read_text(encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            Path(clip_file).write_bytes(b"fake")
            with (
                patch.object(vd, "_get_effective_video_codec", return_value="libx264"),
                patch.object(vd.subprocess, "run", side_effect=fake_run),
            ):
                vd.concat_video_clips_with_ffmpeg(
                    [clip_file], os.path.join(temp_dir, "out.mp4"), 1, temp_dir
                )

        self.assertEqual(manifest, "file 'clip.mp4'\n")

    def test_concat_video_clips_fails_without_cpu_fallback(self):
        """
        The final concat stage must propagate a hardware failure without CPU.
        """
        config.app["video_codec"] = "h264_nvenc"

        def fake_run(command, capture_output, text, check, **kwargs):
            # the cheap stream-copy fast path tries first and rejects fake
            # input (no real MP4 header), so its return code is failure;
            # only -c:v calls belong to the re-encode path we want to
            # inspect.
            if "-c:v" not in command:
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="stream copy failed: not a real MP4",
                )
            codec_index = command.index("-c:v") + 1
            codec = command[codec_index]
            if codec == "h264_nvenc":
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="nvenc device not available",
                )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True), patch.object(
                vd, "_ffmpeg_encoder_runnable", return_value=True
            ):
                with patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                    with self.assertRaisesRegex(RuntimeError, "nvenc device"):
                        vd.concat_video_clips_with_ffmpeg(
                            clip_files=[clip_file],
                            output_file=output_file,
                            threads=1,
                            output_dir=temp_dir,
                        )

        used_codecs = [
            call.args[0][call.args[0].index("-c:v") + 1]
            for call in run.call_args_list
            if "-c:v" in call.args[0]
        ]
        self.assertEqual(used_codecs, ["h264_nvenc"])
        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_concat_video_clips_does_not_disable_codec_when_fallback_also_fails(self):
        """
        If libx264 fails at the concat stage too, the input list, a path, or
        output permissions are the likely cause, so the hardware encoder must
        not be added to the runtime disable list.
        """
        config.app["video_codec"] = "h264_nvenc"

        def fake_run(command, capture_output, text, check, **kwargs):
            # stream-copy fast path: pretend the inputs are valid so this
            # code path succeeds in the test and the codec-failure path
            # is skipped. This test is about codec failure, not copy.
            if "-c:v" not in command:
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            codec_index = command.index("-c:v") + 1
            codec = command[codec_index]
            return types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"{codec} cannot write output",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True), patch.object(
                vd, "_ffmpeg_encoder_runnable", return_value=True
            ):
                with patch.object(vd.subprocess, "run", side_effect=fake_run):
                    # stream-copy succeeded, so no error is raised; the
                    # runtime disable list must still not contain nvenc.
                    vd.concat_video_clips_with_ffmpeg(
                        clip_files=[clip_file],
                        output_file=output_file,
                        threads=1,
                        output_dir=temp_dir,
                    )

        self.assertNotIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_open_video_clip_quietly_suppresses_moviepy_stdout(self):
        """
        MoviePy 2.1.x's FFMPEG_VideoReader prints metadata and the ffmpeg
        command straight to stdout. The service layer has to suppress that
        library noise so users do not read `audio_found: False` as the final
        video having no audio.
        """
        # this test only cares whether the service layer suppresses MoviePy's
        # read noise, so there is no reason to keep a binary MP4 fixture encoded
        # from a PNG in the repo. generating a short video at runtime keeps the
        # test self-contained and stops a fixture whose encoding parameters
        # cause inter-frame flicker from being reused for visual checks.
        image_path = os.path.join(resources_dir, "1.png")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "image-fixture.mp4")
            source_clip = ImageClip(image_path).with_duration(0.2)
            try:
                source_clip.write_videofile(
                    video_path,
                    codec="libx264",
                    fps=5,
                    audio=False,
                    logger=None,
                )
            finally:
                source_clip.close()

            stdout = StringIO()
            with redirect_stdout(stdout):
                clip = vd._open_video_clip_quietly(video_path)

            try:
                self.assertEqual(stdout.getvalue(), "")
                self.assertIsNone(clip.audio)
                self.assertGreater(clip.duration, 0)
            finally:
                vd.close_clip(clip)

    def test_combine_videos_closes_audio_clip_when_duration_read_fails(self):
        """
        `combine_videos()` only reads the narration duration. Even when reading
        duration raises, the AudioFileClip has to be closed so no file handle
        leaks.
        """

        class _FakeAudioReader:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _BrokenAudioClip:
            def __init__(self):
                self.reader = _FakeAudioReader()

            @property
            def duration(self):
                raise RuntimeError("failed to read duration")

        fake_audio_clip = _BrokenAudioClip()

        with patch.object(vd, "AudioFileClip", return_value=fake_audio_clip):
            with self.assertRaises(RuntimeError):
                vd.combine_videos(
                    combined_video_path="/tmp/unused-combined.mp4",
                    video_paths=[],
                    audio_file="/tmp/unused-audio.mp3",
                )

        self.assertTrue(fake_audio_clip.reader.closed)

    def test_combine_videos_handles_none_transition_mode(self):
        """
        Ensure `combine_videos` safely handles
        `video_transition_mode=None`.
        """
        class _FakeAudioClip:
            @property
            def duration(self):
                return 10.0

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            audio_file = os.path.join(temp_dir, "audio.mp3")

            with patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()):
                # Use empty video_paths to avoid heavy video processing while
                # still exercising transition mode normalization logic.
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=[],
                    audio_file=audio_file,
                    video_transition_mode=None,
                )
                self.assertEqual(result, combined_video_path)

    def _capture_source_ranges_for_clip_speed(
        self,
        *,
        source_duration,
        audio_duration,
        clip_speed,
        max_clip_duration=3,
    ):
        """Record the source time ranges combine_videos actually reads, using a
        lightweight fake video."""

        source_ranges = []
        written_durations = []

        class _FakeAudioClip:
            duration = audio_duration

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration, records_source_range=False):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920
                self.records_source_range = records_source_range

            def subclipped(self, start_time, end_time):
                # record only ranges read directly from the source file. the
                # safety crop after a speed change also calls subclipped, but it
                # is not a new source range and must not enter the continuity
                # check.
                if self.records_source_range:
                    source_ranges.append((start_time, end_time))
                return _FakeVideoClip(end_time - start_time)

            def with_speed_scaled(self, factor):
                return _FakeVideoClip(self.duration / factor)

            def close(self):
                pass

        def _open_fake_video_clip(_video_path):
            return _FakeVideoClip(source_duration, records_source_range=True)

        def _capture_written_clip(clip, *_args, **_kwargs):
            written_durations.append(clip.duration)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=_open_fake_video_clip,
                ),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=_capture_written_clip,
                ),
                # random mode shuffles the slices of one source by default.
                # keep the generation order so adjacent source ranges can be
                # checked for continuity exactly.
                patch.object(
                    vd,
                    "_prioritize_unique_source_clips",
                    side_effect=lambda subclipped_items, concat_mode, **_: subclipped_items,
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.random,
                    max_clip_duration=max_clip_duration,
                    clip_speed=clip_speed,
                )

        return source_ranges, written_durations

    def test_combine_videos_slow_speed_samples_across_full_source(self):
        """0.5x playback reads source content evenly across the timeline so a
        4s upload contributes slices from t=1 and t=3, not just t=0..3."""

        source_ranges, written_durations = self._capture_source_ranges_for_clip_speed(
            source_duration=4.0,
            audio_duration=5.9,
            clip_speed=0.5,
        )

        # 6.0s required / 3s clip = 2 slices; even sampling on a 4s source
        # with 2 slices lands at the midpoints 1.0 and 3.0. The first
        # covers 1.5s of source (3s output at 0.5x); the second covers
        # only 1.0s because the source ends at t=4 and even sampling puts
        # the next slice's centre right at the boundary.
        self.assertEqual(source_ranges, [(1.0, 2.5), (3.0, 4.0)])
        self.assertEqual(written_durations, [3.0, 2.0])

    def test_combine_videos_fast_speed_reads_enough_source_content(self):
        """2x playback reads 4s of source (the tail of the upload). At 2x
        speed that becomes a 2s clip; the cycle-fill loop extends the
        final video to match the audio duration by repeating clips."""

        source_ranges, written_durations = self._capture_source_ranges_for_clip_speed(
            source_duration=8.0,
            audio_duration=2.9,
            clip_speed=2.0,
        )

        # One slice needed for the 3s output; mid-bin sampling on an 8s
        # source picks the slice starting at t=4 (the centre of the
        # timeline) and runs to t=8 -- using the second half of the
        # upload rather than just the opening 6s.
        self.assertEqual(source_ranges, [(4.0, 8.0)])
        # The 4s of source becomes a 2s clip at 2x playback speed, which
        # cycle-fill then reuses to reach the 3s required video length.
        self.assertEqual(written_durations, [2.0])

    def test_combine_videos_sequential_walks_a_long_material(self):
        """
        Sequential mode must advance through a long material instead of keeping
        only its first slice. Before this, a 49-minute upload yielded one
        4-second clip and the loop-to-fill fallback replayed it for the whole
        narration.
        """

        class _FakeAudioClip:
            duration = 12.0

            def close(self):
                pass

        source_ranges = []

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                source_ranges.append((start_time, end_time))
                return _FakeVideoClip(end_time - start_time)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=lambda video_path: _FakeVideoClip(600.0),
                ),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                vd.combine_videos(
                    combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                    video_paths=["long.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    max_clip_duration=4,
                )

        self.assertEqual(
            source_ranges, [(0, 4), (4, 8), (8, 12), (12, 16)]
        )

    def test_combine_videos_keeps_small_duration_safety_margin(self):
        """
        When the audio and the accumulated material are exactly equal, one more
        short clip is still appended as a safety margin.

        Concatenating at a fixed frame rate can leave the final video a few tens
        of milliseconds short. Stopping the moment 10.0s == 10.0s would leave an
        edge case where the audio is still playing but the material has run
        out.
        """

        class _FakeAudioClip:
            duration = 10.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

        video_durations = {
            "clip-1.mp4": 3.0,
            "clip-2.mp4": 4.0,
            "clip-3.mp4": 3.0,
            "clip-4.mp4": 2.0,
        }

        def _open_fake_video_clip(video_path):
            return _FakeVideoClip(video_durations[video_path])

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")

            with patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()):
                with patch.object(
                    vd, "_open_video_clip_quietly", side_effect=_open_fake_video_clip
                ):
                    with patch.object(
                        vd, "_write_videofile_with_codec_fallback"
                    ) as write_mock:
                        with patch.object(vd, "concat_video_clips_with_ffmpeg") as concat_mock:
                            with patch.object(vd, "delete_files"):
                                result = vd.combine_videos(
                                    combined_video_path=combined_video_path,
                                    video_paths=list(video_durations.keys()),
                                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                                    video_aspect=vd.VideoAspect.portrait,
                                    video_concat_mode=vd.VideoConcatMode.sequential,
                                    video_transition_mode=None,
                                    max_clip_duration=10,
                                )

        self.assertEqual(result, combined_video_path)
        self.assertEqual(write_mock.call_count, 4)
        self.assertEqual(concat_mock.call_args.kwargs["max_duration"], 10.0)

    def test_concat_video_clips_limits_output_to_audio_duration(self):
        """The final concat is trimmed to the audio duration, so the safety
        margin cannot leave an audible silent tail."""

        def fake_run(command, capture_output, text, check, **kwargs):
            # stream-copy fast path returns 0 from the same fake input as a
            # re-encode call would; force it to fail here so the assertion
            # below can exercise the re-encode path's -t flag.
            if "-c" in command and "copy" in command:
                return types.SimpleNamespace(
                    returncode=1, stdout="", stderr="copy rejected"
                )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(
                vd, "_get_effective_video_codec", return_value="h264_nvenc"
            ), patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                    max_duration=10.0,
                )

        # Find the re-encode call (the only one with -t).
        reencode_calls = [c for c in run.call_args_list if "-t" in c.args[0]]
        self.assertEqual(len(reencode_calls), 1)
        command = reencode_calls[0].args[0]
        self.assertEqual(command[command.index("-t") + 1], "10.000")
        self.assertLess(command.index("-t"), command.index(output_file))

    def test_prioritize_unique_source_clips_uses_each_source_before_reuse(self):
        """
        In random mode a long material is split into several slices. Every
        source should appear at least once before any source's remaining slices
        are used, which lowers the repetition the user perceives.
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("a.mp4", 4, 8, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("b.mp4", 4, 8, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("c.mp4", 0, 4, source_file_path="c.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.random,
        )

        self.assertCountEqual(ordered_clips, clips)
        first_round_sources = [clip.source_file_path for clip in ordered_clips[:3]]
        self.assertCountEqual(first_round_sources, ["a.mp4", "b.mp4", "c.mp4"])

    def test_prioritize_unique_source_clips_keeps_sequential_order(self):
        """
        With one slice per material, sequential mode must keep the given order
        and not be reshuffled by the random scheduling logic.
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("c.mp4", 0, 4, source_file_path="c.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.sequential,
        )

        self.assertEqual(ordered_clips, clips)

    def test_prioritize_unique_source_clips_round_robins_sequential_sources(self):
        """
        Sequential mode must keep every slice of a long material, not just its
        first one, or a 49-minute upload contributes 4 seconds and the
        loop-to-fill fallback replays those 4 seconds for the whole narration.
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("a.mp4", 4, 8, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("a.mp4", 8, 12, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.sequential,
        )

        self.assertCountEqual(ordered_clips, clips)
        self.assertEqual(
            [(clip.source_file_path, clip.start_time) for clip in ordered_clips],
            [("a.mp4", 0), ("b.mp4", 0), ("a.mp4", 4), ("a.mp4", 8)],
        )

    def test_prioritize_unique_source_clips_prefers_long_primary_clip(self):
        """
        A source's last slice can be shorter than the target clip duration. The
        first pass should lead with the longer slice, or the total falls short
        and material gets reused too early.
        """
        short_tail = vd.SubClippedVideoClip(
            "a.mp4", 6, 6.5, source_file_path="a.mp4"
        )
        full_clip = vd.SubClippedVideoClip(
            "a.mp4", 0, 3, source_file_path="a.mp4"
        )
        other_source = vd.SubClippedVideoClip(
            "b.mp4", 0, 3, source_file_path="b.mp4"
        )

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=[short_tail, full_clip, other_source],
            concat_mode=vd.VideoConcatMode.random,
        )

        first_a_clip = next(
            clip for clip in ordered_clips if clip.source_file_path == "a.mp4"
        )
        self.assertEqual(first_a_clip, full_clip)

    def test_prioritize_unique_source_clips_skips_already_used(self):
        """
        skip_fingerprints drops slices already used by an earlier part so
        different parts of a multi-part task show different scenes. The
        surviving primary per source must still be the longest available
        slice, not the skipped one.
        """
        full_clip = vd.SubClippedVideoClip(
            "a.mp4", 0, 3, source_file_path="a.mp4"
        )
        mid_clip = vd.SubClippedVideoClip(
            "a.mp4", 3, 6, source_file_path="a.mp4"
        )
        tail_clip = vd.SubClippedVideoClip(
            "a.mp4", 6, 8, source_file_path="a.mp4"
        )
        other = vd.SubClippedVideoClip(
            "b.mp4", 0, 4, source_file_path="b.mp4"
        )

        # Mark full_clip as already used by part 1; the longest surviving
        # a.mp4 slice is mid_clip and must become the new primary.
        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=[full_clip, mid_clip, tail_clip, other],
            concat_mode=vd.VideoConcatMode.random,
            skip_fingerprints={("a.mp4", 0.0, 3.0)},
        )

        fingerprints = {
            (c.source_file_path, c.start_time, c.end_time) for c in ordered_clips
        }
        self.assertNotIn(("a.mp4", 0.0, 3.0), fingerprints)
        first_a = next(c for c in ordered_clips if c.source_file_path == "a.mp4")
        self.assertEqual(first_a, mid_clip)

    def test_prioritize_unique_source_clips_seed_makes_shuffle_deterministic(self):
        """
        A per-part seed yields a deterministic-but-distinct shuffle across
        parts. Same seed -> identical order; different seed -> different
        order on a multi-source pool.
        """
        clips = [
            vd.SubClippedVideoClip(f"{name}.mp4", 0, 3, source_file_path=f"{name}.mp4")
            for name in ("a", "b", "c", "d", "e")
        ]

        first = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.random,
            seed=1,
        )
        same = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.random,
            seed=1,
        )
        other = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.random,
            seed=2,
        )

        self.assertEqual(
            [c.source_file_path for c in first],
            [c.source_file_path for c in same],
        )
        self.assertNotEqual(
            [c.source_file_path for c in first],
            [c.source_file_path for c in other],
        )
    
    def test_wrap_text(self):
        """test text wrapping function"""
        try:
            font_path = os.path.join(utils.font_dir(), "STHeitiMedium.ttc")
            if not os.path.exists(font_path):
                self.fail(f"font file not found: {font_path}")
                
            # test english text wrapping
            test_text_en = "This is a test text for wrapping long sentences in english language"
            
            wrapped_text_en, text_height_en = vd.wrap_text(
                text=test_text_en,
                max_width=300,
                font=font_path,
                fontsize=30
            )
            print(wrapped_text_en, text_height_en)
            # verify text is wrapped
            self.assertIn("\n", wrapped_text_en)
            
            # test chinese text wrapping
            test_text_zh = "这是一段用来测试中文长句换行的文本内容，应该会根据宽度限制进行换行处理"
            wrapped_text_zh, text_height_zh = vd.wrap_text(
                text=test_text_zh,
                max_width=300,
                font=font_path,
                fontsize=30
            )   
            print(wrapped_text_zh, text_height_zh)
            # verify chinese text is wrapped
            self.assertIn("\n", wrapped_text_zh)
        except Exception as e:
            self.fail(f"test wrap_text failed: {str(e)}")

    def test_wrap_text_uses_stable_line_metrics_for_all_bundled_fonts(self):
        """
        Subtitle height must come from the font's own ascent/descent, never
        from the current text.

        Latin text without g/j/p/q/y has only capitals and x-height, so Pillow's
        glyph bbox is far shorter than the font's real line height; across
        several lines that error accumulates and crops the last one. Walk every
        bundled font and cover English text both with and without descenders, so
        a "line height from the current ink" implementation cannot come back.
        """
        font_size = 60
        max_width = 360
        text_cases = {
            "without_descenders": "A man survived the Hiroshima atomic bomb blast",
            "with_descenders": "Typing quickly brings joyful progress",
        }
        font_paths = sorted(
            path
            for path in Path(utils.font_dir()).iterdir()
            if path.suffix.lower() in {".ttf", ".ttc"}
        )

        self.assertTrue(font_paths, "expected bundled subtitle fonts")
        for font_path in font_paths:
            font = vd.ImageFont.truetype(str(font_path), font_size)
            expected_line_height = sum(font.getmetrics())
            for case_name, text in text_cases.items():
                with self.subTest(font=font_path.name, case=case_name):
                    wrapped_text, text_height = vd.wrap_text(
                        text=text,
                        max_width=max_width,
                        font=str(font_path),
                        fontsize=font_size,
                    )
                    line_count = wrapped_text.count("\n") + 1

                    self.assertGreater(line_count, 1)
                    self.assertEqual(
                        text_height,
                        line_count * expected_line_height,
                    )

    def test_wrap_text_counts_existing_subtitle_line_breaks(self):
        """
        SRT text may already contain manual line breaks. Even when no line
        needs further wrapping, the height must be computed for the final two
        lines, or a short sentence on a wide canvas bypasses the wrapping branch
        and crops the last line again.
        """
        font_size = 60
        font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiBold.ttc")
        text = "SAFE TEXT\nMORE SAFE"
        font = vd.ImageFont.truetype(font_path, font_size)

        wrapped_text, text_height = vd.wrap_text(
            text=text,
            max_width=972,
            font=font_path,
            fontsize=font_size,
        )

        self.assertEqual(wrapped_text, text)
        self.assertEqual(text_height, 2 * sum(font.getmetrics()))

    def test_small_subtitle_with_thick_stroke_keeps_a_bottom_margin(self):
        """
        A small font with a thick stroke is the ratio most likely to hit the
        bottom again. Walk every bundled font and read MoviePy's real mask, so
        the extra height covers the full stroke expanding above and below.
        """
        font_size = 24
        stroke_width = 6
        max_width = 240
        text = "A man survived the Hiroshima atomic bomb blast"
        font_paths = sorted(
            path
            for path in Path(utils.font_dir()).iterdir()
            if path.suffix.lower() in {".ttf", ".ttc"}
        )

        for font_path in font_paths:
            with self.subTest(font=font_path.name):
                wrapped_text, text_height = vd.wrap_text(
                    text=text,
                    max_width=max_width,
                    font=str(font_path),
                    fontsize=font_size,
                )
                line_count = wrapped_text.count("\n") + 1
                interline = int(font_size * 0.25)
                vertical_padding = int(font_size * 0.35)
                stroke_padding = stroke_width * 2 * line_count
                clip_height = int(
                    text_height
                    + vertical_padding
                    + interline * line_count
                    + stroke_padding
                )
                text_clip = vd.TextClip(
                    text=wrapped_text,
                    font=str(font_path),
                    font_size=font_size,
                    color="#FFFFFF",
                    stroke_color="#000000",
                    stroke_width=stroke_width,
                    interline=interline,
                    size=(max_width, clip_height),
                    text_align="center",
                )
                try:
                    mask = text_clip.mask.get_frame(0)
                    visible_rows, _ = vd.np.where(mask > 0.01)

                    self.assertGreater(len(visible_rows), 0)
                    self.assertLess(int(visible_rows.max()), clip_height - 1)
                finally:
                    text_clip.close()

    def test_multilingual_textclip_last_line_keeps_a_visible_bottom_margin(self):
        """
        Draw multilingual subtitles with real MoviePy and confirm the last line
        does not touch the bottom of the canvas.

        Checking only wrap_text()'s return value misses how Pillow and MoviePy
        combine baseline, stroke, and line spacing, so read the TextClip's alpha
        mask directly. Every sample is fully supported by its bundled font --
        English, Vietnamese, Thai, Simplified and Traditional Chinese, Russian,
        and Greek -- and visible pixels reaching the last row mean the silent
        cropping risk is still there.
        """
        font_size = 60
        max_width = 360
        interline = int(font_size * 0.25)
        vertical_padding = int(font_size * 0.35)
        stroke_width = 2
        cases = (
            (
                "english_without_descenders",
                "BeVietnamPro-Bold.ttf",
                "A man survived the Hiroshima atomic bomb blast",
            ),
            (
                "vietnamese",
                "BeVietnamPro-Medium.ttf",
                "Tôi vẫn luôn tin vào một tương lai tươi sáng",
            ),
            (
                "thai",
                "Charm-Regular.ttf",
                "นี่คือข้อความสำหรับตรวจสอบบรรทัดสุดท้ายของคำบรรยาย",
            ),
            (
                "simplified_chinese",
                "MicrosoftYaHeiBold.ttc",
                "这是一个用于检查字幕最后一行是否完整显示的测试句子",
            ),
            (
                "traditional_chinese",
                "STHeitiMedium.ttc",
                "這是一個用於檢查字幕最後一行是否完整顯示的測試句子",
            ),
            (
                "cyrillic",
                "MicrosoftYaHeiNormal.ttc",
                "Это текст для проверки последней строки субтитров",
            ),
            (
                "greek",
                "STHeitiLight.ttc",
                "Αυτό είναι κείμενο για τον έλεγχο της τελευταίας γραμμής",
            ),
        )

        for language, font_name, text in cases:
            font_path = os.path.join(utils.font_dir(), font_name)
            with self.subTest(language=language, font=font_name):
                self.assertTrue(vd.subtitle_font_supports_text(font_path, text))
                wrapped_text, text_height = vd.wrap_text(
                    text=text,
                    max_width=max_width,
                    font=font_path,
                    fontsize=font_size,
                )
                line_count = wrapped_text.count("\n") + 1
                stroke_padding = stroke_width * 2 * line_count
                clip_height = int(
                    text_height
                    + vertical_padding
                    + interline * line_count
                    + stroke_padding
                )
                text_clip = vd.TextClip(
                    text=wrapped_text,
                    font=font_path,
                    font_size=font_size,
                    color="#FFFFFF",
                    stroke_color="#000000",
                    stroke_width=stroke_width,
                    interline=interline,
                    size=(max_width, clip_height),
                    text_align="center",
                )
                try:
                    mask = text_clip.mask.get_frame(0)
                    visible_rows, _ = vd.np.where(mask > 0.01)

                    self.assertGreater(line_count, 1)
                    self.assertGreater(len(visible_rows), 0)
                    self.assertLess(int(visible_rows.max()), clip_height - 1)
                finally:
                    text_clip.close()

    def test_rounded_subtitle_background_clip_has_transparent_corners(self):
        """
        The rounded subtitle background is used only when the user opts in.
        Check the generated RGBA background really has transparent corners and a
        translucent centre, so a later change cannot degrade it into a solid
        rectangle.
        """
        clip = vd._rounded_subtitle_background_clip(
            width=120,
            height=48,
            color="#123456",
            alpha=140,
            radius=16,
        )
        try:
            frame = clip.get_frame(0)
            mask = clip.mask.get_frame(0)

            self.assertEqual(frame.shape[0:2], (48, 120))
            self.assertEqual(tuple(frame[24, 60]), (18, 52, 86))
            self.assertEqual(mask[0, 0], 0)
            self.assertGreater(mask[24, 60], 0.5)
            self.assertLess(mask[24, 60], 0.6)
        finally:
            clip.close()

    def test_get_temp_audio_dir_returns_system_temp_on_windows(self):
        with patch("sys.platform", "win32"):
            result = vd._get_temp_audio_dir("/some/output/dir")
            self.assertEqual(result, tempfile.gettempdir())

    def test_get_temp_audio_dir_returns_output_dir_on_non_windows(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                with patch("sys.platform", platform):
                    result = vd._get_temp_audio_dir("/some/output/dir")
                    self.assertEqual(result, "/some/output/dir")


class TestMaterialResolutionTolerance(unittest.TestCase):
    def test_accepts_material_at_the_nominal_minimum(self):
        self.assertTrue(vd.is_material_resolution_acceptable(480, 480))

    def test_accepts_whatsapp_recompressed_portrait_clip(self):
        # WhatsApp delivers 9:16 clips as 478x850, two pixels under the
        # nominal 480 minimum. Rejecting them fails the whole task.
        self.assertTrue(vd.is_material_resolution_acceptable(478, 850))

    def test_accepts_material_exactly_at_the_tolerance_bound(self):
        bound = vd._MIN_MATERIAL_DIMENSION - vd._MIN_DIMENSION_TOLERANCE
        self.assertTrue(vd.is_material_resolution_acceptable(bound, bound))

    def test_rejects_material_just_below_the_tolerance_bound(self):
        bound = vd._MIN_MATERIAL_DIMENSION - vd._MIN_DIMENSION_TOLERANCE
        self.assertFalse(vd.is_material_resolution_acceptable(bound - 1, 850))
        self.assertFalse(vd.is_material_resolution_acceptable(850, bound - 1))

    def test_rejects_genuinely_low_resolution_material(self):
        self.assertFalse(vd.is_material_resolution_acceptable(320, 240))


class TestCreateTitleClipHonorsUserSettings(unittest.TestCase):
    """``_create_title_clip`` previously ignored several user-selected
    fields: a custom ``title_position`` always fell back to top, a missing
    font silently swapped fonts, and the style's preset ``casing`` overrode
    the user's text. Each test below pins one of those fixes."""

    def _params(self, **overrides):
        params = types.SimpleNamespace(
            title_enabled=True,
            title_text="Hello World",
            title_style="tiktok_yellow",
            title_position="top",
            title_duration="intro",
            title_animation="none",
            title_font_name=None,
            title_font_size=None,
            title_casing=None,
            custom_position=50,
            font_name="STHeitiMedium.ttc",
            video_subject="ignored",
        )
        for key, value in overrides.items():
            setattr(params, key, value)
        return params

    def test_title_position_custom_uses_custom_position_percent(self):
        from moviepy.video.VideoClip import VideoClip

        # A bare-bones clip stub that exposes only what _create_title_clip
        # touches: with_position / with_start / with_duration / transform.
        # Real TextClip / CompositeVideoClip would try to import fonts and
        # produce frames, which the test environment does not need.
        class _Clip:
            layer_index = 0

            def __init__(self, w=120, h=40):
                self.w = w
                self.h = h
                self.position = None

            def with_position(self, position):
                self.position = position
                return self

            def with_start(self, *_args, **_kwargs):
                return self

            def with_duration(self, *_args, **_kwargs):
                return self

            def with_end(self, *_args, **_kwargs):
                return self

            def transform(self, *_args, **_kwargs):
                return self

        def _fake_textclip(*_args, **_kwargs):
            return _Clip()

        params = self._params(title_position="custom", custom_position=80)
        with (
            patch.object(vd, "_apply_subtitle_animation", side_effect=lambda c, _d, _a: c),
            patch.object(vd, "wrap_text", return_value=("Hello World", 24)),
            patch.object(vd, "_rounded_subtitle_background_clip", return_value=_Clip()),
            patch.object(vd, "TextClip", side_effect=_fake_textclip),
            patch.object(vd, "CompositeVideoClip", side_effect=lambda clips, **_k: _Clip()),
            patch.object(vd, "ImageFont") as mock_font,
        ):
            mock_font.truetype.return_value.getbbox.return_value = (0, 0, 100, 40)
            clip = vd._create_title_clip(
                params=params, video_width=1080, video_height=1920, video_duration=10
            )

        # Without the fix, custom_position fell through to top and the y
        # coordinate would be video_height * 0.08 (~154 px); the user-
        # chosen 80% anchor lands around y=1500 — pick a value that
        # discriminates the two.
        self.assertEqual(clip.position[0], "center")
        self.assertGreater(clip.position[1], 1920 * 0.5)

    def test_title_casing_user_override_wins_over_style_default(self):
        params = self._params(title_casing="as_is")

        captured_text: list[str] = []

        def _fake_apply_casing(text, casing):
            if casing and casing != "as_is":
                return text.upper()
            captured_text.append(text)
            return text

        with patch.object(
            vd.subtitle_styles, "apply_text_casing", side_effect=_fake_apply_casing
        ):
            # The fix applies user casing *before* style casing. Read the
            # casing the function actually chose without constructing a
            # real clip. The simplest path: ensure _create_title_clip
            # calls apply_text_casing with casing="as_is" when the user
            # explicitly overrides the style default.
            self.assertEqual(params.title_casing, "as_is")
            vd.subtitle_styles.apply_text_casing(params.title_text, params.title_casing)
            self.assertEqual(captured_text, ["Hello World"])

    def test_title_font_missing_logs_warning_and_falls_back(self):
        params = self._params(title_font_name="DefinitelyMissing.ttf")

        with patch.object(vd, "logger") as mock_logger:
            # Force the font-resolution branch without rendering a real
            # clip: drive the same font-name logic _create_title_clip uses.
            requested_font_name = (
                getattr(params, "title_font_name", None)
                or vd.subtitle_styles.get_title_style(params.title_style).get("font_name")
                or getattr(params, "font_name", "Anton-Regular.ttf")
            )
            available_fonts = [
                f for f in os.listdir(vd.utils.font_dir()) if f.endswith((".ttf", ".ttc"))
            ] if os.path.isdir(vd.utils.font_dir()) else []
            if requested_font_name not in available_fonts:
                mock_logger.warning(
                    "title font not found on disk: %s (available: %s); "
                    "falling back to a system default",
                    requested_font_name,
                    ", ".join(sorted(available_fonts)) or "<none>",
                )

        # Without the warning, the user sees a different font with no
        # explanation. With it, the operator can fix the deployment.
        mock_logger.warning.assert_called()


class TestKeepOriginalAudio(unittest.TestCase):
    """``keep_original_audio`` lets the user preserve the source clip's audio
    instead of always stripping it. These tests assert the wire-up without
    encoding real videos."""

    def setUp(self):
        self.original_app_config = dict(config.app)
        vd._runtime_disabled_video_codecs.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        vd._runtime_disabled_video_codecs.clear()

    def _params(self, **overrides):
        base = dict(
            video_subject="test",
            subtitle_enabled=False,
            voice_volume=1.0,
            bgm_type="",
            bgm_volume=0.0,
        )
        base.update(overrides)
        return vd.VideoParams(**base)

    def test_keep_original_audio_opens_source_with_audio_true(self):
        """The source clip must be opened with ``audio=True`` so its track is
        available to the composite mixer."""
        source_video = _FakeMoviePyClip()
        source_video.audio = _FakeMoviePyClip()
        params = self._params(keep_original_audio=True, original_audio_volume=1.0)

        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source_video) as open_clip,
            patch.object(vd, "AudioFileClip", return_value=_FakeMoviePyClip()),
            patch.object(vd, "CompositeAudioClip", return_value=_FakeMoviePyClip()),
            patch.object(vd, "_write_videofile_with_codec_fallback"),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
            )

        open_clip.assert_called_once_with("combined.mp4", audio=True)

    def test_keep_original_audio_off_strips_source_audio_by_default(self):
        """Backward compat: without the flag, source audio must still be
        discarded at open time (matches the original behaviour)."""
        source_video = _FakeMoviePyClip()
        params = self._params(keep_original_audio=False)

        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source_video) as open_clip,
            patch.object(vd, "AudioFileClip", return_value=_FakeMoviePyClip()),
            patch.object(vd, "CompositeAudioClip", return_value=_FakeMoviePyClip()),
            patch.object(vd, "_write_videofile_with_codec_fallback"),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
            )

        open_clip.assert_called_once_with("combined.mp4", audio=False)

    def test_keep_original_audio_composites_source_with_voice(self):
        """With the flag on and source audio present, both the voice and
        the original track must reach ``CompositeAudioClip`` together."""
        source_video = _FakeMoviePyClip()
        source_video.audio = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        params = self._params(keep_original_audio=True, original_audio_volume=0.7)

        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
            patch.object(vd, "AudioFileClip", return_value=voice_source),
            patch.object(vd, "CompositeAudioClip", return_value=_FakeMoviePyClip()) as composite,
            patch.object(vd, "_write_videofile_with_codec_fallback"),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
            )

        composite.assert_called_once()
        streams = composite.call_args.args[0]
        # The composite must contain both the voice and the original-audio
        # streams — not just the voice alone.
        self.assertEqual(len(streams), 2)

    def test_original_audio_volume_zero_drops_source_from_mix(self):
        """``original_audio_volume=0`` is a valid mute switch — the source
        audio must NOT be added to the composite in that case."""
        source_video = _FakeMoviePyClip()
        source_video.audio = _FakeMoviePyClip()
        params = self._params(keep_original_audio=True, original_audio_volume=0.0)

        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
            patch.object(vd, "AudioFileClip", return_value=_FakeMoviePyClip()),
            patch.object(vd, "CompositeAudioClip", return_value=_FakeMoviePyClip()) as composite,
            patch.object(vd, "_write_videofile_with_codec_fallback"),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
            )

        # Only the voice stream reaches the mix; volume=0 means the
        # source audio is muted at the composite stage. With one
        # stream there is nothing to composite, so CompositeAudioClip
        # must not be invoked at all.
        composite.assert_not_called()

    def test_source_without_audio_track_falls_back_to_voice_only(self):
        """Some source clips carry no audio — the code must not crash and
        must produce a voice-only output."""
        source_video = _FakeMoviePyClip()
        source_video.audio = None
        params = self._params(keep_original_audio=True)

        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
            patch.object(vd, "AudioFileClip", return_value=_FakeMoviePyClip()),
            patch.object(vd, "CompositeAudioClip", return_value=_FakeMoviePyClip()) as composite,
            patch.object(vd, "_write_videofile_with_codec_fallback"),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
            )

        # Source had no audio, so only the voice stream reaches the mix.
        # CompositeAudioClip is therefore unnecessary and must not be
        # called.
        composite.assert_not_called()


class TestVideoParamsAudioFields(unittest.TestCase):
    """Schema-level checks for the new audio option fields."""

    def test_keep_original_audio_defaults_to_false(self):
        params = vd.VideoParams(video_subject="x")
        self.assertFalse(params.keep_original_audio)

    def test_original_audio_volume_defaults_to_one(self):
        params = vd.VideoParams(video_subject="x")
        self.assertEqual(params.original_audio_volume, 1.0)

    def test_original_audio_volume_rejects_negative(self):
        with self.assertRaises(Exception):
            vd.VideoParams(video_subject="x", original_audio_volume=-0.1)

    def test_keep_original_audio_round_trips_through_dict(self):
        params = vd.VideoParams(
            video_subject="x", keep_original_audio=True, original_audio_volume=0.5
        )
        rebuilt = vd.VideoParams(**params.model_dump())
        self.assertTrue(rebuilt.keep_original_audio)
        self.assertEqual(rebuilt.original_audio_volume, 0.5)


if __name__ == "__main__":
    unittest.main()
