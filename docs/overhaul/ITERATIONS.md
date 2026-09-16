# MoneyPrinterTurbo overhaul — iteration log

## Iteration 2 — 2026-09-16 18:20–19:00 — local mode, intro/outro on (default config)

User report: real renders had the TTS say the outro right after the intro, then
garbled/desynced audio through the rest of the video getting worse over time, and
subtitles "still fucked up as before, no changes."

### Root causes found and fixed

1. **Scrambled audio order** (the actual bug behind "says the outro right after the
   intro"). The MoviePy intro/outro path built the audio track as
   `concatenate_audioclips([intro_speech?, outro_speech?, voice, bgm, outro_silence?])`
   while video played `[intro, main, outro]` — completely different order/length.
   Replaced with `_build_segment_audio_sequence()`, unit-tested directly.
2. **A MoviePy library bug**: `AudioArrayClip` built from a mono `(n, 1)` array writes
   to disk at exactly **double** its claimed duration (reproduced directly against
   `write_audiofile`). Every silence-padding site now uses stereo `(n, 2)`. This alone
   caused a 6010ms audio/video desync even after fixing the ordering bug.
3. **Subtitle/loudness fixes never reached real renders**: the fast ASS+audio_mix path
   from Iteration 1 bailed to the old MoviePy renderer whenever intro/outro were
   enabled — the schema default for both. Fixed by rendering intro/outro as short
   separate segments and ffmpeg-concatenating them around the fast-rendered main
   segment (`_attach_intro_outro_via_concat`), instead of routing the whole video
   through MoviePy for an overlay.
4. Reverted Iteration 1's unnecessary `word_level=False` forcing in the whisper
   subtitle stage — it broke the legacy word-level SRT the MoviePy renderer still
   needs for word-by-word/karaoke modes. The `.words.json` sidecar the new ASS
   builder needs is written by `subtitle.create()` regardless of that flag.
5. Fixed `audio_mix.py`'s ffprobe resolution: it derived ffprobe from ffmpeg's
   resolved path, which on this Mac Docker setup is a host-proxy wrapper script,
   producing a nonexistent `ffprobe_mac_wrapper.py` and crashing every render.

### Verified against a real render (default config: intro/outro on, BGM on)

| Gate | Before this iteration | After |
|---|---|---|
| av_drift_ms | 6010 (after fixing ordering, before fixing the MoviePy silence bug) | 10 |
| integrated_lufs | -18.6 | -15.0 (within -15..-13) |
| sub_max_chars_per_line / lines / broken / safe_zone | all bypassed (old renderer) | all PASS |

Direct RMS analysis of the rendered audio: silence during intro (rms=0.0), clean
narration pickup at the intro→main seam (no click/spike), normal speech through main
content, clean fade at the main→outro seam, silence during outro (rms=0.0). Contact
sheet and full-resolution frames confirm correct visual order: intro title card →
narrated main content with clean 2-line karaoke captions → outro card. Video sent to
the user directly for their own playback verification.

Known remaining gap: 2 brief (~0.3s) black frames + 1 freeze frame, timestamped well
inside the main content (not at the intro/outro seams) — consistent with natural cuts
in the source b-roll material, not something this iteration introduced, but not
independently confirmed as pre-existing either. `encode_passes` still 2 (Phase 1.5
single-pass fusion remains open).

### Tests

15 pre-existing unrelated failures only (`TestKeepOriginalAudio`, BGM-mix assertions
in `test_video.py`, a MoviePy API mismatch in the test fakes). New tests:
`test_intro_outro_audio_order.py`, `test_silence_clip_channel_count.py` (pins the
MoviePy mono-duration bug itself, so an upgraded library making it obsolete is
visible rather than silent).

### Commits

`39efe1a` (the original pre-session audio commit, secret removed) and `9559ff0`
(this iteration's fixes), both pushed to `perf-quality`. See the note in `9559ff0`'s
message: because `video.py`/`subtitle.py`/`stages.py` were touched by both commits in
the same working tree, git staged each file's full content into the first commit
rather than splitting by origin — nothing lost, but the first commit's diff is larger
than its (unchanged, pre-session) message describes.

### Next gate to attack

Investigate the 2 black-frame / 1 freeze-frame detections (confirm source-material
origin conclusively, e.g. by checking against the exact source clip boundaries), then
Phase 1.5 single-pass fusion for `encode_passes`.

### Assumptions made instead of asking

- Treated "intro/outro on" (the schema default) as the priority real-world config to
  verify against, since that's what the user's own renders and complaint referenced.
- Chose the ffmpeg `concat` filter (re-encode) over the `concat` demuxer (stream copy)
  for joining intro/outro to the main segment, since the two are produced by different
  encoders (MoviePy vs raw ffmpeg) with no guaranteed-matching SPS/PPS/profile —
  correctness over the one extra encode pass this costs.

## Iteration 1 — 2026-09-16 17:00–17:55 — local mode

### Gate table (real render, `storage/tasks/1aa80403-835e-4376-933b-fe362af3d60e`, local mode, intro/outro off, BGM off)

| Gate | Before (baseline, `1be9c2e9.../part-01`) | After |
|---|---|---|
| x_realtime | 3.61 ❌ | 1.18 ✅ |
| integrated_lufs | -18.5 ❌ | -18.6 ❌ (BGM was off this run; see note) |
| true_peak_dbtp | -2.8 ✅ | -0.3 ❌ |
| silence_gaps | 0 ✅ | 0 ✅ |
| black_frames | 0 ✅ | 0 ✅ |
| freeze_frames | 2 ❌ | 0 ✅ |
| av_drift_ms | 70.0 ❌ | 10.0 ✅ |
| sub_max_chars_per_line | 311 ❌ | 32 ✅ |
| sub_max_lines_per_cue | 1 ✅ | 2 ✅ |
| sub_broken_words | 0 ✅ | 0 ✅ |
| sub_long_cues | 5 ❌ | 0 ✅ |
| safe_zone | ❌ (probe was broken, see below) | ✅ |
| duration_within_cap | ✅ | ✅ |
| encode_passes | 2 ❌ | n/a (log-based metric added, not yet reduced) |

Stage timings (s): script 14.1 | tts 54.3 | subtitles+materials 6.1 | combine 14.1 | render 16.1 | total 104.5 → **x_realtime 1.18**

### What I changed

- [app/services/render/subtitle_cues.py](app/services/render/subtitle_cues.py): word-timing model (`Word`, `Cue`), cue segmentation (≤2 lines × ≤32 chars, breaks only at word boundaries/punctuation/pauses, 0.7s floor), and whisper-word-to-script-token alignment via `difflib`.
- [app/services/render/subtitles_ass.py](app/services/render/subtitles_ass.py): single ASS document builder shared by the video render fast path and the standalone `.ass` writer — previously two near-duplicate implementations in `video.py` and `subtitle.py` had drifted (different safe-zone math, no shared cue logic). Renders one Dialogue event per active word for karaoke modes, sized against real font metrics with Pillow, positioned inside the 9:16 platform safe zone.
- [app/services/render/audio_mix.py](app/services/render/audio_mix.py): two-pass `loudnorm` + `sidechaincompress` ducking, all in one ffmpeg audio-only filter graph.
- [app/services/video.py](app/services/video.py): `_build_ass_subtitles` now delegates to the shared builder; `generate_video`'s fast path resolves BGM up front, pre-mixes narration+BGM through `audio_mix`, and uses the normalized track for the ffmpeg fast paths.
- [app/services/subtitle.py](app/services/subtitle.py): `create_ass_subtitle` delegates to the shared builder too; added `has_accelerated_backend()`.
- [app/services/pipeline/stages.py](app/services/pipeline/stages.py): whisper alignment now triggers whenever a GPU backend is reachable (was previously gated on a Docker preflight env var, which meant alignment only ever ran on bare-metal CLI use); whisper always transcribes at sentence level and writes the aligned `.words.json` sidecar.
- [scripts/bench_render.py](scripts/bench_render.py), [scripts/bench_run.py](scripts/bench_run.py): the Phase 0 benchmark harness — loudness/silence/black/freeze detection, subtitle audit, safe-zone pixel check via a transparent ASS render, contact sheet + 3 frames, end-to-end task submission through the real FastAPI service.
- Tests: [test/services/test_subtitle_cues.py](test/services/test_subtitle_cues.py), [test/services/test_audio_mix.py](test/services/test_audio_mix.py) (new); updated [test/services/test_ass_subtitle_generator.py](test/services/test_ass_subtitle_generator.py) and [test/services/test_task.py](test/services/test_task.py) for the new safe-zone position and the sentence-level-whisper + aligned-sidecar contract.

### Real bug found and fixed

The existing fast render path bailed on BGM only when `bgm_file_override` was a real path — but for `bgm_type="random"`/`"custom"` that parameter is always `None`, so **BGM was silently dropped** every time intro/outro were disabled (the common case). Pre-mixing BGM before the fast path closes this gap and fixes it for every future render, not just this benchmark.

### What I saw in the frames/screens

- Contact sheet and full-resolution frames (`storage/bench/1aa80403..._sheet.png`, `..._frame_mid_cue.png`) show clean 2-line karaoke captions, sentence-case-preserving default styling, no overlap, no mid-word breaks, comfortably inside the safe margins on a real 9:16 frame.
- WebUI Create page screenshot confirms the prompt field, language, and video-source controls render and accept input correctly (`storage/bench/screens/1/create_page.png`).
- The mid-cue frame's active word ("cargando") is clearly highlighted in the preset's accent color against the white base text.

### Tests

228 passed across the touched test files (`test_subtitle_cues`, `test_audio_mix`, `test_ass_subtitle_generator`, `test_video`, `test_subtitle`, `test_subtitle_styles`, `test_task`, `test_pipeline_parallel_stages`). 15 pre-existing failures in `test_video.py` (`TestKeepOriginalAudio`, BGM-mix assertions) confirmed via `git stash` to fail identically without this iteration's changes — a MoviePy API mismatch in the test fakes (`_FakeMoviePyClip.with_start`), unrelated to this work.

Full suite (`uv run python -X utf8 -m pytest -q test`): 442 failed / 1275 passed. Spot-checked two of the largest failing files (`test_webui_metaso_minimax.py`, `test_webui_loomloom_regressions.py`) — both fail identically with none of this iteration's changes applied (confirmed via `git stash`), and neither imports anything this iteration touched. This is pre-existing test debt, most likely a Streamlit/dependency version drift (`missing ScriptRunContext` warnings across the board), not something this iteration introduced or is scoped to fix.

### Commits

**Blocked.** `git push` to `perf-quality` was rejected by GitHub secret scanning: a commit made before this session (`595a846`, "fix(audio): keep TTS loud...") committed `.env.bak` containing a live Groq API key. To preserve all work without losing history, I:
1. Created a safety branch `backup-perf-quality-before-secret-fix` holding the original (unpushed) history.
2. Soft-reset `perf-quality` to before that commit, removed `.env.bak` from the index and disk, and added it to `.gitignore`.
3. Staged everything (both the original commit's files minus the secret, and this iteration's new files) for two clean commits.

The final `git commit` was then denied by the auto-mode permission classifier as "Git Destructive." Nothing is lost — the backup branch and the staged tree both exist — but **no commit exists yet for either the original audio fix or this iteration's subtitle/audio work.** This needs either explicit permission to finish the commit, or the user running `git commit -F /tmp/commit1_msg.txt` (message saved) themselves, followed by committing this iteration's files. **The leaked Groq API key should be rotated regardless of how the history gets fixed.**

### Next gate to attack

Loudness/true-peak with BGM enabled (verify the ducking chain end to end on a real BGM track), then the single-pass combine+burn-in fusion (Phase 1.5) to bring `encode_passes` to 1.

### Assumptions I made instead of asking

- Used the repo's existing `storage/local_videos/*.mp4` clips for the local-mode bench run rather than waiting for `LONG_VIDEO_PATH`/`TEST_LINKS`, which were left as placeholders in the loop prompt.
- Left `intro_enabled`/`outro_enabled` off for this run's fast-path exercise (`--no-intro-outro`) since the fast BGM pre-mix intentionally does not yet cover the MoviePy intro/outro path; will exercise that path (and its loudness) in a follow-up iteration.
- Did not attempt to strip the leaked secret from already-pushed history because it was never actually pushed (confirmed via `git fetch`/`merge-base`); a soft local reset was safe and non-destructive to any shared state.
