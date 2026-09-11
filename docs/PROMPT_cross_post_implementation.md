# PROMPT — Enable TikTok, Instagram Reels, and YouTube Shorts cross-posting in MoneyPrinterTurbo

> **Purpose of this document.** MoneyPrinterTurbo already has a partial
> cross-posting path that funnels every platform through the third-party
> **Upload-Post API**. YouTube and Instagram are *technically* supported
> there but are not exposed in the WebUI credential panel, no per-platform
> tests exist, and there is no way to enter direct platform credentials.
> Use this prompt to ship a complete, production-quality cross-posting
> feature: credential entry, end-to-end publish to all three platforms,
> status visibility, error recovery, and tests.

---

## 1. Mission

Deliver a working, user-facing cross-posting feature that takes any video
produced by `app/services/pipeline/single.py` (or a chapter produced by
`app/services/pipeline/series.py`) and publishes it to **TikTok**,
**Instagram Reels**, and **YouTube Shorts** without requiring the user
to leave the app.

The feature must:

1. Accept platform credentials (API keys, OAuth tokens, account
   handles) directly inside the WebUI — no hand-editing `config.toml`.
2. Persist those credentials to `config.toml` on the server side so
   they survive restarts, with secrets treated like every other secret
   in this repo (kept out of source control).
3. Publish every generated video to the selected platforms after
   generation completes, without blocking the API response or
   retroactively changing the video task's success state on failure.
4. Surface upload progress and per-platform results in both the WebUI
   task panel and the HTTP API.
5. Survive process restarts cleanly: no orphaned "uploading" tasks, no
   lost results.

---

## 2. Current state (read this first)

You **must** understand the existing implementation before changing it.
The pieces already in place:

| File | Role |
| --- | --- |
| `app/services/upload_post.py` | `UploadPostService` singleton + `cross_post_video()` helper. Wraps `https://api.upload-post.com/api/upload`. Accepts `tiktok`, `instagram`, `youtube`, `facebook`, `twitter`, `linkedin`, `threads`, `pinterest`, `reddit`, `telegram`, `snapchat`, `bluesky` as platform ids. |
| `config.example.toml:478-498` | `upload_post_*` keys: enabled, api_key, username, platforms, auto_upload, youtube_privacy_status, max_pending_tasks. |
| `app/services/pipeline/single.py:207-263` | After `generate_final_videos`, sets `cross_post_state=PENDING` and calls `_schedule_cross_post` only if configured and `auto_upload`. |
| `app/services/pipeline/series.py` | Series chapters delegate to `single.run_single_video` per part; cross-post runs once per chapter. |
| `app/services/task.py:125-630` | `ThreadPoolExecutor(max_workers=2)`, `BoundedSemaphore` queue, per-task `Future` registry, recovery on startup via `recover_interrupted_cross_posts()`, retry/backoff on state writes, `generate_social_metadata` call for platform-specific titles/captions/hashtags. |
| `app/models/const.py:34-37` | `CROSS_POST_STATE_PENDING / PROCESSING / COMPLETE / FAILED`. |
| `app/models/schema.py:320-324` | `cross_post_state`, `cross_post_results`, `cross_post_error` fields on the task record. |
| `app/services/llm.py:1206-1425` | `SOCIAL_PLATFORMS` table with per-platform title/caption/hashtag caps; `generate_social_metadata()` builds platform-aware copy. |
| `app/controllers/v1/video.py:105,310` | Strips `cross_post_owner` from public responses; logs `cross_post_state`. |
| `webui/Main.py:3199-3269` | Cross-platform Publishing panel in WebUI — exposes `upload_post_enabled`, `upload_post_auto_upload`, `upload_post_api_key`, `upload_post_username`, `upload_post_platforms`, `upload_post_youtube_privacy_status`. **No per-platform direct credential fields exist.** |
| `webui/Main.py:651,1074` | Startup recovery call + per-task `cross_post_state` display. |

**What is missing:**

1. The WebUI panel is labelled "Cross-platform Publishing" but only
   surfaces *Upload-Post* credentials. There is no UI for entering
   direct TikTok / Instagram / YouTube credentials, even though the
   documentation talks about all three.
2. There is no `instagram` / `tiktok` / `youtube` direct upload path.
   Everything goes through Upload-Post as a single third-party hop.
3. `app/services/upload_post.py` exists but its `check_status()`
   helper is dead code (no callers — `task.py` never polls the
   Upload-Post `/api/uploadposts/status` endpoint).
4. No retry/backoff inside `upload_video()` itself. A transient 5xx
   from Upload-Post fails the upload immediately and writes
   `cross_post_state=FAILED` even though Upload-Post may eventually
   succeed server-side.
5. There are no tests under `test/` that exercise `UploadPostService`
   or the cross-post scheduling path. The existing test suite covers
   `task.py` end-to-end with monkey-patched services, not this path.

---

## 3. Pick a path before writing code

There are two valid end states. **Choose one and stick with it** —
mixing them mid-feature creates a config nightmare.

### Path A — Stay on Upload-Post (recommended for shipping fast)

Upload-Post already abstracts all three platforms behind one API key.
The work here is hardening the existing path:

- Expose per-platform toggles in the WebUI and per-platform
  fallbacks (if Upload-Post is unavailable, fail soft and tell the
  user).
- Add the missing YouTube-only fields (`youtube_title`,
  `youtube_description`, `tags[]`, `privacyStatus`,
  `containsSyntheticMedia`) to the request and verify they reach
  YouTube correctly.
- Wire `UploadPostService.check_status()` into a small background
  poller so the task record updates from `processing` → `complete`
  for the slower platforms (TikTok and Instagram processing can take
  minutes after the HTTP call returns).
- Add retry-with-backoff for `upload_video()` (3 attempts, 5s → 15s).
- Write the WebUI for the credentials that already exist; do **not**
  add direct-platform fields.

**Pros:** days of work, not weeks; one set of credentials; Upload-Post
handles per-platform quirks and rate limits; works today.

**Cons:** another third-party in the chain; per-platform metadata
shapes still go through Upload-Post's flattened schema.

### Path B — Direct platform SDKs (recommended for long-term control)

Replace `UploadPostService` with three thin service classes:

- `app/services/tiktok_uploader.py` — TikTok Content Posting API.
  Requires a registered TikTok developer app, `client_key`,
  `client_secret`, and per-account OAuth refresh tokens. Videos are
  uploaded via the *creator API* which mandates that the app be in
  *Content Posting API* access tier — sandbox access is not enough
  for production.
- `app/services/instagram_uploader.py` — Instagram Graph API
  (`/{ig-user-id}/media` + `/{ig-user-id}/media_publish`). Requires a
  Facebook App with Instagram Graph API product enabled, a long-lived
  user access token, and an Instagram Business or Creator account.
- `app/services/youtube_uploader.py` — YouTube Data API v3
  (`videos.insert` with `uploadType=multipart`). Requires a Google
  Cloud project, OAuth 2.0 Web/Desktop flow with the
  `youtube.upload` and `youtube.force-ssl` scopes, and a refresh
  token stored server-side.

Each service exposes:

```python
class PlatformUploader(Protocol):
    name: str                       # "tiktok" | "instagram" | "youtube"
    is_configured() -> bool
    upload(video_path: str, *, title: str, caption: str,
           hashtags: list[str], privacy: str = "public") -> UploadResult
    check_status(upload_id: str) -> UploadResult
```

A single orchestrator (`app/services/cross_post.py`) replaces
`UploadPostService` and fans out to whichever platform uploaders are
configured. The pipeline's existing state machine, semaphore, and
recovery code in `task.py` is reused unchanged.

**Pros:** no third-party in the loop; per-platform features unlocked
(end screens, scheduled publish, polls); easier debugging.

**Cons:** weeks of work; per-platform OAuth dance; per-platform
content policy compliance (YouTube "Made for Kids" field, TikTok
sandbox vs production, Instagram Reels aspect ratio restrictions);
three credential sets in `config.toml`.

---

## 4. Detailed plan (Path A assumed below — adapt for Path B)

### 4.1 Configuration schema

Extend `config.example.toml` section `[app]` (lines 478-498) and add
matching getters in `app/services/upload_post.py`. Keys must be
flat, ASCII-quoted, and follow the existing snake_case convention.

```toml
# Master switch — must be true for any cross-post to run.
upload_post_enabled = false

# Single third-party API key from upload-post.com.
upload_post_api_key = ""

# Single account handle inside Upload-Post (their internal user id).
upload_post_username = ""

# Restrict to a subset of Upload-Post's supported ids. The default
# already covers TikTok + Instagram; YouTube Shorts is opt-in because
# it requires a separate API setup inside Upload-Post.
# Allowed values: tiktok, instagram, youtube, facebook, twitter,
# linkedin, threads, pinterest, reddit, telegram, snapchat, bluesky.
upload_post_platforms = ["tiktok", "instagram", "youtube"]

# After a successful video generation, automatically kick off
# cross-posting. When false, expose a manual "Publish" button in the
# WebUI task panel that calls the HTTP API.
upload_post_auto_upload = false

# YouTube-only. Mirrored to Upload-Post as `privacyStatus`.
# Allowed: "public" | "unlisted" | "private".
upload_post_youtube_privacy_status = "public"

# Per-platform retries and timeout (Path A hardening).
upload_post_max_attempts = 3
upload_post_retry_base_seconds = 5
upload_post_request_timeout_seconds = 300

# Concurrent and queued cross-post jobs per process.
upload_post_max_pending_tasks = 10

# Polling: Upload-Post returns a request_id immediately but the actual
# platform-side processing happens async. Poll every N seconds until
# the platform reports a terminal state or T seconds elapse.
upload_post_poll_interval_seconds = 10
upload_post_poll_timeout_seconds = 1800

# Optional webhook to receive Upload-Post completion events without
# polling. Empty disables webhooks. The endpoint must verify the
# shared secret in `X-Upload-Post-Signature`.
upload_post_webhook_url = ""
upload_post_webhook_secret = ""
```

### 4.2 Credential storage

Follow the existing rule: write real keys to `config.toml`, leave
`config.example.toml` empty. Never log `upload_post_api_key`,
`upload_post_webhook_secret`, or any value returned by the Upload-Post
API that includes those credentials (sanitize before logging).

When writing from the WebUI, use the existing
`_set_runtime_config("app", key, value)` helper in
`webui/Main.py:3199` and persist through the same mechanism other
fields use. Reject `config.toml` writes that contain characters the
TOML parser cannot read (`"`-escapes, control characters).

### 4.3 Service layer

Harden `app/services/upload_post.py`:

1. **Retry.** Wrap the `requests.post` call in `_upload_with_retry()`
   that retries on `requests.exceptions.RequestException` and HTTP
   429/5xx up to `upload_post_max_attempts`, sleeping
   `upload_post_retry_base_seconds * 2 ** (attempt - 1)` between
   tries. Drop the upload entirely after the last failure.
2. **Polling.** `UploadPostService` exposes `poll_status(request_id)`
   that calls `check_status` until the response reports
   `success=True` or `failed=True`, or the timeout elapses. The
   pipeline worker (`task.py:_run_cross_post`) calls this after the
   initial upload returns `success=True` so the task record reflects
   the actual platform-side outcome rather than Upload-Post's
   "received" status.
3. **Per-platform metadata.** Move the per-platform field shaping
   out of `upload_video()` into helpers (`_tiktok_payload(title,
   caption, hashtags)`, `_youtube_payload(...)`). Each helper returns
   the form fields Upload-Post expects for that platform, plus the
   shared `title`/`privacy_level`/`platform[]` fields every platform
   gets.
4. **Strict typing.** Replace the loose `dict` returns with a
   `UploadResult` dataclass: `success: bool`, `request_id: str |
   None`, `platform_statuses: dict[str, str]`, `error: str | None`,
   `raw: dict`. This stops `task.py` from indexing into the response
   ad-hoc.
5. **Test seams.** Accept an injected `session` argument so tests
   can pass a `requests.Session` mock without monkey-patching
   `requests`.

### 4.4 Pipeline integration

The existing wiring in `app/services/pipeline/single.py:207-263` is
correct in shape. Adjustments:

1. After `_schedule_cross_post` returns `None`, kick off the polling
   loop inside the worker thread (`_run_cross_post`) so the
   `cross_post_state` stays `processing` until Upload-Post reports
   the platform-side result.
2. When polling reports a per-platform failure, write
   `cross_post_results[i].success = False` and include the platform
   id (`"tiktok"`, `"instagram"`, `"youtube"`) so the UI can show
   "TikTok: ✓, Instagram: ✗ (rate limited), YouTube: ✓".
3. On the API response, return the **video generation** result
   synchronously with `cross_post_state=processing`. Do not wait for
   upload completion in the HTTP request — that already happens, but
   add a unit test asserting the HTTP response returns within 2s of
   the pipeline reaching `_schedule_cross_post`.
4. In `series.py`, the per-chapter `run_single_video` already
   schedules a cross-post. Confirm the `cross_post_results` array is
   aggregated to the series-level task, not just per chapter.

### 4.5 WebUI

The panel at `webui/Main.py:3199-3269` already covers the Upload-Post
fields. Add:

1. **Manual "Publish" button** in the task panel
   (`webui/Main.py:1074`) when `upload_post_enabled=true` but
   `upload_post_auto_upload=false`. The button calls the HTTP API
   `POST /api/v1/video/{task_id}/publish` and updates the displayed
   `cross_post_state`.
2. **Per-platform status row** below each task card. Read
   `task.get("cross_post_results", [])` and render a tick/cross per
   platform with the error message on hover.
3. **"Test connection" button** inside the cross-platform panel.
   Calls `GET /api/v1/upload-post/test` which calls
   `UploadPostService.check_status` with a dummy id (or a real
   `request_id` if the user just uploaded) and reports back whether
   Upload-Post authenticated successfully.
4. **Tooltip text** linking to `https://docs.upload-post.com/` for
   each field. Copy from the docs, do not paraphrase the API itself.
5. **i18n.** Add English and Chinese strings under `webui/i18n/`.
   Match the existing key naming (`cross_post_*`). Never edit the
   existing Asian-language comments in `webui/Main.py` — only add
   new code.

### 4.6 HTTP API

Add three endpoints under `app/controllers/v1/`:

```
POST /api/v1/video/{task_id}/publish
    Body: {"platforms": ["tiktok","instagram","youtube"]?,"force": bool?}
    Action: Reschedule a cross-post for a task whose
            cross_post_state is None, FAILED, or COMPLETE.
            Returns 409 if a cross-post is already active.
    Auth: x-api-key header (already enforced by middleware).

GET  /api/v1/video/{task_id}/cross-post
    Action: Return cross_post_state, cross_post_results,
            cross_post_error as JSON. Refreshes status from
            Upload-Post if state == processing.

GET  /api/v1/upload-post/test
    Action: Verify the configured Upload-Post API key is valid by
            making a no-op status check. Returns
            {"configured": bool, "valid": bool, "platforms": [...]}.
```

Wire them through the existing `router.py` and the controller
blueprint pattern. Keep response shapes consistent with the rest of
the v1 API (snake_case keys, ISO timestamps, no PII in errors).

### 4.7 Error handling

Define the failure surface so the WebUI can map errors to user-facing
messages:

| Condition | UI message | State |
| --- | --- | --- |
| `upload_post_enabled=false` | "Cross-platform publishing is off" | no-op |
| `is_configured()` false | "Add your Upload-Post API key in Settings" | no-op |
| `platforms` empty | "Pick at least one platform" | no-op |
| `requests` 401 | "Upload-Post rejected the API key" | FAILED |
| `requests` 429 | "Upload-Post rate limited — retry in Ns" | FAILED, retry |
| `requests` 5xx | "Upload-Post had a server error — retrying" | PROCESSING |
| `poll_status` timeout | "Upload-Post did not finish in N seconds" | FAILED |
| Per-platform `failed=True` | "TikTok rejected the video: <reason>" | FAILED (partial) |

The worker writes the error reason into `cross_post_error`. Do **not**
write `cross_post_state=FAILED` for transient (retryable) errors —
keep `PROCESSING` and let the retry loop settle.

### 4.8 Tests

Add the following under `test/`:

```
test/services/test_upload_post.py
    - test_is_configured_returns_false_when_keys_missing
    - test_upload_retries_on_5xx
    - test_upload_does_not_retry_on_4xx
    - test_upload_raises_after_max_attempts
    - test_poll_status_succeeds_after_two_attempts
    - test_poll_status_times_out
    - test_tiktok_payload_shapes_correctly
    - test_youtube_payload_includes_contains_synthetic_media

test/services/test_cross_post_pipeline.py
    - test_pipeline_schedules_cross_post_when_configured
    - test_pipeline_skips_cross_post_when_auto_upload_false
    - test_pipeline_returns_pending_state_in_api_response
    - test_pipeline_does_not_block_video_return
    - test_recovery_marks_orphaned_cross_posts_failed

test/api/test_video_publish_endpoint.py
    - test_publish_returns_409_when_cross_post_active
    - test_publish_returns_202_when_scheduled
    - test_publish_requires_x_api_key
    - test_get_cross_post_status_returns_results
```

Use `unittest.mock` with the injected `session` seam from §4.3. No
live HTTP calls to Upload-Post in tests. The fixtures live in
`test/conftest.py` if needed.

### 4.9 Documentation

1. Add a "Cross-platform Publishing" section to `README-en.md` and
   `README.md` explaining the Upload-Post setup steps. Match the
   tone of the existing "API Service" section.
2. Add a Chinese translation under the same heading in the
   Chinese-language README files.
3. Add the endpoint signatures to `docs/MoneyPrinterTurbo.ipynb`
   under the API reference cell, if one exists.
4. Update the WebUI screenshot in `docs/webui-en.jpg` if the panel
   changes shape.

---

## 5. Files to touch (Path A)

Mandatory:

- `app/services/upload_post.py` — retry, polling, dataclass return,
  injected session.
- `app/services/pipeline/single.py` — pass through per-platform
  results from the worker.
- `app/services/task.py` — call `poll_status` in the worker before
  writing terminal state.
- `app/controllers/v1/video.py` — new endpoints.
- `app/models/schema.py` — extend `cross_post_results` schema with
  `platform_statuses: dict[str, str]` field.
- `config.example.toml` — new keys from §4.1.
- `webui/Main.py` — manual publish button, per-platform status row,
  test connection button.
- `webui/i18n/*.json` — new strings (en + zh).
- `README-en.md`, `README.md`, `README-ja.md` — new section.
- `test/services/test_upload_post.py` — new.
- `test/services/test_cross_post_pipeline.py` — new.
- `test/api/test_video_publish_endpoint.py` — new.

Do not touch unless required:

- `app/services/llm.py` — the social metadata generator already
  handles YouTube Shorts and Instagram Reels shapes. Only revisit
  if a new platform id is added.
- `app/services/pipeline/stages.py` — no changes needed; the
  pipeline ends at `generate_final_videos`.
- `app/services/pipeline/series.py` — only if cross-post
  aggregation behaviour is wrong.

---

## 6. Out of scope

- Direct TikTok / Instagram / YouTube SDKs (Path B). If the user
  wants those, run this prompt again with §3 set to Path B and the
  same §4 sections re-targeted at per-platform service classes.
- Encrypted credential storage. The repo stores secrets in
  `config.toml` plaintext today. Adding a keyring-backed secret
  store is a separate change.
- Multi-account support. Upload-Post is single-account by design;
  per-account fan-out is a different product.
- Scheduling posts for a future time. Upload-Post publishes
  immediately; deferred publishing is out of scope.
- Per-platform analytics. Upload-Post's status endpoint reports
  upload state, not views or engagement.
- Webhook receiver implementation. Webhook configuration is exposed
  but the actual `POST /upload-post-webhook` Flask/ASGI route is a
  separate task. Stub the field for now.

---

## 7. Open questions for the user

Resolve these **before** writing code:

1. **Path A or Path B?** Path A ships in days; Path B in weeks. The
   rest of this prompt assumes Path A.
2. **Single-account vs multi-account.** Should the WebUI support
   more than one Upload-Post username? Today the schema is single.
3. **Per-platform toggles or one global switch?** The current schema
   has a single `upload_post_platforms` list. Do you also want a
   per-platform enable/disable so the user can say "TikTok off, IG
   on, YouTube on" without editing TOML?
4. **Manual publish UX.** Should the "Publish" button be inside the
   task card, the settings panel, or both?
5. **Auto-upload default.** Should `upload_post_auto_upload` default
   to `false` (manual) or `true` (fire-and-forget)? Current default
   is `false`; keep it.
6. **i18n.** English + Chinese already ship. Any other locale for the
   new strings? (Japanese README exists but the WebUI ships English
   + Chinese only — confirm before adding Japanese strings.)
7. **Retry budget.** `max_attempts=3` and `retry_base_seconds=5` are
   sensible defaults. Confirm or override.

---

## 8. Acceptance criteria

The feature is done when:

- [ ] `config.toml` has all `upload_post_*` keys documented in
      §4.1, with empty defaults in `config.example.toml`.
- [ ] `UploadPostService.upload_video` retries on 5xx/429 and stops
      cleanly on 4xx.
- [ ] `UploadPostService.poll_status` updates `cross_post_results`
      for each platform that Upload-Post eventually reports on.
- [ ] The WebUI panel shows the Upload-Post credentials, a "Test
      connection" button, and per-platform status rows under each
      task card.
- [ ] A manual "Publish" button works for tasks that have already
      completed video generation.
- [ ] HTTP endpoints from §4.6 return the documented shapes.
- [ ] Tests from §4.8 pass with no live HTTP calls.
- [ ] README files explain the setup flow with a link to
      `https://docs.upload-post.com/`.
- [ ] Process restart with active uploads does not leave tasks in
      `processing` state forever — `recover_interrupted_cross_posts`
      marks them `failed` within one startup cycle.
- [ ] No secrets are logged at any level (`api_key`, response bodies
      containing tokens).

---

## 9. Reading order for the implementer

1. Read `app/services/upload_post.py` end to end.
2. Read `app/services/pipeline/single.py:207-263`.
3. Read `app/services/task.py:125-630`.
4. Read `webui/Main.py:3199-3269`.
5. Read `config.example.toml:478-498`.
6. Then start implementing from §4.1.

If anything in §2 (current state) no longer matches the file you
just read, **stop and update this prompt first** — the rest of the
plan depends on the file map being accurate.