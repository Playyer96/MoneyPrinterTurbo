import json
import math
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any, Iterable
from uuid import uuid4

from loguru import logger

from app.models import const


def get_response(status: int, data: Any = None, message: str = ""):
    obj = {
        "status": status,
    }
    if data:
        obj["data"] = data
    if message:
        obj["message"] = message
    return obj


def to_json(obj):
    try:
        # Define a helper function to handle different types of objects
        def serialize(o):
            # If the object is a serializable type, return it directly
            if isinstance(o, (int, float, bool, str)) or o is None:
                return o
            # If the object is binary data, convert it to a base64-encoded string
            elif isinstance(o, bytes):
                return "*** binary data ***"
            # If the object is a dictionary, recursively process each key-value pair
            elif isinstance(o, dict):
                return {k: serialize(v) for k, v in o.items()}
            # If the object is a list or tuple, recursively process each element
            elif isinstance(o, (list, tuple)):
                return [serialize(item) for item in o]
            # If the object is a custom type, attempt to return its __dict__ attribute
            elif hasattr(o, "__dict__"):
                return serialize(o.__dict__)
            # Return None for other cases (or choose to raise an exception)
            else:
                return None

        # Use the serialize function to process the input object
        serialized_obj = serialize(obj)

        # Serialize the processed object into a JSON string
        return json.dumps(serialized_obj, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"failed to serialize object to json: {str(e)}")
        return None


def get_uuid(remove_hyphen: bool = False):
    u = str(uuid4())
    if remove_hyphen:
        u = u.replace("-", "")
    return u


_CLIP_SPEED_MIN = 0.5
_CLIP_SPEED_MAX = 2.0


def normalize_clip_speed(value, default: float = 1.0) -> float:
    """Normalize clip playback speed to the safe range supported by the WebUI."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return default

    # NaN bypasses ordinary comparisons and propagates when MoviePy calculates
    # duration; infinity is also invalid user input. Fall back to the default for
    # both so API and internal callers cannot create an invalid timeline. Zero and
    # negative values cannot represent a valid playback speed either.
    if not math.isfinite(speed) or speed <= 0:
        return default

    return min(max(speed, _CLIP_SPEED_MIN), _CLIP_SPEED_MAX)


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def storage_dir(sub_dir: str = "", create: bool = False):
    d = os.path.join(root_dir(), "storage")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if create and not os.path.exists(d):
        os.makedirs(d)

    return d


def resource_dir(sub_dir: str = ""):
    d = os.path.join(root_dir(), "resource")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    return d


def task_dir(sub_dir: str = ""):
    d = os.path.join(storage_dir(), "tasks")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def font_dir(sub_dir: str = ""):
    d = resource_dir("fonts")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def song_dir(sub_dir: str = ""):
    d = resource_dir("songs")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def public_dir(sub_dir: str = ""):
    d = resource_dir("public")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def _running_in_docker() -> bool:
    # /.dockerenv is the canonical Docker signal; the cgroup check is the
    # fallback for runtimes that omit it (Podman, some rootless setups).
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(errors="ignore")
    except OSError:
        return False
    return "docker" in cgroup or "containerd" in cgroup or "podman" in cgroup


def _mac_docker_wrapper_path() -> str | None:
    """
    Return the path to the host-ffmpeg forwarding wrapper if (and only if)
    we are running inside a Docker container that has been configured to use
    the Mac host's ffmpeg via the LaunchAgent proxy.

    Detection is opt-in: the env var `FFMPEG_MAC_PROXY_URL` must be set, AND
    we must be inside a container. On any other setup (native macOS, Linux
    host, non-Mac Docker) this returns None so the normal ffmpeg lookup
    proceeds.
    """
    proxy_url = os.environ.get("FFMPEG_MAC_PROXY_URL", "").strip()
    if not proxy_url or not _running_in_docker():
        return None
    # The wrapper ships with the repo. The bind mount `./:/MoneyPrinterTurbo`
    # makes it visible at the same path on host and container.
    wrapper = Path("/MoneyPrinterTurbo/scripts/ffmpeg_mac_wrapper.py")
    if wrapper.is_file():
        return str(wrapper)
    # Fallback for native macOS where the repo lives somewhere else.
    repo_wrapper = Path(__file__).resolve().parent.parent.parent / "scripts" / "ffmpeg_mac_wrapper.py"
    if repo_wrapper.is_file():
        return str(repo_wrapper)
    return None


def get_ffmpeg_binary() -> str:
    """
    Resolve the FFmpeg executable for the current process.

    Rationale:
    1. Video encoding, silent-audio generation, and pydub transcoding depend on FFmpeg.
    2. Windows portable bundles, Docker, and custom installations often have different PATHs.
    3. Central resolution gives every caller the same priority and avoids one path finding
       FFmpeg while another cannot.

    Priority:
    0. Inside Docker with FFMPEG_MAC_PROXY_URL: use the Mac host FFmpeg forwarding wrapper
       for VideoToolbox (see scripts/ffmpeg_mac_proxy.py).
    1. IMAGEIO_FFMPEG_EXE: explicit MoviePy/imageio configuration.
    2. ffmpeg from the system PATH.
    3. The binary bundled by imageio-ffmpeg.
    4. The string "ffmpeg", allowing subprocess to expose a more specific runtime error.
    """
    wrapper = _mac_docker_wrapper_path()
    if wrapper:
        return wrapper

    configured_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured_ffmpeg:
        return configured_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg:
            return bundled_ffmpeg
    except Exception as exc:
        logger.warning(f"failed to resolve bundled ffmpeg binary: {str(exc)}")

    return "ffmpeg"


_FFMPEG_INSTALL_HINT = (
    "Install FFmpeg on your system, or set app.ffmpeg_path in config.toml to "
    "the full path of an ffmpeg executable (e.g. downloaded from "
    "https://www.gyan.dev/ffmpeg/builds/)."
)


def check_ffmpeg_ready(timeout: int = 10) -> bool:
    """
    Check that FFmpeg is available before video generation begins.

    Previously, missing or unusable FFmpeg surfaced only during video composition or
    silent-track generation, often after most of a task had run, and the error offered
    no remedy. Probe once in the shared ``app/services/task.py:_run_pipeline`` path so
    API, CLI, and WebUI fail early with an actionable message.

    This lightweight ``-version`` call does not download anything or change the main
    flow. Callers must treat False as a hard precondition failure because the pinned
    imageio-ffmpeg==0.6.0 does not download a usable binary on demand.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    try:
        completed = subprocess.run(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.warning(
            f"no usable ffmpeg executable found (tried: {ffmpeg_bin}). "
            f"{_FFMPEG_INSTALL_HINT}"
        )
        return False
    except Exception as exc:
        logger.warning(
            f"failed to probe ffmpeg ({ffmpeg_bin}): {exc}. {_FFMPEG_INSTALL_HINT}"
        )
        return False

    if completed.returncode != 0:
        logger.warning(
            f"ffmpeg ({ffmpeg_bin}) probe exited with status {completed.returncode}; "
            f"video generation may fail later. {_FFMPEG_INSTALL_HINT}"
        )
        return False

    logger.info(f"ffmpeg check passed, using: {ffmpeg_bin}")
    return True


def run_in_background(func, *args, **kwargs):
    def run():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"run_in_background error: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=False)
    thread.start()
    return thread


def time_convert_seconds_to_hmsm(seconds) -> str:
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    milliseconds = int(seconds * 1000) % 1000
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)


def text_to_srt(idx: int, msg: str, start_time: float, end_time: float) -> str:
    start_time = time_convert_seconds_to_hmsm(start_time)
    end_time = time_convert_seconds_to_hmsm(end_time)
    srt = """%d
%s --> %s
%s
        """ % (
        idx,
        start_time,
        end_time,
        msg,
    )
    return srt


def str_contains_punctuation(word):
    for p in const.PUNCTUATIONS:
        if p in word:
            return True
    return False


def split_string_by_punctuations(s):
    result = []
    txt = ""

    previous_char = ""
    next_char = ""
    for i in range(len(s)):
        char = s[i]
        if char == "\n":
            result.append(txt.strip())
            txt = ""
            continue

        if i > 0:
            previous_char = s[i - 1]
        if i < len(s) - 1:
            next_char = s[i + 1]

        if char == "." and previous_char.isdigit() and next_char.isdigit():
            # # In the case of "withdraw 10,000, charged at 2.5% fee", the dot in "2.5" should not be treated as a line break marker
            txt += char
            continue

        if char == "," and previous_char.isdigit() and next_char.isdigit():
            # A thousands separator in an English number is not a sentence break,
            # for example "1,000 years". Edge TTS normally returns the whole number
            # as one word boundary; splitting it into "1" and "000 years" prevents
            # subtitle aggregation from matching the script and incorrectly falls
            # back to Whisper.
            txt += char
            continue

        if char not in const.PUNCTUATIONS:
            txt += char
        else:
            result.append(txt.strip())
            txt = ""
    result.append(txt.strip())
    # filter empty string
    result = list(filter(None, result))
    return result


PAUSE_TAG_KEYWORDS = (
    r"pause|pausa|silence|silencio|silêncio|silenzio|stille|"
    r"пауза|тишина|停顿|暂停|静音|ポーズ|一時停止|無音|일시중지|정지"
)
# Match every bracketed or parenthesized pause tag regardless of argument
# validity, ensuring invalid tags such as [pause: -2s], [pause: nope], and
# [pause: 0s] are removed before synthesis and never reach TTS.
PAUSE_TAG_PATTERN = re.compile(
    rf"[\[\(]\s*(?:{PAUSE_TAG_KEYWORDS})\b(?:\s*[:：]?\s*([^\]\)]*?))?\s*[\]\)]",
    re.IGNORECASE,
)

# Pause-duration safety limits in seconds:
# clamp positive pauses below 0.1 seconds (100 ms) to 0.1 seconds; remove
# non-positive or non-numeric pause tags without speaking or generating silence;
# clamp pauses above 10.0 seconds to the safe maximum.
MIN_PAUSE_DURATION_SECONDS = 0.1
MAX_PAUSE_DURATION_SECONDS = 10.0


def has_pause_tags(text: str) -> bool:
    """Return whether the text contains a pause tag."""
    if not text:
        return False
    return bool(PAUSE_TAG_PATTERN.search(text))


def remove_pause_tags(text: str) -> str:
    """
    Remove every valid and invalid pause tag from script text.

    Strip these non-spoken markers before subtitle splitting, LLM keyword
    extraction, or TTS so unhandled tags are neither spoken nor used as visual
    search terms.
    """
    if not text:
        return ""
    cleaned = PAUSE_TAG_PATTERN.sub(" ", text)
    # Collapse consecutive horizontal spaces while preserving newlines.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def parse_script_with_pauses(text: str) -> list[tuple[str, Any]]:
    """
    Parse speech and pause tags from a script.

    Consecutive pause tags are merged. Invalid tags, including non-numeric or
    non-positive durations, are removed and never sent to TTS as speech. Pauses
    below MIN_PAUSE_DURATION_SECONDS are raised to the minimum, and pauses above
    MAX_PAUSE_DURATION_SECONDS are clamped to the safe maximum.

    Returns:
        Ordered tuples such as [("speech", "copy"), ("pause", 2.0), ...].
    """
    if not text:
        return []

    segments: list[tuple[str, Any]] = []
    last_idx = 0

    for match in PAUSE_TAG_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_idx:
            speech_text = text[last_idx:start].strip()
            if speech_text:
                segments.append(("speech", speech_text))

        raw_arg = match.group(1)
        if raw_arg is None or not raw_arg.strip():
            # Default to a 1.0-second pause when no argument is given.
            duration = 1.0
        else:
            raw_arg_str = raw_arg.strip()
            num_match = re.match(
                r"^([+-]?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|ms|msec|msecs|秒|毫秒)?$",
                raw_arg_str,
                re.IGNORECASE,
            )
            if not num_match:
                logger.warning(
                    f"invalid pause tag value '{raw_arg_str}', tag removed and ignored"
                )
                last_idx = end
                continue

            val = float(num_match.group(1))
            unit = (num_match.group(2) or "s").lower()
            duration = val / 1000.0 if ("ms" in unit or "毫秒" in unit) else val

        if duration <= 0:
            logger.warning(
                f"invalid non-positive pause duration {duration}s, tag removed and ignored"
            )
            last_idx = end
            continue

        if duration < MIN_PAUSE_DURATION_SECONDS:
            logger.warning(
                f"pause duration {duration:.3f}s is below minimum limit of {MIN_PAUSE_DURATION_SECONDS}s (100ms), "
                f"clamped to {MIN_PAUSE_DURATION_SECONDS}s"
            )
            duration = MIN_PAUSE_DURATION_SECONDS

        if duration > MAX_PAUSE_DURATION_SECONDS:
            logger.warning(
                f"pause duration {duration}s exceeds maximum limit of {MAX_PAUSE_DURATION_SECONDS}s, "
                f"clamped to {MAX_PAUSE_DURATION_SECONDS}s"
            )
            duration = MAX_PAUSE_DURATION_SECONDS

        # Merge consecutive pause tags to avoid fragmented silence files.
        if segments and segments[-1][0] == "pause":
            merged_duration = min(
                segments[-1][1] + duration, MAX_PAUSE_DURATION_SECONDS
            )
            segments[-1] = ("pause", merged_duration)
        else:
            segments.append(("pause", duration))

        last_idx = end

    if last_idx < len(text):
        speech_text = text[last_idx:].strip()
        if speech_text:
            segments.append(("speech", speech_text))

    return segments


def normalize_script_for_subtitle_matching(video_script: str) -> str:
    """
    Clean script text before matching subtitles.

    Users may enter Markdown separators, heading emphasis, or `_` formatting.
    These characters normally do not appear in TTS/Whisper recognition. Keeping
    them during line matching can create more script lines than subtitle lines and
    produce `00:00:00,000 --> 00:00:00,000`, which editors cannot import as SRT.
    """
    video_script = remove_pause_tags(video_script or "")
    underscore_count = video_script.count("_")
    video_script = video_script.replace("_", "")
    cleaned_lines = []
    removed_separator_lines = 0
    for line in video_script.splitlines():
        line = line.strip()
        # Standalone Markdown separators or emphasis are not spoken by TTS; remove
        # them so subtitle aggregation does not stall on an unpronounceable target.
        if re.fullmatch(r"[-*_]{3,}", line):
            removed_separator_lines += 1
            continue
        cleaned_lines.append(line)

    normalized_script = "\n".join(cleaned_lines).strip()
    if underscore_count or removed_separator_lines:
        logger.debug(
            "normalized script for subtitle matching, "
            f"removed underscores: {underscore_count}, "
            f"removed markdown separator lines: {removed_separator_lines}"
        )
    return normalized_script


def md5(text):
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def resolve_ui_language(
    saved_language: str | None,
    browser_locale: str | None,
    supported_languages: Iterable[str],
    default_language: str = "en",
) -> str:
    """
    Select the UI language by saved setting, browser locale, then default.

    Browsers often return regional locales such as ``zh-CN`` and ``pt-BR``, while
    language files use base codes such as ``zh`` and ``pt``. Try the full locale,
    then the code before the hyphen. Keep this function pure for easy testing.
    """
    supported = [str(language).strip() for language in supported_languages]
    supported_by_lower = {
        language.lower(): language for language in supported if language
    }

    def match_language(value: str | None) -> str | None:
        normalized = str(value or "").strip().replace("_", "-").lower()
        if not normalized:
            return None
        if normalized in supported_by_lower:
            return supported_by_lower[normalized]
        base_language = normalized.split("-", 1)[0]
        return supported_by_lower.get(base_language)

    saved_match = match_language(saved_language)
    if saved_match:
        return saved_match

    browser_match = match_language(browser_locale)
    if browser_match:
        return browser_match

    default_match = match_language(default_language)
    if default_match:
        return default_match

    # A normal project always contains English. Keep an empty-set fallback so a
    # damaged locale directory does not crash page initialization; translation
    # functions can continue showing raw keys for diagnosis.
    return supported[0] if supported else default_language


@lru_cache(maxsize=8)
def load_locales(i18n_dir):
    # Streamlit reruns the script after every WebUI interaction, while locale files
    # do not change at runtime. Cache parsing to avoid repeatedly reading every
    # i18n JSON file.
    _locales = {}
    for root, dirs, files in os.walk(i18n_dir):
        for file in files:
            if file.endswith(".json"):
                lang = file.split(".")[0]
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    _locales[lang] = json.loads(f.read())
    return _locales


def parse_extension(filename):
    return Path(filename).suffix.lower().lstrip('.')
