import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Uploads are bounded by disk, not by a byte cap. Validation still decodes the
# whole track, so the FFmpeg timeout scales with the file instead of rejecting
# large but valid audio up front.
VALIDATION_BASE_TIMEOUT_SECONDS = 30
# ponytail: 10 MB/s decode floor; raise it if slow hardware trips the timeout.
VALIDATION_BYTES_PER_SECOND = 10 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
# MoviePy decodes background music through FFmpeg, so there is no reason to
# restrict this to MP3. Only mainstream, unambiguous audio extensions are
# allowed, so a video container such as MP4 cannot be uploaded as music. The
# tuple is also the single source of truth for the WebUI uploader, so adding or
# removing a format never leaves the front end and back end out of sync.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """The uploaded file does not meet the background music safety or format requirements."""


class BgmServiceError(RuntimeError):
    """A server-side failure such as FFmpeg or the filesystem being unavailable."""


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Decide once whether the current task needs any background music at all.

    the rule is provider-independent: with no source selected, an invalid
    volume, or a volume of 0, random, custom, Sonilo and any future provider
    must all skip file parsing, external generation and the final mix. Keeping
    it in the shared BGM service avoids copying a zero-volume check into every
    new provider.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Return the persistent directory for user background music.

    builtin songs are code resources and stay in resource/songs; user uploads
    are runtime data and must live under the Docker-mounted storage dir so they
    survive a container rebuild and never pollute the Git working tree.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """Best-effort cleanup of a staged upload that never masks the caller's original exception."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # staged files use a reserved prefix and never enter the BGM list. a
        # failed cleanup must not mask a more precise original error such as
        # "invalid audio", but it must leave the path and OS error in the log.
        logger.warning(
            f"failed to remove staged background music: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Extract a cross-platform-safe audio filename and reject invalid names and unsupported extensions."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(ord(character) < 32 for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("invalid background music filename")

    # Windows treats the first segment before the extension as a device name,
    # so CON.mp3 and LPT1.wav cannot be created as ordinary files. even though
    # the server stores under a UUID, rejecting these names up front keeps API
    # input behaviour identical across platforms.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("invalid background music filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"unsupported background music format; supported formats: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: float | None = None) -> None:
    """
    Verify the file holds a fully decodable audio stream using only the FFmpeg the project is configured with.

    the project allows imageio-ffmpeg to supply a portable FFmpeg, and that
    install does not guarantee FFprobe, so no extra binary dependency may be
    added. `-map 0:a:0` fails when there is no audio stream and `-xerror`
    promotes decode errors to failures; a full decode also catches encrypted
    files or random data that happens to hit an audio frame header. extra
    streams such as cover art are fine, only the first audio stream is
    checked.
    """
    if timeout_seconds is None:
        try:
            size_bytes = os.path.getsize(file_path)
        except OSError:
            size_bytes = 0
        timeout_seconds = VALIDATION_BASE_TIMEOUT_SECONDS + (
            size_bytes / VALIDATION_BYTES_PER_SECOND
        )

    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("FFmpeg background music validation timed out") from exc
    except OSError as exc:
        raise BgmServiceError("failed to run FFmpeg for background music validation") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("uploaded file must contain a decodable audio stream")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Check that an on-disk audio file decodes completely with the project FFmpeg.

    upload preflight sizes its own timeout from the file; Sonilo tracks run up
    to 6 minutes, so this shared entry point takes an explicit timeout. the
    service depends on FFmpeg only and never requires a system FFprobe.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("background music file is empty or missing")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Write the upload stream to a same-directory temp file and return the safe name, temp path and byte count.

    the WebUI preflight and the final persist must use exactly the same chunked
    read and filename rules, otherwise the UI can show a file as usable and the
    server can still reject it on generate. the caller deletes or atomically
    replaces the temp file once audio probing finishes.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("failed to prepare background music storage") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("background music upload is not seekable") from exc

        # keep the original extension so FFmpeg picks the right demuxer for
        # headerless formats such as raw AAC. the temp file stays in the target
        # directory so the final os.replace is atomic.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("background music upload must be binary")
                total_bytes += len(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("background music file is empty")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("failed to stage background music upload") from exc
        raise
    finally:
        # Streamlit reuses the same UploadedFile for in-browser preview, so
        # rewinding keeps the player and the final save from reading nothing.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Fully validate uploaded audio without persisting it, for the WebUI preflight before showing "ready"."""
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"background music upload validated: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Save user background music via chunked writes and an atomic replace.

    callers include FastAPI UploadFile and Streamlit UploadedFile, both of which
    expose a binary file interface. writing and validating a same-directory temp
    file before os.replace avoids leaving half an audio file behind on a
    concurrent upload or an interrupted process, and gives same-named uploads
    distinct UUID storage keys so queued or running tasks keep referencing the
    original immutable file.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("failed to persist background music upload") from exc
        temp_path = ""
        logger.info(
            f"background music uploaded: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def _list_bgm_files(directories: tuple[str, ...]) -> list[str]:
    """Enumerate safe, supported background music files in directory priority order."""
    files_by_name: dict[str, str] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # both preflight and final save briefly create same-directory temp
            # files. they carry a valid audio extension but are not yet
            # validated, so the random BGM list must not pick them up.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # enumerated results need the same real-path check, otherwise
                # an attacker could drop an audio symlink to an outside file in
                # an allowed directory and get it handed to MoviePy as random
                # BGM.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"skip unsafe background music file: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def list_builtin_bgm_files() -> list[str]:
    """
    List the background music shipped with the project.

    the WebUI "preset songs" control and settings preset import/export use only
    this list, so a saved filename restores on another machine running the same
    version; user uploads stay under custom music.
    """
    return _list_bgm_files((utils.song_dir(),))


def list_bgm_files() -> list[str]:
    """List available uploaded and builtin background music; uploads win on a name collision."""
    return _list_bgm_files((utils.song_dir(), uploaded_bgm_dir(create=True)))


def resolve_builtin_bgm_file(unsafe_path: str) -> str:
    """Resolve builtin background music by filename, rejecting paths, unknown files and user uploads."""
    if not unsafe_path:
        raise ValueError("background music filename is required")

    filename = str(unsafe_path)
    if filename != os.path.basename(filename):
        raise ValueError("preset background music must use a filename")

    files_by_name = {
        os.path.basename(file_path): file_path for file_path in list_builtin_bgm_files()
    }
    if filename not in files_by_name:
        raise ValueError("preset background music is not available")
    return files_by_name[filename]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Resolve BGM in the upload and builtin song directories, rejecting any path outside those two allowlists.

    a bare filename hits the upload directory first, while legacy forms such as
    `output000.mp3`, an absolute allowlisted path and
    `./resource/songs/output000.mp3` keep working. new uploads use a UUID, so
    they normally cannot collide with a builtin song or an earlier upload.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("unsupported background music path")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("background music file does not exist")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
