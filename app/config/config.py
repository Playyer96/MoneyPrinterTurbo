import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar

import toml
from loguru import logger

from app import __version__

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"
_task_config_snapshot: ContextVar[dict[int, dict] | None] = ContextVar(
    "task_config_snapshot", default=None
)


class _SynchronizedConfig(dict):
    """Keep dict semantics while serializing runtime configuration writes."""

    def _snapshot(self) -> dict | None:
        snapshot = _task_config_snapshot.get()
        return snapshot.get(id(self)) if snapshot else None

    def __getitem__(self, key):
        snapshot = self._snapshot()
        return snapshot[key] if snapshot is not None else super().__getitem__(key)

    def __contains__(self, key):
        snapshot = self._snapshot()
        return key in snapshot if snapshot is not None else super().__contains__(key)

    def get(self, key, default=None):
        snapshot = self._snapshot()
        return snapshot.get(key, default) if snapshot is not None else super().get(key, default)

    def __iter__(self):
        snapshot = self._snapshot()
        return iter(snapshot) if snapshot is not None else super().__iter__()

    def __len__(self):
        snapshot = self._snapshot()
        return len(snapshot) if snapshot is not None else super().__len__()

    def items(self):
        snapshot = self._snapshot()
        return snapshot.items() if snapshot is not None else super().items()

    def keys(self):
        snapshot = self._snapshot()
        return snapshot.keys() if snapshot is not None else super().keys()

    def values(self):
        snapshot = self._snapshot()
        return snapshot.values() if snapshot is not None else super().values()

    def __setitem__(self, key, value):
        snapshot = self._snapshot()
        if snapshot is not None:
            snapshot[key] = value
            return
        # Each Streamlit rerun writes current widget values back to config.
        # Unchanged values need no lock; real changes remain serialized with
        # other configuration writes while submitted jobs use their own snapshot.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        snapshot = self._snapshot()
        if snapshot is not None:
            del snapshot[key]
            return
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        snapshot = self._snapshot()
        if snapshot is not None:
            snapshot.clear()
            return
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        snapshot = self._snapshot()
        if snapshot is not None:
            if default is _MISSING:
                return snapshot.pop(key)
            return snapshot.pop(key, default)
        # ``pop(key, default)`` does not mutate config when the key is absent. The
        # WebUI uses it to select defaults, which must complete during refresh.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        snapshot = self._snapshot()
        if snapshot is not None:
            return snapshot.setdefault(key, default)
        # Like __setitem__, setdefault for an existing key is read-only. Returning
        # early keeps pages reading defaults responsive during long-running tasks.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        snapshot = self._snapshot()
        if snapshot is not None:
            snapshot.update(changes)
            return
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Create a pending-update key for an in-process configuration section."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Update WebUI runtime configuration without blocking.

    Submitted jobs capture their own settings, so widget changes can apply to
    later jobs immediately. When another configuration write is in progress,
    retain only the newest value for each setting until that write completes.

    Return True when applied, False when queued.
    """
    # Queue every update before trying the config lock. Concurrent pages then
    # retain write order and an earlier thread cannot erase a newer queued value.
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            copy.deepcopy(value),
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # Callers usually save at the end of a Streamlit rerun, but that can fail
        # when a page errors or an update arrives as a task exits. A background
        # flush guarantees queued values are eventually applied.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return config_section.get(key, _MISSING) == value
    finally:
        _config_save_lock.release()


def delete_config_nonblocking(config_section, key):
    """
    Delete a WebUI configuration entry without blocking.

    “Use default” must remove a key rather than write an empty string. When a
    video task owns the lock, deletion supersedes earlier queued updates for the
    same key and runs when the task finishes.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Apply the latest pending WebUI values while holding the config write lock."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Keep the pending lock while applying so readers of the current-plus-
        # pending snapshot see a complete state before or after application.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Return the effective configuration snapshot, including pending WebUI updates.

    This lets a newly submitted task include UI updates that are waiting behind
    another configuration write without changing a task already in progress.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Apply and save all pending configuration while holding the write lock."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # The in-memory update succeeded. Keep only the save marker on failure:
        # a temporarily unwritable config must not fail the video task, and the
        # next interaction retries the save.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Wait for the current config writer, then reliably flush queued updates."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Ensure at most one background thread waits to flush configuration."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    threading.Thread(
        target=_run_deferred_config_flush,
        name="mpt-config-flush",
        daemon=True,
    ).start()


def try_save_config():
    """
    Save WebUI configuration without blocking; defer it while another save is busy.

    APIs, the CLI, and maintenance scripts retain ``save_config``'s blocking
    semantics. Only Streamlit reruns use this function to avoid page stalls.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Group configuration reads and writes into one consistent operation.

    The project defaults to a local loopback address and uses one global user
    configuration. Submitted jobs use ``capture_runtime_config`` instead; this
    lock remains for short operations that must read a consistent live config.
    """
    with _config_save_lock:
        # Apply queued updates before and after the grouped operation so callers
        # see a complete configuration state.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Try to acquire the runtime configuration lock and return immediately.

    A WebUI preview is a short user-triggered action and must not wait minutes
    for a background task. Callers can prompt a retry on failure; a successful
    lock still prevents other sessions changing providers, keys, or models.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Return whether the current process runs inside a container.

    This selects Ollama's default address:
    - locally, `localhost` is the user's machine;
    - inside Docker, it is the container and host Ollama usually needs
      `host.docker.internal`.

    `/proc/1/cgroup` also exists on normal Linux, so return True only for a
    clear container marker. Paths remain injectable for environment tests.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # The /proc/net/route gateway is little-endian hexadecimal: 010011AC means
    # 172.17.0.1. Parse it so native Linux Docker can reach host services through
    # the default gateway when host.docker.internal has no DNS record.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Read the default gateway IP from a Linux container.

    Docker Desktop usually provides `host.docker.internal`, but native Linux
    Docker may not. The default gateway is a fallback for host services; if
    Ollama listens only on 127.0.0.1, it must bind a host interface or use an
    explicit `ollama_base_url`.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 3:
            continue

        destination = fields[1]
        gateway = fields[2]
        if destination != "00000000" or gateway == "00000000":
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Return Ollama's default OpenAI-compatible base URL.

    An explicit `ollama_base_url` bypasses this. Without one, containers target
    the host and ordinary local runs target localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Load TOML while accepting repeated UTF-8 BOMs written by Windows editors.

    ``utf-8-sig`` removes only one leading BOM. Some Windows edit, archive, or
    save flows add another, causing TOML to see an invisible first-line character.
    Normalize only after a normal read fails and never rewrite the original file,
    protecting user-entered API keys.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise


def _apply_environment_overlay(loaded_config):
    """
    Lift secret-style keys from ``.env`` (gitignored) on top of the TOML
    config so an API key never lands inside ``config.toml``.

    Order of precedence (highest wins): ``os.environ`` > ``.env`` file >
    ``config.toml``. Only the keys explicitly mapped below are eligible --
    everything else stays as written in TOML.

    Kept as one tight function so the rest of the loader can stay
    decoupled from any specific provider; add a new line when a new
    service needs an API key.
    """
    env_path = os.path.join(root_dir, ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError as exc:
            logger.debug(
                f"could not read .env for env overlay: path={env_path}, "
                f"error={type(exc).__name__}: {exc}"
            )

    mappings = (
        ("app", "groq_api_key", "GROQ_API_KEY"),
        ("app", "openai_api_key", "OPENAI_API_KEY"),
        ("app", "anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("app", "elevenlabs_api_key", "ELEVENLABS_API_KEY"),
        ("app", "gemini_api_key", "GEMINI_API_KEY"),
    )
    for section_name, key_name, env_var in mappings:
        env_value = os.environ.get(env_var)
        if not env_value:
            continue
        section = loaded_config.setdefault(section_name, {})
        if isinstance(section, dict):
            section[key_name] = env_value


def load_config():
    # fix: IsADirectoryError: [Errno 21] Is a directory: '/MoneyPrinterTurbo/config.toml'
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    loaded_config = _load_toml_config(config_file)
    _apply_environment_overlay(loaded_config)

    loaded_config = _load_toml_config(config_file)
    _apply_environment_overlay(loaded_config)
    loaded_config.pop("azure", None)
    legacy_omnivoice_config = loaded_config.pop("voicestudio", None)
    if isinstance(legacy_omnivoice_config, dict):
        loaded_config.setdefault("omnivoice", legacy_omnivoice_config)
    ui_config = loaded_config.get("ui")
    if isinstance(ui_config, dict):
        if ui_config.get("tts_server") == "voicestudio":
            ui_config["tts_server"] = "omnivoice"
        voice_name = ui_config.get("voice_name")
        if isinstance(voice_name, str) and voice_name.startswith("voicestudio:"):
            ui_config["voice_name"] = voice_name.replace("voicestudio:", "omnivoice:", 1)
    app_config = loaded_config.get("app")
    if isinstance(app_config, dict):
        for key in tuple(app_config):
            if key.startswith("azure_"):
                app_config.pop(key)
    return loaded_config


def save_config():
    """
    Save runtime configuration atomically.

    Streamlit sessions can save at nearly the same time. A direct overwrite lets
    another thread read partial TOML, so serialize writes with a process-local
    reentrant lock, write a sibling temporary file, then atomically replace it.

    A Docker Desktop single-file bind mount makes config.toml the mount point and
    Linux rejects rename/replace with EBUSY. In that case overwrite in place under
    the lock; re-raise other errors so permissions, disk, and path issues remain visible.

    This keeps the existing single-user global configuration semantics and avoids
    corrupting config files during fast reruns or activity in multiple tabs.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["kokoro"] = dict(kokoro)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["omnivoice"] = dict(omnivoice)
        config_to_save["ui"] = dict(ui)
        serialized_config = toml.dumps(config_to_save)

        # A complete WebUI rerun saves here. Return unchanged content directly to
        # avoid a disk write and fsync for every ordinary control click.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _SynchronizedConfig(_cfg.get("whisper", {}))
proxy = _SynchronizedConfig(_cfg.get("proxy", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
omnivoice = _SynchronizedConfig(_cfg.get("omnivoice", {}))

# Default voice used when the WebUI/API doesn't pass one. An empty string
# keeps the previous behavior: fall through to Edge TTS. Set this in
# config.toml (or via the VOICE_NAME env var) to e.g. "omnivoice:boy_voice"
# once a cloned profile exists on the local OmniVoice service.
default_voice_name = str(_cfg.get("default_voice_name", "")).strip()
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

_TASK_CONFIG_SECTIONS = (
    app,
    whisper,
    proxy,
    siliconflow,
    minimax_tts,
    elevenlabs,
    chatterbox,
    kokoro,
    fish_audio,
    omnivoice,
    ui,
)


def capture_runtime_config() -> dict[int, dict]:
    """Capture the effective configuration used by one background task."""
    return {
        id(section): copy.deepcopy(snapshot_config_with_pending(section))
        for section in _TASK_CONFIG_SECTIONS
    }


@contextmanager
def use_runtime_config_snapshot(snapshot: dict[int, dict]):
    """Make a captured configuration visible only to the current task thread."""
    token = _task_config_snapshot.set(snapshot)
    try:
        yield
    finally:
        _task_config_snapshot.reset(token)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
