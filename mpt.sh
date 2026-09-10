#!/usr/bin/env bash
# Single entrypoint for MoneyPrinterTurbo: the containers AND the host-side
# GPU server that serves TTS + whisper.
#
# The GPU server has to run outside Docker -- a Linux container never sees
# Apple Metal, so torch's MPS backend and MLX are both unreachable from
# inside. That means `docker compose down` alone leaves a multi-GB python
# process running on the host, which is what this script exists to prevent.
#
#   ./mpt.sh up       start GPU server + containers
#   ./mpt.sh down     stop containers + GPU server (use this, not compose down)
#   ./mpt.sh status   what is running
#   ./mpt.sh tts      run the GPU server in the foreground (ctrl-c to stop)
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
PIDFILE=".voicestudio.pid"
LOGFILE="/tmp/voicestudio-native.log"
PORT="${VOICESTUDIO_PORT:-8780}"

server_pid() {
  # The port is the source of truth: a pidfile can go stale across reboots,
  # and a server started by hand has no pidfile at all.
  lsof -ti:"$PORT" 2>/dev/null || true
}

start_server() {
  if [ -n "$(server_pid)" ]; then
    echo "GPU server already up on :$PORT"
    return
  fi
  [ -x "$PY" ] || { echo "no .venv; run: uv venv && uv pip install -r vendor/voice_studio/requirements.txt" >&2; exit 1; }
  echo "starting GPU server on :$PORT (log: $LOGFILE)"
  VOICESTUDIO_HOST=0.0.0.0 VOICESTUDIO_PORT="$PORT" \
    nohup "$PY" vendor/voice_studio/server.py > "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "GPU server ready"; return; }
    sleep 1
  done
  echo "GPU server did not answer /health in 30s; see $LOGFILE" >&2
  exit 1
}

stop_server() {
  local pid
  pid="$(server_pid)"
  if [ -z "$pid" ]; then
    echo "GPU server not running"
  else
    echo "stopping GPU server (pid $pid)"
    kill $pid 2>/dev/null || true
    for _ in $(seq 1 10); do
      [ -z "$(server_pid)" ] && break
      sleep 1
    done
    # It holds the model in memory; SIGKILL it if it ignored SIGTERM.
    [ -n "$(server_pid)" ] && kill -9 $(server_pid) 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
}

case "${1:-up}" in
  up)
    start_server
    docker compose up -d
    ;;
  down)
    docker compose down
    stop_server
    ;;
  status)
    docker compose ps --format 'table {{.Name}}\t{{.Status}}'
    pid="$(server_pid)"
    if [ -n "$pid" ]; then
      echo "GPU server: up (pid $pid, :$PORT)"
      grep -i "loading OmniVoice" "$LOGFILE" 2>/dev/null | tail -1 || true
    else
      echo "GPU server: down"
    fi
    ;;
  tts)
    # Foreground: ctrl-c stops it, no stray process left behind.
    exec env VOICESTUDIO_HOST=0.0.0.0 VOICESTUDIO_PORT="$PORT" "$PY" vendor/voice_studio/server.py
    ;;
  *)
    echo "usage: $0 {up|down|status|tts}" >&2
    exit 1
    ;;
esac
