# Run the stack with `make up`, stop it with `make down`, check it with
# `make status`. The GPU stack is chosen automatically: docker/detect-stack.sh
# reports whether this host is a Mac, has the NVIDIA container runtime, or has
# an AMD/ROCm device node, and the right override is layered in. Setting
# COMPOSE_FILE (in .env or the environment) overrides the detection.
#
# Windows has no make; docker\up.ps1 does the same detection there.
#
# macOS does not pass Metal through to Docker's Linux VM, and MLX has no Linux
# build at all, so the model server has to run on the host to reach the GPU.
# `make up` treats it as part of the stack: it starts the host server if it is
# not answering, then brings the containers up. On Linux there is no host
# server and `make up` is just `docker compose up -d`.

PLIST := $(HOME)/Library/LaunchAgents/com.moneyprinterturbo.omnivoice.plist
FFMPEG_PROXY_PLIST := $(HOME)/Library/LaunchAgents/com.moneyprinterturbo.ffmpeg-proxy.plist
REPO  := $(shell pwd)
UID   := $(shell id -u)

.PHONY: up down status mac-setup mac-teardown mac-status

mac-setup:
	@test -x "$(REPO)/.venv/bin/python" || { \
		echo "creating .venv for the GPU server..."; \
		uv venv --python 3.11 && \
		uv pip install --python "$(REPO)/.venv/bin/python" \
			-r vendor/omnivoice/requirements.txt; }
	# Idempotent: pin refresh + a known-bad dep check before bootstrapping.
	# If you ever see the LaunchAgent crash-loop on omnivoice import, this is
	# what fixes it -- run `make mac-setup` again.
	@uv pip install --python "$(REPO)/.venv/bin/python" --quiet \
		-r vendor/omnivoice/requirements.txt
	@if ! "$(REPO)/.venv/bin/python" -c \
		"import tokenizers; v=tuple(int(x) for x in tokenizers.__version__.split('.')[:2]); assert (0,23) <= v < (0,24)" 2>/dev/null; then \
		echo "tokenizers out of range, pinning..."; \
		uv pip install --python "$(REPO)/.venv/bin/python" --quiet \
			'tokenizers>=0.23.1,<0.24.0'; \
	fi
	@mkdir -p "$(HOME)/Library/LaunchAgents" "$(REPO)/storage/logs"
	@printf '%s\n' \
	  '<?xml version="1.0" encoding="UTF-8"?>' \
	  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
	  '<plist version="1.0"><dict>' \
	  '  <key>Label</key><string>com.moneyprinterturbo.omnivoice</string>' \
	  '  <key>ProgramArguments</key><array>' \
	  '    <string>$(REPO)/scripts/omnivoice-runner.sh</string>' \
	  '  </array>' \
	  '  <key>EnvironmentVariables</key><dict>' \
	  '    <key>OMNIVOICE_HOST</key><string>0.0.0.0</string>' \
	  '    <key>OMNIVOICE_PORT</key><string>8780</string>' \
	  '    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>' \
	  '  </dict>' \
	  '  <key>WorkingDirectory</key><string>$(REPO)</string>' \
	  '  <key>RunAtLoad</key><true/>' \
	  '  <key>KeepAlive</key><true/>' \
	  '  <key>ProcessType</key><string>Interactive</string>' \
	  '  <key>LowPriorityIO</key><false/>' \
	  '  <key>StandardOutPath</key><string>$(REPO)/storage/logs/omnivoice.log</string>' \
	  '  <key>StandardErrorPath</key><string>$(REPO)/storage/logs/omnivoice.log</string>' \
	  '</dict></plist>' > "$(PLIST)"
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.omnivoice 2>/dev/null || true
	@# bootout is asynchronous; bootstrapping too soon fails with EBUSY.
	@sleep 2
	@launchctl bootstrap gui/$(UID) "$(PLIST)"
	@echo "waiting for the GPU server..."
	@for i in $$(seq 1 40); do \
		curl -sf --max-time 2 http://127.0.0.1:8780/health >/dev/null && \
			{ echo "GPU server ready on :8780 (starts automatically at login)"; break; }; \
		sleep 1; done; \
	if ! curl -sf --max-time 2 http://127.0.0.1:8780/health >/dev/null; then \
		echo "no response on :8780; see storage/logs/omnivoice.log" >&2; exit 1; \
	fi

	# ffmpeg host proxy: lets containers on this Mac reach the host's
	# VideoToolbox-enabled ffmpeg. Loopback only; the container talks to it
	# via docker-compose.mac.yml's socat bridge. Same venv Python because
	# the proxy only uses the stdlib.
	@printf '%s\n' \
	  '<?xml version="1.0" encoding="UTF-8"?>' \
	  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
	  '<plist version="1.0"><dict>' \
	  '  <key>Label</key><string>com.moneyprinterturbo.ffmpeg-proxy</string>' \
	  '  <key>ProgramArguments</key><array>' \
	  '    <string>$(REPO)/.venv/bin/python</string>' \
	  '    <string>$(REPO)/scripts/ffmpeg_mac_proxy.py</string>' \
	  '  </array>' \
	  '  <key>EnvironmentVariables</key><dict>' \
	  '    <key>MPT_FFMPEG_PROXY_HOST</key><string>127.0.0.1</string>' \
	  '    <key>MPT_FFMPEG_PROXY_PORT</key><string>8781</string>' \
	  '    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>' \
	  '  </dict>' \
	  '  <key>WorkingDirectory</key><string>$(REPO)</string>' \
	  '  <key>RunAtLoad</key><true/>' \
	  '  <key>KeepAlive</key><true/>' \
	  '  <key>ProcessType</key><string>Interactive</string>' \
	  '  <key>LowPriorityIO</key><false/>' \
	  '  <key>StandardOutPath</key><string>$(REPO)/storage/logs/ffmpeg-mac-proxy.log</string>' \
	  '  <key>StandardErrorPath</key><string>$(REPO)/storage/logs/ffmpeg-mac-proxy.log</string>' \
	  '</dict></plist>' > "$(FFMPEG_PROXY_PLIST)"
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null || true
	@sleep 2
	@launchctl bootstrap gui/$(UID) "$(FFMPEG_PROXY_PLIST)"
	@for i in $$(seq 1 20); do \
		curl -sf --max-time 2 http://127.0.0.1:8781/health >/dev/null && \
			{ echo "ffmpeg host proxy ready on :8781"; break; }; \
		sleep 1; done; \
	if ! curl -sf --max-time 2 http://127.0.0.1:8781/health >/dev/null; then \
		echo "no response on :8781; see storage/logs/ffmpeg-mac-proxy.log" >&2; exit 1; \
	fi

mac-teardown:
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.omnivoice 2>/dev/null || true
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null || true
	@rm -f "$(PLIST)" "$(FFMPEG_PROXY_PLIST)"
	@echo "GPU server and ffmpeg proxy removed; they will not start at login anymore"

mac-status:
	@launchctl print gui/$(UID)/com.moneyprinterturbo.omnivoice 2>/dev/null \
		| grep -E "state|pid" || echo "omnivoice: LaunchAgent not installed"
	@curl -sf --max-time 2 http://127.0.0.1:8780/health || echo "(omnivoice: server not answering)"
	@grep -i "loading OmniVoice" storage/logs/omnivoice.log 2>/dev/null | tail -1 || true
	@launchctl print gui/$(UID)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null \
		| grep -E "state|pid" || echo "ffmpeg-proxy: LaunchAgent not installed"
	@curl -sf --max-time 2 http://127.0.0.1:8781/health || echo "(ffmpeg-proxy: server not answering)"

# --- the whole stack, GPU included -------------------------------------------

# Apple hosts need the host-side model server; everywhere else the containers
# are the whole stack. uname decides, so one target works on both.
IS_MAC := $(shell [ "$$(uname -s)" = "Darwin" ] && echo 1)

# Exported so every `docker compose` below -- and any bare `docker compose` the
# user types in the same shell -- agrees on which overrides are in play.
export COMPOSE_FILE := $(shell sh docker/detect-stack.sh)

# `docker compose watch <service>` rebuilds + recreates a container when any
# of its declared `develop.watch` paths change. webui and api pull in this
# repo, so editing Python files inside ``app/``, ``webui/``, ``main.py``
# reaches the running UI without a manual restart. ``config.toml`` is
# excluded deliberately: the WebUI persists widget state to it on every
# keystroke, so watching it would restart the containers on every
# checkbox toggle and wipe session state.
watch:
	@docker compose watch webui api

watch-polling:
ifdef IS_MAC
	@echo "Watching app/ webui/ main.py (no compose needed; config.toml excluded so widget edits do not restart the UI)"
	@st=$$(mktemp -t mpt-watch); touch $$st; \
	trap 'rm -f $$st' EXIT; \
	while true; do \
		find app webui main.py -type f -newer $$st 2>/dev/null \
			| head -1 | grep -q . && { \
				touch $$st; \
				echo "[watch-polling] $$(date +%H:%M:%S) change detected, restarting webui+api"; \
				docker compose restart webui api 2>/dev/null || echo "(docker not running; keep editing, the next 'make up' picks it up)"; \
			}; \
		sleep 1; \
	done
else
	@echo "watch-polling is macOS-specific"
	@exit 1
endif

up:
ifdef IS_MAC
	@curl -sf --max-time 2 http://127.0.0.1:8780/health >/dev/null 2>&1 \
		|| $(MAKE) --no-print-directory mac-setup
endif
	@echo "stack: $(COMPOSE_FILE)"
	@docker compose up -d
	@$(MAKE) --no-print-directory status
	@echo
	@echo "Docker stack up. To reload on save run 'make watch' in another shell."
	@echo "Everything (docker + host launchd + auto-rebuild) is reachable from 'make up' / 'make down'."

# Single cohesive command: bring everything up AND keep auto-rebuilding
# on every save until Ctrl-C. ``up-and-watch`` is the day-to-day entry
# point -- the user never types ``make watch`` plus ``make up`` separately.
up-and-watch: up
	@echo "watching for changes (Ctrl-C to stop); omnivoice reloads every 1s, webui/api rebuild via compose watch"
ifdef IS_MAC
	@st=$$(mktemp -t mpt-watch); touch $$st; \
	trap 'rm -f $$st' EXIT INT; \
	while true; do \
		find vendor/omnivoice scripts/omnivoice-runner.sh app webui main.py .mpt-host/bin \
			-type f -newer $$st 2>/dev/null | head -1 | grep -q . && { \
			touch $$st; \
			echo "[up-and-watch] $$(date +%H:%M:%S) change detected, waiting 2s for edits to settle"; \
			sleep 2; \
			echo "[up-and-watch] reloading everything now"; \
			launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.omnivoice 2>/dev/null; \
			launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null; \
			docker compose restart webui api 2>/dev/null || true; \
			touch $$st; \
		}; \
		sleep 1; \
	done
else
	@docker compose watch webui api
endif

# Watch the Python source files of the macOS host services and kick the
# launchd agents on every save so `vendor/omnivoice/server.py` (or any
# other editable file) reaches the running process without a manual
# `launchctl kickstart``. fswatch is the preferred path; ``up-and-watch``
# polls ``find -newer`` on the same paths every second for users who
# do not have fswatch installed.
mac-reload:
ifndef IS_MAC
	@echo "mac-reload is macOS-only"
	@exit 1
endif
	@command -v fswatch >/dev/null 2>&1 || { \
		echo "fswatch not found; run 'brew install fswatch' or use 'make mac-reload-poll' instead"; exit 1; }
	@echo "watching: vendor/omnivoice scripts .mpt-host/bin (Ctrl-C to stop)"
	@fswatch -o \
		vendor/omnivoice \
		scripts/omnivoice-runner.sh \
		.mpt-host/bin \
		| xargs -n1 -I{} sh -c '\
			echo "[mac-reload] $$(date +%H:%M:%S) change detected, kicking services"; \
			launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.omnivoice 2>/dev/null; \
			launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null; \
		'

# Same as mac-reload but without fswatch; polls the same paths every
# second and kicks the launchd agents when ``find -newer`` reports a
# change. Lives behind the same Reload path so users who do not want
# to install fswatch still get a working one-target workflow.
mac-reload-poll:
ifndef IS_MAC
	@echo "mac-reload-poll is macOS-only"
	@exit 1
endif
	@echo "polling: vendor/omnivoice scripts .mpt-host/bin (Ctrl-C to stop)"
	@st=$$(mktemp -t mpt-reload); touch $$st; \
	trap 'rm -f $$st' EXIT; \
	while true; do \
		find vendor/omnivoice scripts/omnivoice-runner.sh .mpt-host/bin -type f -newer $$st 2>/dev/null \
			| head -1 | grep -q . && { \
				touch $$st; \
				echo "[mac-reload-poll] $$(date +%H:%M:%S) change detected, kicking services"; \
				launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.omnivoice 2>/dev/null; \
				launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null; \
			}; \
		sleep 1; \
	done

down:
	@docker compose down
ifdef IS_MAC
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.omnivoice 2>/dev/null || true
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null || true
endif
	@echo "everything down (docker stack + mac host launchd agents)"

# Cohesive rebuild — the "I edited code and want a clean slate" path.
# Tear down every container, network, volume, locally-built image, then
# pull every base image fresh, build from scratch, and bring the stack
# back up. ``--remove-orphans`` drops stale containers from previous
# compose files; ``--rmi all`` drops the locally-built app/omnivoice
# images so the ``build --no-cache --pull`` step below actually rebuilds
# them; ``--volumes`` drops the named volumes (omnivoice_models,
# omnivoice_profiles, omnivoice_output) so the next run starts with an
# empty cache instead of stale profile/weight state. The Mac host
# launchd agents are also kicked so the new omnivoice image (when
# running under the ``container-tts`` profile on Linux) and the host
# GPU server (on Mac) both end up on the freshly-pulled tree.
rebuild:
	@echo "stopping everything"
	@docker compose down --remove-orphans --rmi local --volumes 2>/dev/null || true
ifdef IS_MAC
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.omnivoice 2>/dev/null || true
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null || true
endif
	@echo "pulling every base image"
	@docker compose pull --ignore-pull-failures
	@echo "rebuilding from scratch (--no-cache --pull of base images)"
	@docker compose build --no-cache --pull
	@echo "bringing the stack back up"
	@$(MAKE) --no-print-directory up

rebuild-and-watch: rebuild
ifdef IS_MAC
	@echo "rebuild complete; watching for changes (Ctrl-C to stop)"
	@st=$$(mktemp -t mpt-watch); touch $$st; \
	trap 'rm -f $$st' EXIT INT; \
	while true; do \
		find vendor/omnivoice scripts/omnivoice-runner.sh app webui main.py .mpt-host/bin \
			-type f -newer $$st 2>/dev/null | head -1 | grep -q . && { \
			touch $$st; \
			echo "[rebuild-and-watch] $$(date +%H:%M:%S) change detected, waiting 2s for edits to settle"; \
			sleep 2; \
			echo "[rebuild-and-watch] reloading everything now"; \
			launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.omnivoice 2>/dev/null; \
			launchctl kickstart -k gui/$$(id -u)/com.moneyprinterturbo.ffmpeg-proxy 2>/dev/null; \
			docker compose restart webui api 2>/dev/null || true; \
			touch $$st; \
		}; \
		sleep 1; \
	done
else
	@docker compose watch webui api
endif

status:
	@docker compose ps --format "  {{.Name}}\t{{.Status}}"
ifdef IS_MAC
	@printf "  moneyprinterturbo-omnivoice\thost process, "
	@curl -sf --max-time 2 http://127.0.0.1:8780/health >/dev/null 2>&1 \
		&& echo "Up ($$(grep -i 'loading OmniVoice' storage/logs/omnivoice.log 2>/dev/null \
			| tail -1 | sed 's/.* on //;s/ \.\.\..*//' || echo 'not loaded yet'))" \
		|| echo "DOWN -- run: make mac-setup"
endif
