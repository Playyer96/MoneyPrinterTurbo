# One-time host setup for Apple Silicon. Everything else is plain
# `docker compose` -- there is no make target to run, build or stop the app.
#
# macOS does not pass Metal through to Docker's Linux VM, so the model server
# has to run on the host to reach the GPU. These targets install it as a
# LaunchAgent, which starts it at login and restarts it if it dies, so you
# never launch it by hand.

PLIST := $(HOME)/Library/LaunchAgents/com.moneyprinterturbo.voicestudio.plist
REPO  := $(shell pwd)
UID   := $(shell id -u)

.PHONY: mac-setup mac-teardown mac-status

mac-setup:
	@test -x "$(REPO)/.venv/bin/python" || { \
		echo "creating .venv for the GPU server..."; \
		uv venv --python 3.11 && \
		uv pip install --python "$(REPO)/.venv/bin/python" \
			-r vendor/voice_studio/requirements.txt; }
	@mkdir -p "$(HOME)/Library/LaunchAgents" "$(REPO)/storage/logs"
	@printf '%s\n' \
	  '<?xml version="1.0" encoding="UTF-8"?>' \
	  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
	  '<plist version="1.0"><dict>' \
	  '  <key>Label</key><string>com.moneyprinterturbo.voicestudio</string>' \
	  '  <key>ProgramArguments</key><array>' \
	  '    <string>$(REPO)/.venv/bin/python</string>' \
	  '    <string>$(REPO)/vendor/voice_studio/server.py</string>' \
	  '  </array>' \
	  '  <key>EnvironmentVariables</key><dict>' \
	  '    <key>VOICESTUDIO_HOST</key><string>0.0.0.0</string>' \
	  '    <key>VOICESTUDIO_PORT</key><string>8780</string>' \
	  '    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>' \
	  '  </dict>' \
	  '  <key>WorkingDirectory</key><string>$(REPO)</string>' \
	  '  <key>RunAtLoad</key><true/>' \
	  '  <key>KeepAlive</key><true/>' \
	  '  <key>ProcessType</key><string>Interactive</string>' \
	  '  <key>LowPriorityIO</key><false/>' \
	  '  <key>StandardOutPath</key><string>$(REPO)/storage/logs/voicestudio.log</string>' \
	  '  <key>StandardErrorPath</key><string>$(REPO)/storage/logs/voicestudio.log</string>' \
	  '</dict></plist>' > "$(PLIST)"
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.voicestudio 2>/dev/null || true
	@# bootout is asynchronous; bootstrapping too soon fails with EBUSY.
	@sleep 2
	@launchctl bootstrap gui/$(UID) "$(PLIST)"
	@echo "waiting for the GPU server..."
	@for i in $$(seq 1 40); do \
		curl -sf --max-time 2 http://127.0.0.1:8780/health >/dev/null && \
			{ echo "GPU server ready on :8780 (starts automatically at login)"; exit 0; }; \
		sleep 1; done; \
		echo "no response on :8780; see storage/logs/voicestudio.log" >&2; exit 1

mac-teardown:
	@launchctl bootout gui/$(UID)/com.moneyprinterturbo.voicestudio 2>/dev/null || true
	@rm -f "$(PLIST)"
	@echo "GPU server removed; it will not start at login anymore"

mac-status:
	@launchctl print gui/$(UID)/com.moneyprinterturbo.voicestudio 2>/dev/null \
		| grep -E "state|pid" || echo "LaunchAgent not installed"
	@curl -sf --max-time 2 http://127.0.0.1:8780/health || echo "(server not answering)"
	@grep -i "loading OmniVoice" storage/logs/voicestudio.log 2>/dev/null | tail -1 || true
