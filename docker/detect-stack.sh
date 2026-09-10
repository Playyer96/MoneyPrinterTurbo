#!/bin/sh
# Print the COMPOSE_FILE chain this host should run, e.g.
#     docker-compose.yml:docker-compose.gpu.yml
#
# Compose cannot do this itself. Asking for a device driver the host lacks is
# a hard startup failure, not a soft one, so the GPU reservations cannot live
# in the base file and be ignored when absent -- they have to be layered in by
# a caller that already knows what the host has. That caller is this script.
#
# Usage:
#   COMPOSE_FILE=$(sh docker/detect-stack.sh) docker compose up -d
#   make up          # does the same thing for you
#
# An existing COMPOSE_FILE always wins, so pinning a stack by hand (or via
# .env) is still the last word.
set -u

if [ -n "${COMPOSE_FILE:-}" ]; then
    echo "$COMPOSE_FILE"
    exit 0
fi

base="docker-compose.yml"

# Apple Silicon first, and without consulting Docker at all: macOS does not
# pass Metal through to Docker's Linux VM, so a Mac never has a container-
# visible GPU no matter what the daemon reports. The mac override routes to
# the host-side model server instead.
if [ "$(uname -s 2>/dev/null || echo unknown)" = "Darwin" ]; then
    echo "$base:docker-compose.mac.yml"
    exit 0
fi

# NVIDIA: ask the daemon whether it actually has the nvidia runtime wired up,
# rather than whether the host has a card. A GPU with no Container Toolkit
# still cannot be reserved by compose, and guessing "yes" there turns a
# working CPU stack into one that refuses to start.
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then
    echo "$base:docker-compose.gpu.yml"
    exit 0
fi

# AMD: ROCm exposes the GPU as device nodes rather than a docker runtime, so
# there is nothing in `docker info` to look for. /dev/kfd is the kernel
# driver's control node and is present only when amdgpu is loaded.
if [ -e /dev/kfd ]; then
    echo "$base:docker-compose.rocm.yml"
    exit 0
fi

echo "$base"
