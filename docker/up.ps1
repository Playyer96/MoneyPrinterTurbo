# Start the stack on Windows, picking the GPU override automatically.
#
#   powershell -ExecutionPolicy Bypass -File docker\up.ps1
#   powershell -ExecutionPolicy Bypass -File docker\up.ps1 down
#
# This is the Windows half of docker/detect-stack.sh -- `make` is not a thing
# on Windows, and compose cannot probe for a device driver on its own, so
# something has to decide whether to layer docker-compose.gpu.yml in. Two
# small implementations of one three-line rule beat making Windows users
# install a Unix shell.
#
# Prerequisites for the CUDA path: an NVIDIA driver installed on Windows (not
# inside WSL) and Docker Desktop on the WSL2 backend, which ships the
# Container Toolkit already.

param([string]$Action = "up")

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if ($env:COMPOSE_FILE) {
    # A pinned COMPOSE_FILE is the last word, same as the shell version.
    $files = $env:COMPOSE_FILE -split ":"
} else {
    # Ask the daemon whether the nvidia runtime is actually wired up. A card
    # with no Container Toolkit still cannot be reserved by compose, and
    # guessing "yes" turns a working CPU stack into one that will not start.
    $runtimes = & docker info --format "{{json .Runtimes}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "cannot talk to Docker -- is Docker Desktop running?"
    }
    if ($runtimes -match "nvidia") {
        Write-Host "NVIDIA runtime detected; using the CUDA stack"
        $files = @("docker-compose.yml", "docker-compose.gpu.yml")
    } else {
        Write-Host "no NVIDIA container runtime; using the CPU stack"
        $files = @("docker-compose.yml")
    }
}

$fileArgs = $files | ForEach-Object { @("-f", $_) }

switch ($Action) {
    "up"     { & docker compose @fileArgs up -d --build; & docker compose @fileArgs ps }
    "down"   { & docker compose @fileArgs down }
    "build"  { & docker compose @fileArgs build }
    "status" { & docker compose @fileArgs ps }
    default  { throw "unknown action '$Action' (expected: up, down, build, status)" }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
