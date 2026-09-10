# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /MoneyPrinterTurbo

# The app writes into its own working directory (storage/, models/).
RUN chmod 777 /MoneyPrinterTurbo

ENV PYTHONPATH="/MoneyPrinterTurbo"

# Which package mirrors to prefer: "auto" measures it (see docker/pick-mirror.py),
# "china" and "default" force one side. The GHCR release workflow passes
# "default" explicitly so a GitHub runner never probes.
ARG DOCKER_BUILD_MIRROR=auto
# Legacy override kept because .github/workflows/docker-ghcr.yml sets it:
# 1 = official PyPI only, 0 = mirrors first. "auto" follows DOCKER_BUILD_MIRROR.
ARG PIP_USE_OFFICIAL=auto

COPY docker/pick-mirror.py /usr/local/bin/pick-mirror.py

# Resolving the mirror and installing system packages happen in one layer so
# the choice can be written to /etc/mpt-mirror and reused by the pip step.
#
# Both mirror sets are tried in preference order rather than only the preferred
# one: a fallback costs nothing when the first choice works, and it means a
# blocked or down mirror degrades to a slow build instead of a failed one.
# All sources are HTTPS -- some networks drop plaintext HTTP outright.
#
# Every path must end in either a working apt or a non-zero exit. The original
# loop ended in `sleep`, which always returns 0, so a build with no git and no
# ffmpeg still produced an image that failed later at runtime.
RUN set -u; \
    if [ "$DOCKER_BUILD_MIRROR" = "auto" ]; then \
        DOCKER_BUILD_MIRROR="$(python3 /usr/local/bin/pick-mirror.py)"; \
    fi; \
    echo "$DOCKER_BUILD_MIRROR" > /etc/mpt-mirror; \
    echo "using $DOCKER_BUILD_MIRROR package mirrors"; \
    ALIYUN="https://mirrors.aliyun.com/debian https://mirrors.aliyun.com/debian-security"; \
    TUNA="https://mirrors.tuna.tsinghua.edu.cn/debian https://mirrors.tuna.tsinghua.edu.cn/debian-security"; \
    DEBIAN="https://deb.debian.org/debian https://deb.debian.org/debian-security"; \
    if [ "$DOCKER_BUILD_MIRROR" = "china" ]; then \
        SOURCES="$ALIYUN|$TUNA|$DEBIAN"; \
    else \
        SOURCES="$DEBIAN|$ALIYUN|$TUNA"; \
    fi; \
    installed=0; \
    IFS='|'; for pair in $SOURCES; do \
        unset IFS; \
        set -- $pair; \
        printf 'deb %s bookworm main\ndeb %s bookworm-updates main\ndeb %s bookworm-security main\n' \
            "$1" "$1" "$2" > /etc/apt/sources.list; \
        rm -rf /var/lib/apt/lists/*; \
        echo "trying $1"; \
        # Check-Valid-Until: bookworm-security release files can outlive their \
        # Valid-Until after a point release; allow stale InRelease so apt does \
        # not refuse the whole run over a clock detail. \
        # \
        # The timeouts are what make the fallback list usable. apt defaults to \
        # ~120s per connection with retries on top, so one unreachable mirror \
        # cost about three minutes before the next was tried -- long enough to \
        # look like a hung build rather than a failover. ForceIPv4 is in here \
        # for the same reason: a host with broken IPv6 egress otherwise waits \
        # out the full timeout on every AAAA record before falling back to A. \
        if apt-get -o Acquire::Check-Valid-Until=false \
                   -o Acquire::ForceIPv4=true \
                   -o Acquire::http::Timeout=15 \
                   -o Acquire::https::Timeout=15 \
                   -o Acquire::Retries=1 update \
           && apt-get -o Acquire::ForceIPv4=true \
                      -o Acquire::http::Timeout=15 \
                      -o Acquire::https::Timeout=15 \
                      -o Acquire::Retries=1 \
                      install -y --no-install-recommends git ffmpeg; then \
            installed=1; break; \
        fi; \
        echo "mirror $1 failed, trying the next one" >&2; \
        IFS='|'; \
    done; \
    unset IFS; \
    if [ "$installed" != "1" ]; then \
        echo "no configured Debian mirror could install git and ffmpeg" >&2; \
        exit 1; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt ./

# Same ordering rule as apt: preferred index first, the other as fallback.
RUN set -u; \
    if [ "$PIP_USE_OFFICIAL" = "1" ]; then \
        MIRROR=default; \
    elif [ "$PIP_USE_OFFICIAL" = "0" ]; then \
        MIRROR=china; \
    else \
        MIRROR="$(cat /etc/mpt-mirror)"; \
    fi; \
    ALIYUN="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"; \
    TUNA="-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ --trusted-host mirrors.tuna.tsinghua.edu.cn"; \
    if [ "$MIRROR" = "china" ]; then \
        set -- "$ALIYUN" "$TUNA" ""; \
    else \
        set -- "" "$ALIYUN" "$TUNA"; \
    fi; \
    for index in "$@"; do \
        echo "pip install via ${index:-official PyPI}"; \
        # shellcheck disable=SC2086 -- $index must word-split into flags. \
        if pip install --no-cache-dir $index --retries 3 --timeout 60 -r requirements.txt; then \
            exit 0; \
        fi; \
        echo "index ${index:-official PyPI} failed, trying the next one" >&2; \
    done; \
    echo "no configured PyPI index could install the requirements" >&2; \
    exit 1

# Now copy the rest of the codebase into the image
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Inside the container the server must listen on 0.0.0.0; the host side is
# still limited to 127.0.0.1 by the docker port mapping. browser.serverAddress
# only sets the URL shown to the user and is not a substitute for
# server.address.
CMD ["streamlit", "run", "./webui/Main.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=127.0.0.1", "--server.enableCORS=True", "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal", "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"]

# 1. Build the Docker image using the following command
# docker build -t moneyprinterturbo .

# 2. Run the Docker container using the following command
## For Linux or MacOS:
# docker run -v $(pwd)/config.toml:/MoneyPrinterTurbo/config.toml -v $(pwd)/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
## For Windows (PowerShell):
# docker run -v ${PWD}/config.toml:/MoneyPrinterTurbo/config.toml -v ${PWD}/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
