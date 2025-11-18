FROM ghcr.io/astral-sh/uv:latest AS uv

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    GO_VERSION=1.22.0 \
    PATH=/usr/local/go/bin:$PATH

# Install Python 3.10 (and alias python→python3.10) plus build prerequisites
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl ca-certificates python3.10 python3.10-venv \
    python3-pip python-is-python3 wget git build-essential jq curl locales \
 && rm -rf /var/lib/apt/lists/*

# Copy official uv binary from multi-stage build
COPY --from=uv /uv /uvx /bin/

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/root/.local/bin:$PATH"

# Set architecture and install Go 1.22
RUN ARCH=$(dpkg --print-architecture) && \
    echo "Building for architecture: $ARCH" && \
    curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz -o /tmp/go.tar.gz && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && \
    rm /tmp/go.tar.gz

# Inject GitHub token for private repo access
RUN git clone https://github.com/CodeClash-ai/BattleSnake.git /workspace \
    && cd /workspace \
    && git remote set-url origin https://github.com/CodeClash-ai/BattleSnake.git
WORKDIR /workspace

RUN uv venv --python python3.10 /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

RUN cd game && go build -o battlesnake ./cli/battlesnake/main.go
# Install dependencies (use cache mount for faster builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync --python /workspace/.venv/bin/python requirements.txt
