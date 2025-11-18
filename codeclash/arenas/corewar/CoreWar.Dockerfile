FROM ghcr.io/astral-sh/uv:latest AS uv

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

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

RUN git clone https://github.com/CodeClash-ai/CoreWar.git /workspace \
    && cd /workspace \
    && git remote set-url origin https://github.com/CodeClash-ai/CoreWar.git
WORKDIR /workspace

# Create virtual environment
RUN uv venv --python python3.10 /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

RUN cd src/ && make CFLAGS="-O -DEXT94 -DPERMUTATE -DRWLIMIT" LIB=""

# Copy dwarf example to home directory for validation purposes
RUN cp /workspace/doc/examples/dwarf.red /home/dwarf.red
