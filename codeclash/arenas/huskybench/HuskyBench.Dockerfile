FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
curl \
wget \
git \
build-essential \
unzip \
lsof \
&& rm -rf /var/lib/apt/lists/*

# Copy official uv binary from multi-stage build
COPY --from=uv /uv /uvx /bin/

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/root/.local/bin:$PATH"

RUN git clone https://github.com/CodeClash-ai/HuskyBench.git /workspace \
    && cd /workspace \
    && git remote set-url origin https://github.com/CodeClash-ai/HuskyBench.git
WORKDIR /workspace

RUN uv venv --python python3.10 /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

# Install Cython first as it's required for building eval7 (use cache mount for faster builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /workspace/.venv/bin/python Cython
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync --python /workspace/.venv/bin/python --no-build-isolation engine/requirements.txt
RUN mkdir -p /workspace/engine/output
