FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy official uv binary from multi-stage build
COPY --from=uv /uv /uvx /bin/

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/root/.local/bin:$PATH"

RUN git clone https://github.com/CodeClash-ai/BattleCode.git /workspace \
    && cd /workspace \
    && git remote set-url origin https://github.com/CodeClash-ai/BattleCode.git
WORKDIR /workspace

# Create virtual environment
RUN uv venv --python python3.12 /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

# Install Cython first as it may be required for building dependencies
RUN uv pip install --python /workspace/.venv/bin/python Cython

RUN UV_NO_BUILD_ISOLATION=1 python run.py update
