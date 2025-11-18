FROM ghcr.io/astral-sh/uv:latest AS uv

FROM maven:3.9-eclipse-temurin-24

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.10+, pip, and common tools
RUN apt update && apt install -y \
    wget \
    git \
    build-essential \
    ant \
    unzip \
    python3 \
    python3-pip \
    python3-venv \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && rm -rf /var/lib/apt/lists/*

# Copy official uv binary from multi-stage build
COPY --from=uv /uv /uvx /bin/

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/root/.local/bin:$PATH"

RUN git clone https://github.com/CodeClash-ai/RoboCode.git /workspace \
    && cd /workspace \
    && git remote set-url origin https://github.com/CodeClash-ai/RoboCode.git

WORKDIR /workspace

# Create virtual environment
RUN uv venv --python python3 /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"
