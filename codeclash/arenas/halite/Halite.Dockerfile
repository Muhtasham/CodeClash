FROM ghcr.io/astral-sh/uv:latest AS uv

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 (and alias python→python3.10), pip, and prerequisites
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

# Install Rust via rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . "$HOME/.cargo/env" \
    && echo 'source $HOME/.cargo/env' >> /etc/bash.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

# Install ocaml
RUN apt-get update && apt-get install -y ocaml ocamlbuild

# Clone Halite repository
RUN git clone https://github.com/CodeClash-ai/Halite.git /workspace \
    && cd /workspace \
    && git remote set-url origin https://github.com/CodeClash-ai/Halite.git
WORKDIR /workspace

# Create virtual environment
RUN uv venv --python python3.10 /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

RUN cd environment && make
