FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt source whose GPG key is no longer valid,
# which breaks every subsequent `apt-get update`. Drop it before installing.
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Node.js LTS + Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code
