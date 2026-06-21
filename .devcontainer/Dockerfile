FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt source whose GPG key is no longer valid,
# which breaks every subsequent `apt-get update`. Drop it before installing.
RUN rm -f /etc/apt/sources.list.d/yarn.list

# System deps:
#   git-lfs        - experiments/**/artifacts/** are LFS-tracked (see .gitattributes);
#                    checkpoints/logs live there. `git lfs install --system` enables it
#                    for every user (writes /etc/gitconfig).
#   rsync, openssh - run_pod_training.py syncs code up to / pulls results down from the
#                    Runpod pod over ssh+rsync.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs rsync openssh-client \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS + Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# runpodctl - drives off-box GPU training (run_pod_training.py). The install script
# drops the binary in /usr/local/bin (on PATH for all users). Auth is per-user at
# runtime: `runpodctl config --apiKey <key>` (not baked into the image).
RUN curl -fsSL https://cli.runpod.net | bash
