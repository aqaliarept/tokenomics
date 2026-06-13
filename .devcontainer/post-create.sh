#!/usr/bin/env bash
set -euo pipefail

# Python venv
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# Sessions are stored in <project>/.claude/sessions/ so host and container
# share them. The host's slug symlink already exists in the bind-mounted
# ~/.claude/projects/ but it targets the host's absolute path (/Users/...)
# which doesn't exist here. Repoint the container's slug at the in-container
# project path. The host's slug is left alone.
SESSIONS_DIR="$PWD/.claude/sessions"
mkdir -p "$SESSIONS_DIR"
mkdir -p "$HOME/.claude/projects"
container_slug="$(printf '%s' "$PWD" | sed 's|/|-|g')"
ln -sfn -- "$SESSIONS_DIR" "$HOME/.claude/projects/$container_slug"
