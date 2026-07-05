#!/usr/bin/env bash
set -euo pipefail

# Python venv + package (editable) with train (torch) and dev (pytest) extras.
# Venv lives at the repo root (.venv); the package is the tokenomics/ subdir, so
# CLAUDE.md commands run as `../.venv/bin/python ...` from inside tokenomics/.
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "./tokenomics[train,dev]"

# The workspace is a bind mount owned by the host user, which trips git's
# "dubious ownership" guard (breaks git-lfs filters, status, etc.). Trust it.
git config --global --add safe.directory "$PWD"

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

# Keep hook scripts in the devcontainer dir (version-controlled);
# symlink them into ~/.claude/ where settings.json expects them.
ln -sfn -- "$PWD/.devcontainer/rtk-hook.sh" "$HOME/.claude/rtk-hook.sh"
ln -sfn -- "$PWD/.devcontainer/lsp-log-hook.sh" "$HOME/.claude/lsp-log-hook.sh"

# Claude Code plugin paths (known_marketplaces.json, installed_plugins.json) are
# written by the host process using the host's home path (e.g. /Users/aqaliarept).
# The bind mount lands at $HOME/.claude, so those paths don't resolve here.
# Extract the host's .claude path from the JSON and symlink it to $HOME/.claude.
host_claude_dir=$(python3 - <<'EOF'
import json, re, os, sys
try:
    with open(os.path.expanduser("~/.claude/plugins/known_marketplaces.json")) as f:
        data = json.load(f)
    for v in data.values():
        loc = v.get("installLocation", "")
        m = re.match(r"(.+/\.claude)/", loc)
        if m and m.group(1) != os.path.expanduser("~/.claude"):
            print(m.group(1))
            sys.exit(0)
except Exception:
    pass
EOF
)
if [[ -n "$host_claude_dir" ]]; then
    sudo mkdir -p "$(dirname "$host_claude_dir")"
    sudo ln -sfn "$HOME/.claude" "$host_claude_dir"
fi
