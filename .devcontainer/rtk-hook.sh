#!/usr/bin/env bash
# PreToolUse hook: strip venv python prefix before rtk rewrites the command.
# e.g. "../.venv/bin/python -m pytest -q" -> "pytest -q" -> "rtk pytest -q"
input=$(cat)
modified=$(python3 - "$input" <<'EOF'
import sys, json

raw = sys.argv[1]
try:
    d = json.loads(raw)
except Exception:
    print(raw)
    sys.exit(0)

cmd = d.get("tool_input", {}).get("command", "")
for prefix in ("../.venv/bin/python -m ", ".venv/bin/python -m "):
    if cmd.startswith(prefix):
        d["tool_input"]["command"] = cmd[len(prefix):]
        break

print(json.dumps(d))
EOF
)
echo "$modified" | rtk hook claude
