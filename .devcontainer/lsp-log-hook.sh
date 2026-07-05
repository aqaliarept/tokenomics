#!/usr/bin/env python3
import sys, json, os
from datetime import datetime, timezone

try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)

if d.get("tool_name") != "LSP":
    sys.exit(0)

inp = d.get("tool_input", {})
op    = inp.get("operation", "?")
path  = inp.get("filePath", "?")
line  = inp.get("line", "?")
char  = inp.get("character", "?")
query = inp.get("query", "")
ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

entry = f"{ts}  {op:<25}  {os.path.basename(path)}:{line}:{char}"
if query:
    entry += f"  q={query}"

log = os.path.join(os.environ.get("PWD", "."), ".claude", "lsp-usage.log")
with open(log, "a") as f:
    f.write(entry + "\n")
