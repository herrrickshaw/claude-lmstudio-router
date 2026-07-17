#!/usr/bin/env bash
# Launch Claude Code wired to your local LM Studio model via the proxy.
#
# Prereqs (see README):
#   1. LM Studio running with a tool-capable model + server started (port 1234).
#   2. The proxy running:  python claude_code_proxy.py   (in another terminal)
#
# Then just run:  ./run_local_claude_code.sh
# Anything after the script name is passed straight to `claude`.

set -euo pipefail

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8787}"
BASE_URL="http://${PROXY_HOST}:${PROXY_PORT}"

# Quick reachability check so you get a clear message instead of a cryptic
# Claude Code error if the proxy/LM Studio isn't up.
if ! curl -s -m 3 "${BASE_URL}/health" | grep -q '"lmstudio": *"ok"'; then
  echo "WARNING: proxy or LM Studio not reachable at ${BASE_URL}." >&2
  echo "  - Is LM Studio running with its server started?" >&2
  echo "  - Did you start the proxy?  python claude_code_proxy.py" >&2
  echo "Continuing anyway..." >&2
fi

export ANTHROPIC_BASE_URL="${BASE_URL}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"   # any non-empty value; proxy ignores it

# These two stop Claude Code from doing extra Anthropic-only housekeeping
# calls that a local model can't serve.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}"

echo "Claude Code -> ${BASE_URL} (local LM Studio)"
exec claude "$@"
