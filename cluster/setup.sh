#!/bin/bash
# Once per synced worktree on a FASRC login node, and after dependency changes: bash cluster/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra cluster
for m in configs/model/*.yaml; do uv run --no-sync --env-file .env hf download "$(sed -n 's/^path: //p' "$m")"; done
