#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./submit_gcp_job.sh <input-audio-path> [extra gcp_submit_job.py args...]" >&2
  exit 1
fi

INPUT_FILE="$1"
shift

cd "$SCRIPT_DIR"
exec uv run python gcp_submit_job.py "$INPUT_FILE" "$@"
