#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-large-v3}"
MODEL_ROOT="${MODEL_ROOT:-/opt/stt-whisper-models}"
MODEL_LINK_PATH="${MODEL_LINK_PATH:-$MODEL_ROOT/$MODEL_ID}"
MARKER_FILE="${MARKER_FILE:-/etc/stt-whisper-image.env}"
VENV_ROOT="${VENV_ROOT:-/opt/stt-whisper-venv}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Preparing custom STT Whisper image"
export DEBIAN_FRONTEND=noninteractive

log "Installing system packages"
apt-get update -y
apt-get install -y curl ffmpeg python3-pip python3-venv tar

log "Creating Python virtual environment: $VENV_ROOT"
python3 -m venv "$VENV_ROOT"
"$VENV_ROOT/bin/python" -m pip install --upgrade pip

log "Installing Python packages into virtual environment"
"$VENV_ROOT/bin/python" -m pip install faster-whisper google-cloud-storage python-dotenv

log "Downloading faster-whisper model cache for $MODEL_ID"
mkdir -p "$MODEL_ROOT"
DOWNLOADED_PATH="$("$VENV_ROOT/bin/python" - "$MODEL_ID" "$MODEL_ROOT" <<'PY'
import sys
from faster_whisper.utils import download_model

model_id = sys.argv[1]
cache_dir = sys.argv[2]
model_path = download_model(model_id, cache_dir=cache_dir)
print(model_path)
PY
)"

if [[ -z "$DOWNLOADED_PATH" || ! -d "$DOWNLOADED_PATH" ]]; then
  log "Failed to resolve downloaded model path"
  exit 1
fi

if [[ "$DOWNLOADED_PATH" != "$MODEL_LINK_PATH" ]]; then
  rm -rf "$MODEL_LINK_PATH"
  ln -s "$DOWNLOADED_PATH" "$MODEL_LINK_PATH"
fi

log "Writing image marker: $MARKER_FILE"
cat > "$MARKER_FILE" <<EOF
STT_WHISPER_IMAGE_READY=1
STT_WHISPER_MODEL_ID=$MODEL_ID
STT_WHISPER_MODEL_PATH=$MODEL_LINK_PATH
STT_WHISPER_VENV_PATH=$VENV_ROOT
EOF

log "Custom image preparation complete"
log "Model path: $MODEL_LINK_PATH"
log "Virtualenv path: $VENV_ROOT"
