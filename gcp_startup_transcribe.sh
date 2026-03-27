#!/bin/bash
set -euo pipefail

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
TOKEN_URL="http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
METADATA_HEADER="Metadata-Flavor: Google"
WORK_ROOT="${WORK_ROOT:-/opt/stt-whisper-job}"
APP_ROOT="$WORK_ROOT/app"
RUNTIME_ROOT="$WORK_ROOT/runtime"
LOG_FILE="/var/log/stt-whisper.log"
DEFAULT_VENV_ROOT="/opt/stt-whisper-venv"
WORK_VENV_ROOT="$WORK_ROOT/venv"

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_metadata() {
  curl -fsS -H "$METADATA_HEADER" "$METADATA_URL/$1"
}

get_metadata_optional() {
  curl -fsS -H "$METADATA_HEADER" "$METADATA_URL/$1" 2>/dev/null || true
}

fetch_secret_payload() {
  local secret_resource="$1"
  local access_token
  local secret_response

  access_token="$(curl -fsS -H "$METADATA_HEADER" "$TOKEN_URL" | "$APP_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')"
  secret_response="$(curl -fsS -H "Authorization: Bearer $access_token" \
    "https://secretmanager.googleapis.com/v1/${secret_resource}:access")"
  printf '%s' "$secret_response" | "$APP_PYTHON" -c \
    'import base64, json, sys; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode("utf-8").strip())'
}

python_module_available() {
  "$APP_PYTHON" -c "import $1" >/dev/null 2>&1
}

log "Startup script begin"
mkdir -p "$WORK_ROOT" "$APP_ROOT" "$RUNTIME_ROOT"

APP_PYTHON="python3"
if [[ -x "$DEFAULT_VENV_ROOT/bin/python" ]]; then
  APP_PYTHON="$DEFAULT_VENV_ROOT/bin/python"
  log "Using prebuilt virtualenv: $DEFAULT_VENV_ROOT"
fi

log "Fetching instance metadata"
BUNDLE_URI="$(get_metadata BUNDLE_URI)"
INPUT_URI="$(get_metadata INPUT_URI)"
OUTPUT_PREFIX="$(get_metadata OUTPUT_PREFIX)"
JOB_ID="$(get_metadata JOB_ID)"
TRANSCRIBE_PRESET="$(get_metadata TRANSCRIBE_PRESET)"
SKIP_CORRECTION="$(get_metadata SKIP_CORRECTION)"
KEEP_PREPROCESSED="$(get_metadata KEEP_PREPROCESSED)"
WHISPER_DEVICE="$(get_metadata WHISPER_DEVICE)"
WHISPER_MODEL_PATH="$(get_metadata_optional WHISPER_MODEL_PATH)"
OPENAI_API_KEY_SECRET="$(get_metadata_optional OPENAI_API_KEY_SECRET)"
log "Metadata loaded: JOB_ID=$JOB_ID PRESET=$TRANSCRIBE_PRESET SKIP_CORRECTION=$SKIP_CORRECTION DEVICE=$WHISPER_DEVICE"
log "Input URI: $INPUT_URI"
log "Output prefix: $OUTPUT_PREFIX"
if [[ -n "$WHISPER_MODEL_PATH" ]]; then
  log "Whisper model path from metadata: $WHISPER_MODEL_PATH"
fi
if [[ -n "$OPENAI_API_KEY_SECRET" ]]; then
  log "OPENAI_API_KEY secret metadata configured"
fi

export DEBIAN_FRONTEND=noninteractive
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1; then
  log "Installing missing base system packages"
  apt-get update -y
  apt-get install -y curl ffmpeg python3-pip python3-venv tar
else
  log "Base system packages already available, skipping apt install"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi not found, starting NVIDIA driver installation"
  mkdir -p /opt/google/cuda-installer
  cd /opt/google/cuda-installer || exit 1

  if [[ ! -f cuda_installer.pyz ]]; then
    log "Downloading CUDA installer helper"
    curl -fSsL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
      --output cuda_installer.pyz
  fi

  log "Installing NVIDIA driver"
  python3 cuda_installer.pyz install_driver --installation-mode=binary --installation-branch=prod

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver installation started. The VM may reboot before the transcription step runs."
    exit 0
  fi
fi

log "GPU status"
nvidia-smi || true

if python_module_available faster_whisper && python_module_available google.cloud.storage && python_module_available dotenv && python_module_available openai; then
  log "Python dependencies already available, skipping pip install"
else
  log "Preparing job virtualenv: $WORK_VENV_ROOT"
  python3 -m venv "$WORK_VENV_ROOT"
  APP_PYTHON="$WORK_VENV_ROOT/bin/python"
  "$APP_PYTHON" -m pip install --upgrade pip
  log "Installing Python dependencies for the VM job runner"
  "$APP_PYTHON" -m pip install faster-whisper google-cloud-storage python-dotenv openai
fi

if [[ -z "${OPENAI_API_KEY:-}" && -n "$OPENAI_API_KEY_SECRET" ]]; then
  log "Fetching OPENAI_API_KEY from Secret Manager"
  OPENAI_API_KEY="$(fetch_secret_payload "$OPENAI_API_KEY_SECRET")"
  export OPENAI_API_KEY
fi

if [[ "$SKIP_CORRECTION" != "true" && -z "${OPENAI_API_KEY:-}" ]]; then
  log "SKIP_CORRECTION=false but OPENAI_API_KEY is not available. Configure OPENAI_API_KEY_SECRET metadata or bake the key into the VM environment."
  exit 1
fi

log "Downloading source bundle from GCS"
"$APP_PYTHON" - "$BUNDLE_URI" "$APP_ROOT/bundle.tar.gz" <<'PY'
from pathlib import Path
import sys

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    bucket_name, _, object_name = uri[5:].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Expected full gs:// URI, got: {uri}")
    return bucket_name, object_name


uri = sys.argv[1]
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
bucket_name, object_name = parse_gs_uri(uri)
client = storage.Client()
blob = client.bucket(bucket_name).blob(object_name)
blob.download_to_filename(str(destination))
PY

log "Extracting source bundle"
tar -xzf "$APP_ROOT/bundle.tar.gz" -C "$APP_ROOT"

KEEP_PREPROCESSED_FLAG="--no-keep-preprocessed"
if [[ "$KEEP_PREPROCESSED" == "true" ]]; then
  KEEP_PREPROCESSED_FLAG="--keep-preprocessed"
fi

SKIP_CORRECTION_FLAG="--no-skip-correction"
if [[ "$SKIP_CORRECTION" == "true" ]]; then
  SKIP_CORRECTION_FLAG="--skip-correction"
fi

log "Launching Python job runner"
RUNNER_ARGS=(
  "$APP_PYTHON" "$APP_ROOT/src/gcp_job_runner.py"
  --input-uri "$INPUT_URI"
  --output-prefix "$OUTPUT_PREFIX"
  --job-id "$JOB_ID"
  --work-root "$RUNTIME_ROOT"
  --preset "$TRANSCRIBE_PRESET"
  --device "$WHISPER_DEVICE"
  "$SKIP_CORRECTION_FLAG"
  "$KEEP_PREPROCESSED_FLAG"
)

if [[ -n "$WHISPER_MODEL_PATH" ]]; then
  if [[ -d "$WHISPER_MODEL_PATH" ]]; then
    log "Using baked local Whisper model directory: $WHISPER_MODEL_PATH"
    RUNNER_ARGS+=(--model-path "$WHISPER_MODEL_PATH")
  else
    log "Configured local Whisper model path does not exist, falling back to Hub model id: $WHISPER_MODEL_PATH"
  fi
fi

"${RUNNER_ARGS[@]}"

log "Job runner finished, shutting down VM"
shutdown -h now
