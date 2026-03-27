from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from googleapiclient import discovery
from googleapiclient.errors import HttpError


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_MARKERS = ("_SUCCESS.json", "_FAILED.json")
DEFAULT_BASE_IMAGE_BOOT_DISK_SIZE_GB = 60
DEFAULT_CUSTOM_IMAGE_BOOT_DISK_SIZE_GB = 80

load_dotenv(PROJECT_ROOT / ".env")


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name, fallback)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def env_flag(name: str, fallback: bool) -> bool:
    value = env_default(name)
    if value is None:
        return fallback
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one job to GCS and create a Spot T4 VM through the Compute Engine API."
    )
    parser.add_argument("input_file", help="Local input audio file.")
    parser.add_argument("--project-id", default=env_default("GCP_PROJECT_ID"), help="GCP project id.")
    parser.add_argument("--zone", default=env_default("GCP_ZONE"), help="Compute Engine zone, for example asia-northeast3-b.")
    parser.add_argument("--service-account", default=env_default("GCP_SERVICE_ACCOUNT"), help="Service account email for the VM.")
    parser.add_argument("--input-bucket", default=env_default("GCP_INPUT_BUCKET"), help="Bucket that receives uploaded input files.")
    parser.add_argument("--output-bucket", default=env_default("GCP_OUTPUT_BUCKET"), help="Bucket that receives generated outputs.")
    parser.add_argument(
        "--staging-bucket",
        default=env_default("GCP_STAGING_BUCKET"),
        help="Bucket used for the source bundle. Defaults to the input bucket.",
    )
    parser.add_argument("--vm-name-prefix", default=env_default("GCP_VM_NAME_PREFIX", "stt-job"), help="Prefix for the transient VM name.")
    parser.add_argument("--machine-type", default=env_default("GCP_MACHINE_TYPE", "n1-standard-4"), help="Compute Engine machine type.")
    parser.add_argument("--accelerator-type", default=env_default("GCP_ACCELERATOR_TYPE", "nvidia-tesla-t4"), help="GPU accelerator type.")
    parser.add_argument("--accelerator-count", type=int, default=int(env_default("GCP_ACCELERATOR_COUNT", "1") or "1"), help="GPU count.")
    parser.add_argument(
        "--image-family",
        default=env_default("GCP_IMAGE_FAMILY", "common-cu128-ubuntu-2204-nvidia-570"),
        help="Boot image family.",
    )
    parser.add_argument(
        "--image-project",
        default=env_default("GCP_IMAGE_PROJECT", "deeplearning-platform-release"),
        help="Boot image project.",
    )
    parser.add_argument(
        "--boot-disk-size-gb",
        type=int,
        default=int(env_default("GCP_BOOT_DISK_SIZE_GB")) if env_default("GCP_BOOT_DISK_SIZE_GB") else None,
        help="Boot disk size in GiB. Defaults to 60 for the base DLVM image and 80 for the stt-whisper custom image family.",
    )
    parser.add_argument("--boot-disk-type", default=env_default("GCP_BOOT_DISK_TYPE", "pd-balanced"), help="Boot disk type.")
    parser.add_argument("--preset", default=env_default("GCP_TRANSCRIBE_PRESET", "high_quality"), help="Transcription preset passed to the VM.")
    parser.add_argument(
        "--skip-correction",
        action=argparse.BooleanOptionalAction,
        default=env_flag("GCP_SKIP_CORRECTION", True),
        help="Skip the LLM correction step on the VM. Disable only if the VM already has OPENAI_API_KEY configured.",
    )
    parser.add_argument("--whisper-device", default=env_default("GCP_WHISPER_DEVICE", "auto"), help="Whisper runtime device on the VM.")
    parser.add_argument(
        "--whisper-model-path",
        default=env_default("GCP_WHISPER_MODEL_PATH"),
        help="Local CTranslate2 Whisper model directory baked into the VM image.",
    )
    parser.add_argument(
        "--openai-api-key-secret",
        default=env_default("GCP_OPENAI_API_KEY_SECRET"),
        help=(
            "Secret Manager resource for OPENAI_API_KEY, for example "
            "projects/<project>/secrets/openai-api-key/versions/latest."
        ),
    )
    parser.add_argument(
        "--max-run-duration-seconds",
        type=int,
        default=int(env_default("GCP_MAX_RUN_DURATION_SECONDS", "1800") or "1800"),
        help="Max VM runtime before automatic deletion.",
    )
    parser.add_argument(
        "--keep-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=env_flag("GCP_KEEP_PREPROCESSED", False),
        help="Upload the prepared wav file to the output prefix.",
    )
    parser.add_argument(
        "--startup-script",
        default=env_default("GCP_STARTUP_SCRIPT", str(PROJECT_ROOT / "gcp_startup_transcribe.sh")),
        help="Path to the Linux startup script template.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=env_flag("GCP_WAIT", True),
        help="Wait for the output marker and optionally download the result files.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=int(env_default("GCP_POLL_INTERVAL_SECONDS", "20") or "20"),
        help="Polling interval when --wait is enabled.",
    )
    parser.add_argument(
        "--download-dir",
        default=env_default("GCP_DOWNLOAD_DIR", "data"),
        help="Local directory where outputs should be downloaded when --wait is enabled.",
    )
    parser.add_argument(
        "--clear-job-objects",
        action=argparse.BooleanOptionalAction,
        default=env_flag("GCP_CLEAR_JOB_OBJECTS", True),
        help="Delete gs://.../jobs/<job_id>/ objects from input, staging, and output buckets after a waited job completes.",
    )
    args = parser.parse_args()

    missing_required: list[str] = []
    required_fields = {
        "project_id": "--project-id or GCP_PROJECT_ID",
        "zone": "--zone or GCP_ZONE",
        "service_account": "--service-account or GCP_SERVICE_ACCOUNT",
        "input_bucket": "--input-bucket or GCP_INPUT_BUCKET",
        "output_bucket": "--output-bucket or GCP_OUTPUT_BUCKET",
    }
    for field_name, label in required_fields.items():
        if not getattr(args, field_name):
            missing_required.append(label)

    if missing_required:
        parser.error("Missing required configuration: " + ", ".join(missing_required))

    return args


def sanitize_label(value: str, max_length: int = 63) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
    return lowered[:max_length] or "job"


def make_gs_uri(bucket: str, object_name: str) -> str:
    return f"gs://{bucket}/{object_name.lstrip('/')}"


def resolve_boot_disk_size_gb(args: argparse.Namespace) -> int:
    if args.boot_disk_size_gb is not None:
        return args.boot_disk_size_gb

    if args.image_project == args.project_id and args.image_family.startswith("stt-whisper"):
        return DEFAULT_CUSTOM_IMAGE_BOOT_DISK_SIZE_GB

    return DEFAULT_BASE_IMAGE_BOOT_DISK_SIZE_GB


def upload_file(client: storage.Client, local_path: Path, bucket_name: str, object_name: str) -> str:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    return make_gs_uri(bucket_name, object_name)


def create_source_bundle(bundle_path: Path) -> None:
    bundle_members = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "prompts",
    ]
    with tarfile.open(bundle_path, "w:gz") as archive:
        for member in bundle_members:
            archive.add(member, arcname=member.relative_to(PROJECT_ROOT).as_posix())


def wait_for_zone_operation(compute, project_id: str, zone: str, operation_name: str) -> dict:
    while True:
        operation = compute.zoneOperations().get(
            project=project_id,
            zone=zone,
            operation=operation_name,
        ).execute()
        if operation.get("status") == "DONE":
            if "error" in operation:
                raise RuntimeError(json.dumps(operation["error"], ensure_ascii=False))
            return operation
        time.sleep(3)


def find_result_marker(client: storage.Client, bucket_name: str, prefix: str):
    clean_prefix = prefix.rstrip("/")
    bucket = client.bucket(bucket_name)
    for marker_name in RESULT_MARKERS:
        object_name = f"{clean_prefix}/{marker_name}"
        blob = bucket.blob(object_name)
        if blob.exists(client):
            return marker_name, blob
    return None


def download_output_prefix(client: storage.Client, bucket_name: str, prefix: str, destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    normalized_prefix = prefix.rstrip("/") + "/"
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=normalized_prefix):
        name = blob.name.rsplit("/", 1)[-1]
        if not name or name.startswith("_"):
            continue
        destination = destination_dir / name
        blob.download_to_filename(str(destination))
        downloaded.append(destination)
    return downloaded


def delete_prefix(client: storage.Client, bucket_name: str, prefix: str) -> int:
    normalized_prefix = prefix.rstrip("/") + "/"
    bucket = client.bucket(bucket_name)
    deleted_count = 0
    for blob in client.list_blobs(bucket, prefix=normalized_prefix):
        blob.delete()
        deleted_count += 1
    return deleted_count


def delete_instance(compute, project_id: str, zone: str, instance_name: str) -> None:
    try:
        operation = compute.instances().delete(
            project=project_id,
            zone=zone,
            instance=instance_name,
        ).execute()
    except HttpError as exc:
        status_code = getattr(getattr(exc, "resp", None), "status", None)
        if status_code == 404:
            return
        raise
    wait_for_zone_operation(compute, project_id, zone, operation["name"])


def build_instance_body(
    args: argparse.Namespace,
    vm_name: str,
    startup_script: str,
    metadata_items: list[dict[str, str]],
) -> dict:
    resolved_boot_disk_size_gb = resolve_boot_disk_size_gb(args)
    return {
        "name": vm_name,
        "machineType": f"zones/{args.zone}/machineTypes/{args.machine_type}",
        "labels": {
            "app": "stt-whisper",
            "job": sanitize_label(vm_name),
        },
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "diskSizeGb": str(resolved_boot_disk_size_gb),
                    "diskType": f"zones/{args.zone}/diskTypes/{args.boot_disk_type}",
                    "sourceImage": f"projects/{args.image_project}/global/images/family/{args.image_family}",
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": "global/networks/default",
                "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}],
            }
        ],
        "guestAccelerators": [
            {
                "acceleratorCount": args.accelerator_count,
                "acceleratorType": f"zones/{args.zone}/acceleratorTypes/{args.accelerator_type}",
            }
        ],
        "serviceAccounts": [
            {
                "email": args.service_account,
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
        ],
        "metadata": {
            "items": [{"key": "startup-script", "value": startup_script}, *metadata_items],
        },
        "scheduling": {
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": str(args.max_run_duration_seconds)},
        },
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    startup_script_path = Path(args.startup_script).resolve()
    if not startup_script_path.exists():
        raise FileNotFoundError(f"Startup script not found: {startup_script_path}")

    storage_client = storage.Client(project=args.project_id)
    compute = discovery.build("compute", "v1", cache_discovery=False)

    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    input_stem = sanitize_label(input_path.stem, max_length=32)
    job_id = f"{input_stem}-{timestamp}"
    vm_name = sanitize_label(f"{args.vm_name_prefix}-{job_id}")
    input_object = f"jobs/{job_id}/input/{input_path.name}"
    output_prefix = f"jobs/{job_id}"
    bundle_object = f"jobs/{job_id}/bundle/stt-whisper-bundle.tar.gz"
    staging_bucket = args.staging_bucket or args.input_bucket

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        bundle_path = temp_dir / "stt-whisper-bundle.tar.gz"
        create_source_bundle(bundle_path)

        input_uri = upload_file(storage_client, input_path, args.input_bucket, input_object)
        bundle_uri = upload_file(storage_client, bundle_path, staging_bucket, bundle_object)

    startup_script = startup_script_path.read_text(encoding="utf-8")
    metadata_items = [
        {"key": "BUNDLE_URI", "value": bundle_uri},
        {"key": "INPUT_URI", "value": input_uri},
        {"key": "OUTPUT_PREFIX", "value": make_gs_uri(args.output_bucket, output_prefix)},
        {"key": "JOB_ID", "value": job_id},
        {"key": "TRANSCRIBE_PRESET", "value": args.preset},
        {"key": "SKIP_CORRECTION", "value": "true" if args.skip_correction else "false"},
        {"key": "KEEP_PREPROCESSED", "value": "true" if args.keep_preprocessed else "false"},
        {"key": "WHISPER_DEVICE", "value": args.whisper_device},
    ]
    if args.whisper_model_path:
        metadata_items.append({"key": "WHISPER_MODEL_PATH", "value": args.whisper_model_path})
    if args.openai_api_key_secret:
        metadata_items.append({"key": "OPENAI_API_KEY_SECRET", "value": args.openai_api_key_secret})
    body = build_instance_body(args, vm_name, startup_script, metadata_items)

    insert_operation = compute.instances().insert(
        project=args.project_id,
        zone=args.zone,
        body=body,
    ).execute()
    wait_for_zone_operation(compute, args.project_id, args.zone, insert_operation["name"])

    payload = {
        "job_id": job_id,
        "vm_name": vm_name,
        "input_uri": input_uri,
        "bundle_uri": bundle_uri,
        "output_prefix": make_gs_uri(args.output_bucket, output_prefix),
        "started_at": started_at.isoformat(),
        "skip_correction": args.skip_correction,
        "whisper_model_path": args.whisper_model_path,
        "openai_api_key_secret_configured": bool(args.openai_api_key_secret),
        "boot_disk_size_gb": resolve_boot_disk_size_gb(args),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not args.wait:
        return 0

    marker = None
    exit_code = 1
    try:
        while marker is None:
            marker = find_result_marker(storage_client, args.output_bucket, output_prefix)
            if marker is None:
                time.sleep(args.poll_interval_seconds)

        marker_name, marker_blob = marker
        print(f"Detected result marker: gs://{args.output_bucket}/{marker_blob.name}")
        if args.download_dir:
            downloaded = download_output_prefix(
                storage_client,
                args.output_bucket,
                output_prefix,
                Path(args.download_dir).resolve(),
            )
            print("Downloaded files:")
            for path in downloaded:
                print(f"- {path}")
        exit_code = 0 if marker_name == "_SUCCESS.json" else 1
        return exit_code
    finally:
        if args.wait and args.clear_job_objects:
            cleanup_targets: list[tuple[str, str]] = [
                (args.input_bucket, output_prefix),
                (staging_bucket, output_prefix),
                (args.output_bucket, output_prefix),
            ]
            seen_targets: set[tuple[str, str]] = set()
            print("Cleaning up GCS job objects:")
            for bucket_name, prefix in cleanup_targets:
                target = (bucket_name, prefix)
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                deleted_count = delete_prefix(storage_client, bucket_name, prefix)
                print(f"- gs://{bucket_name}/{prefix}/ ({deleted_count} objects deleted)")
        delete_instance(compute, args.project_id, args.zone, vm_name)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
