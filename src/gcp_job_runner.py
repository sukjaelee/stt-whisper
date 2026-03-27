from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from transcribe_to_txt import run_transcribe_to_txt


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri}")
    bucket_name, _, object_name = uri[5:].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Expected a full gs://bucket/object URI, got: {uri}")
    return bucket_name, object_name


def join_gs_uri(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


def upload_text(client: storage.Client, uri: str, text: str) -> None:
    bucket_name, object_name = parse_gs_uri(uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(text, content_type="application/json; charset=utf-8")


def upload_file(client: storage.Client, path: Path, uri: str) -> None:
    bucket_name, object_name = parse_gs_uri(uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(path))


def download_file(client: storage.Client, uri: str, destination: Path) -> None:
    bucket_name, object_name = parse_gs_uri(uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one transcription job on a GCP VM.")
    parser.add_argument("--input-uri", required=True, help="Input audio object, for example gs://bucket/path/file.mp3")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix, for example gs://bucket/jobs/job-123",
    )
    parser.add_argument("--job-id", required=True, help="Unique job identifier.")
    parser.add_argument("--work-root", default="/tmp/stt-whisper-job", help="Working directory on the VM.")
    parser.add_argument("--preset", default="high_quality", help="Transcription preset.")
    parser.add_argument("--device", default="auto", help="Whisper runtime device.")
    parser.add_argument("--model-path", default=None, help="Local CTranslate2 Whisper model directory on the VM.")
    parser.add_argument(
        "--skip-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the LLM correction step. Disable only if the VM already has OPENAI_API_KEY configured.",
    )
    parser.add_argument(
        "--keep-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the prepared wav file as part of the outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = storage.Client()
    work_root = Path(args.work_root)
    input_dir = work_root / "input"
    output_dir = work_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name = Path(parse_gs_uri(args.input_uri)[1]).name
    local_input_path = input_dir / input_name
    started_at = datetime.now(timezone.utc)

    try:
        log(f"Job start: {args.job_id}")
        log(f"Input URI: {args.input_uri}")
        log(f"Output prefix: {args.output_prefix}")
        log(f"Work root: {work_root}")
        if args.model_path:
            log(f"Configured local model path: {args.model_path}")
        log(f"Skip correction: {args.skip_correction}")
        log(f"Downloading input: {args.input_uri}")
        download_file(client, args.input_uri, local_input_path)
        log(f"Input downloaded to: {local_input_path}")
        log("Starting transcription")

        result = run_transcribe_to_txt(
            input_audio_path=str(local_input_path),
            preset_name=args.preset,
            skip_correction=args.skip_correction,
            output_dir=output_dir,
            device=args.device,
            model_path=args.model_path,
        )
        log("Transcription finished")

        uploaded_files: dict[str, str] = {}
        for key in ("raw_transcript", "segments_transcript", "prepared_audio"):
            path_value = result.get(key)
            if not path_value:
                continue
            local_path = Path(path_value)
            if key == "prepared_audio" and not args.keep_preprocessed:
                log(f"Skipping prepared audio upload: {local_path}")
                continue
            destination_uri = join_gs_uri(args.output_prefix, local_path.name)
            log(f"Uploading {local_path} -> {destination_uri}")
            upload_file(client, local_path, destination_uri)
            uploaded_files[key] = destination_uri

        payload = {
            "job_id": args.job_id,
            "status": "success",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "input_uri": args.input_uri,
            "output_prefix": args.output_prefix,
            "result": result,
            "uploaded_files": uploaded_files,
        }
        log("Uploading success marker")
        upload_text(
            client,
            join_gs_uri(args.output_prefix, "_SUCCESS.json"),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        log("Job completed successfully")
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        log(f"Job failed: {exc}")
        payload = {
            "job_id": args.job_id,
            "status": "failed",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "input_uri": args.input_uri,
            "output_prefix": args.output_prefix,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        log("Uploading failure marker")
        upload_text(
            client,
            join_gs_uri(args.output_prefix, "_FAILED.json"),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
