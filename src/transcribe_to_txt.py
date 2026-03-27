from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from direct_correction import DEFAULT_CORRECTION_MODEL, build_chunks, correct_text_chunks
from path_utils import PROJECT_ROOT, resolve_input_path, resolve_prompt_path
from transcription_utils import (
    WhisperTranscriptionConfig,
    build_estimate_message,
    collect_transcript_segments,
    elapsed_seconds,
    format_datetime,
    format_duration,
    merge_transcript_segments,
    start_transcription_context,
    transcribe_with_runtime_fallback,
    write_plain_transcript,
    write_timestamped_transcript,
)

TRANSCRIBE_PRESETS: dict[str, dict[str, object]] = {
    "high_quality": {
        "model_size": "large-v3",
        "language": "ko",
        "device": "auto",
        "beam_size": 15,
        "best_of": 5,
        "patience": 1.6,
        "temperatures": (0.0, 0.2, 0.4, 0.6),
        "prompt_reset_on_temperature": 0.5,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        "multilingual": False,
        "vad_min_silence_duration_ms": 500,
        "vad_speech_pad_ms": 500,
        "enable_denoise": True,
    },
    "noise_robust": {
        "model_size": "large-v3",
        "language": "ko",
        "device": "auto",
        "beam_size": 12,
        "best_of": 5,
        "patience": 1.5,
        "temperatures": (0.0, 0.2, 0.4, 0.6),
        "prompt_reset_on_temperature": 0.5,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 4,
        "multilingual": False,
        "vad_min_silence_duration_ms": 700,
        "vad_speech_pad_ms": 650,
        "enable_denoise": True,
    },
    "low_hallucination": {
        "model_size": "large-v3",
        "language": "ko",
        "device": "auto",
        "beam_size": 10,
        "best_of": 3,
        "patience": 1.2,
        "temperatures": (0.0, 0.2, 0.4),
        "prompt_reset_on_temperature": 0.4,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 4,
        "multilingual": False,
        "vad_min_silence_duration_ms": 800,
        "vad_speech_pad_ms": 700,
        "enable_denoise": False,
    },
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def build_corrected_output_path(txt_path: Path) -> Path:
    return txt_path.with_name(f"{txt_path.stem}_direct_corrected.txt")


def build_segments_output_path(txt_path: Path) -> Path:
    return txt_path.with_name(f"{txt_path.stem}_segments.txt")


def resolve_output_paths(audio_path: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    base_dir = output_dir or audio_path.parent
    txt_path = base_dir / f"{audio_path.stem}.txt"
    segments_path = build_segments_output_path(txt_path)
    prepared_audio_path = base_dir / f"{audio_path.stem}_stt_ready.wav"
    return txt_path, segments_path, prepared_audio_path


def build_config(
    preset_name: str,
    device: str | None = None,
    language: str | None = None,
    model_size: str | None = None,
    model_path: str | None = None,
) -> WhisperTranscriptionConfig:
    if preset_name not in TRANSCRIBE_PRESETS:
        available = ", ".join(sorted(TRANSCRIBE_PRESETS))
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")

    preset = dict(TRANSCRIBE_PRESETS[preset_name])
    if device:
        preset["device"] = device
    if language:
        preset["language"] = language
    if model_size:
        preset["model_size"] = model_size
    resolved_model_path = model_path or os.getenv("WHISPER_MODEL_PATH")
    if resolved_model_path:
        preset["model_path"] = resolved_model_path

    return WhisperTranscriptionConfig(
        model_size=str(preset["model_size"]),
        model_path=str(preset["model_path"]) if preset.get("model_path") else None,
        language=str(preset["language"]),
        device=str(preset["device"]),
        beam_size=int(preset["beam_size"]),
        best_of=int(preset["best_of"]),
        patience=float(preset["patience"]),
        temperatures=tuple(float(v) for v in preset["temperatures"]),
        prompt_reset_on_temperature=float(preset["prompt_reset_on_temperature"]),
        repetition_penalty=float(preset["repetition_penalty"]),
        no_repeat_ngram_size=int(preset["no_repeat_ngram_size"]),
        multilingual=bool(preset["multilingual"]),
        vad_min_silence_duration_ms=int(preset["vad_min_silence_duration_ms"]),
        vad_speech_pad_ms=int(preset["vad_speech_pad_ms"]),
        enable_denoise=bool(preset["enable_denoise"]),
    )


def run_transcribe_to_txt(
    input_audio_path: str,
    preset_name: str = "high_quality",
    skip_correction: bool = True,
    correction_model: str = DEFAULT_CORRECTION_MODEL,
    chunk_chars: int = 6000,
    instructions_file: str = "prompts/direct_correction_default.md",
    corrected_output_path: str | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    language: str | None = None,
    model_size: str | None = None,
    model_path: str | None = None,
) -> dict[str, str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be a positive integer.")

    resolved_output_dir = Path(output_dir) if output_dir else None
    if resolved_output_dir:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        audio_path = resolve_input_path(input_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Input file not found: {audio_path}")

        config = build_config(
            preset_name=preset_name,
            device=device,
            language=language,
            model_size=model_size,
            model_path=model_path,
        )
        txt_path, segments_path, prepared_audio_path = resolve_output_paths(audio_path, resolved_output_dir)

        initial_context = start_transcription_context()
        log(f"Transcription start time: {format_datetime(initial_context.started_at)}")
        log(f"Input audio: {audio_path}")
        log(f"Output directory: {resolved_output_dir or audio_path.parent}")
        log(f"Preset: {preset_name}")
        log(f"Config: {config}")
        log(f"Prepared audio path: {prepared_audio_path}")
        log("Starting audio preprocessing and model selection")

        segments, info, runtime_device, prepared_audio_path = transcribe_with_runtime_fallback(
            audio_path,
            config=config,
            prepared_audio_path=prepared_audio_path,
        )
        log("Model invocation started")

        context = start_transcription_context(getattr(info, "duration", None))
        context.started_at = initial_context.started_at
        context.started_perf = initial_context.started_perf
        if getattr(info, "duration", None):
            log(f"Detected audio duration: {format_duration(float(getattr(info, 'duration', 0.0)))}")

        def on_segment(seg: object) -> None:
            estimate_message = build_estimate_message(context, getattr(seg, "end", 0.0) or 0.0)
            if estimate_message:
                log(estimate_message)

        log("Collecting transcript segments")
        transcript_segments = collect_transcript_segments(segments, context, on_segment)
        log(f"Collected segments: {len(transcript_segments)}")
        log("Merging transcript segments")
        merged_blocks = merge_transcript_segments(transcript_segments)
        log(f"Merged transcript blocks: {len(merged_blocks)}")
        log(f"Writing timestamped transcript: {segments_path}")
        write_timestamped_transcript(segments_path, transcript_segments)
        log(f"Writing raw transcript: {txt_path}")
        write_plain_transcript(txt_path, merged_blocks)

        transcription_finished_at = datetime.now()
        log(f"Runtime device: {runtime_device}")
        log(f"Prepared audio: {prepared_audio_path}")
        log(f"Timestamped segments: {segments_path}")
        log(f"Raw transcript: {txt_path}")
        log(f"Transcription end time: {format_datetime(transcription_finished_at)}")
        log(f"Transcription elapsed time: {format_duration(elapsed_seconds(context))}")

        result = {
            "prepared_audio": str(prepared_audio_path),
            "segments_transcript": str(segments_path),
            "raw_transcript": str(txt_path),
            "preset": preset_name,
            "runtime_device": runtime_device,
            "model": config.model_path or config.model_size,
        }

        if skip_correction:
            return result

        correction_started_at = datetime.now()
        correction_context = start_transcription_context()
        log(f"Correction start time: {format_datetime(correction_started_at)}")

        output_path = Path(corrected_output_path) if corrected_output_path else build_corrected_output_path(txt_path)
        instructions_path = resolve_prompt_path(instructions_file)
        chunks = build_chunks(merged_blocks, max_chars=chunk_chars)
        log(f"Correction chunks: {len(chunks)}")
        log(f"Correction model: {correction_model}")
        log(f"Instructions file: {instructions_path}")

        corrected_text = correct_text_chunks(
            chunks,
            model=correction_model,
            instructions_file=instructions_path,
        )

        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(corrected_text)
            if corrected_text and not corrected_text.endswith("\n"):
                file_obj.write("\n")

        correction_finished_at = datetime.now()
        log(f"Direct corrected transcript: {output_path}")
        log(f"Correction end time: {format_datetime(correction_finished_at)}")
        log(f"Correction elapsed time: {format_duration(elapsed_seconds(correction_context))}")
        log(f"Overall elapsed time: {format_duration(elapsed_seconds(context))}")

        result["corrected_transcript"] = str(output_path)
        return result
    finally:
        os.chdir(original_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe an audio file to text.")
    parser.add_argument("input_audio_path", help="Path to an input audio file.")
    parser.add_argument(
        "--preset",
        default="high_quality",
        choices=sorted(TRANSCRIBE_PRESETS),
        help="Named transcription preset.",
    )
    parser.add_argument(
        "--skip-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the LLM correction step.",
    )
    parser.add_argument(
        "--correction-model",
        default=DEFAULT_CORRECTION_MODEL,
        help="Model used for direct correction. Defaults to OPENAI_CORRECTION_MODEL from .env.",
    )
    parser.add_argument("--chunk-chars", type=int, default=6000, help="Maximum characters per correction chunk.")
    parser.add_argument(
        "--instructions-file",
        default="prompts/direct_correction_default.md",
        help="Prompt file used for correction.",
    )
    parser.add_argument("--corrected-output-path", default=None, help="Explicit output path for corrected text.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated outputs.")
    parser.add_argument("--device", default=None, help="Override Whisper device: auto, cuda, or cpu.")
    parser.add_argument("--language", default=None, help="Override transcription language.")
    parser.add_argument("--model-size", default=None, help="Override Whisper model size.")
    parser.add_argument("--model-path", default=None, help="Path to a local CTranslate2 Whisper model directory.")
    parser.add_argument("--json", action="store_true", help="Emit the result payload as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_transcribe_to_txt(
        input_audio_path=args.input_audio_path,
        preset_name=args.preset,
        skip_correction=args.skip_correction,
        correction_model=args.correction_model,
        chunk_chars=args.chunk_chars,
        instructions_file=args.instructions_file,
        corrected_output_path=args.corrected_output_path,
        output_dir=args.output_dir,
        device=args.device,
        language=args.language,
        model_size=args.model_size,
        model_path=args.model_path,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
