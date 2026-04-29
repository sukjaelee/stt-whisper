from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
import re
import shutil
import subprocess
import time

from faster_whisper import WhisperModel

ESTIMATE_THRESHOLD_SECONDS = 120.0
ESTIMATE_MIN_WALL_SECONDS = 30.0
ESTIMATE_MIN_PROGRESS_RATIO = 0.1

DEFAULT_WHISPER_MODEL_SIZE = "large-v3"
DEFAULT_WHISPER_LANGUAGE = "ko"
DEFAULT_WHISPER_DEVICE = "auto"
DEFAULT_BEAM_SIZE = 12
DEFAULT_BEST_OF = 5
DEFAULT_PATIENCE = 1.5
DEFAULT_TEMPERATURES = (0.0, 0.2, 0.4, 0.6)
DEFAULT_PROMPT_RESET_ON_TEMPERATURE = 0.5
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.2
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.5
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = False
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_VAD_MIN_SILENCE_DURATION_MS = 500
DEFAULT_VAD_SPEECH_PAD_MS = 450
DEFAULT_SEGMENT_MERGE_GAP_SECONDS = 0.9
DEFAULT_SEGMENT_MERGE_MAX_CHARS = 220
DEFAULT_ENABLE_DENOISE = True
REFERENCE_AUDIO_DURATION_SECONDS = (1 * 3600) + (30 * 60) + 9
REFERENCE_TRANSCRIPTION_DURATION_SECONDS = (19 * 60) + 13
DEFAULT_ESTIMATED_TRANSCRIPTION_RATIO = (
    REFERENCE_TRANSCRIPTION_DURATION_SECONDS / REFERENCE_AUDIO_DURATION_SECONDS
)


@dataclass
class TranscriptionContext:
    started_at: datetime
    started_perf: float
    audio_duration: float | None
    estimate_printed: bool = False


@dataclass(frozen=True)
class WhisperTranscriptionConfig:
    model_size: str = DEFAULT_WHISPER_MODEL_SIZE
    model_path: str | None = None
    language: str = DEFAULT_WHISPER_LANGUAGE
    device: str = DEFAULT_WHISPER_DEVICE
    beam_size: int = DEFAULT_BEAM_SIZE
    best_of: int = DEFAULT_BEST_OF
    patience: float = DEFAULT_PATIENCE
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    prompt_reset_on_temperature: float = DEFAULT_PROMPT_RESET_ON_TEMPERATURE
    compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD
    log_prob_threshold: float = DEFAULT_LOG_PROB_THRESHOLD
    no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD
    condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREVIOUS_TEXT
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    multilingual: bool = False
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = DEFAULT_VAD_MIN_SILENCE_DURATION_MS
    vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS
    enable_denoise: bool = DEFAULT_ENABLE_DENOISE


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def start_transcription_context(audio_duration: float | None = None) -> TranscriptionContext:
    return TranscriptionContext(
        started_at=datetime.now(),
        started_perf=time.perf_counter(),
        audio_duration=audio_duration,
    )


def resolve_runtime_device(requested_device: str = DEFAULT_WHISPER_DEVICE) -> str:
    if requested_device != "auto":
        return requested_device

    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def build_model(config: WhisperTranscriptionConfig | None = None) -> WhisperModel:
    resolved_config = config or WhisperTranscriptionConfig()
    runtime_device = resolve_runtime_device(resolved_config.device)
    model_id = resolved_config.model_path or resolved_config.model_size
    if runtime_device == "cuda":
        return WhisperModel(
            model_id,
            device="cuda",
            compute_type="float16",
        )

    return WhisperModel(
        model_id,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )


def build_transcribe_options(config: WhisperTranscriptionConfig | None = None) -> dict[str, object]:
    resolved_config = config or WhisperTranscriptionConfig()
    return {
        "task": "transcribe",
        "language": resolved_config.language,
        "beam_size": resolved_config.beam_size,
        "best_of": resolved_config.best_of,
        "patience": resolved_config.patience,
        "temperature": resolved_config.temperatures,
        "prompt_reset_on_temperature": resolved_config.prompt_reset_on_temperature,
        "compression_ratio_threshold": resolved_config.compression_ratio_threshold,
        "log_prob_threshold": resolved_config.log_prob_threshold,
        "no_speech_threshold": resolved_config.no_speech_threshold,
        "condition_on_previous_text": resolved_config.condition_on_previous_text,
        "repetition_penalty": resolved_config.repetition_penalty,
        "no_repeat_ngram_size": resolved_config.no_repeat_ngram_size,
        "multilingual": resolved_config.multilingual,
        "vad_filter": resolved_config.vad_filter,
        "vad_parameters": {
            "min_silence_duration_ms": resolved_config.vad_min_silence_duration_ms,
            "speech_pad_ms": resolved_config.vad_speech_pad_ms,
        },
    }


def should_retry_on_cpu(exc: Exception) -> bool:
    message = str(exc).lower()
    cuda_markers = ("cublas", "cudnn", "cudart", "cuda", "nvcuda", "curand", "cufft")
    return ("dll" in message or "load" in message) and any(marker in message for marker in cuda_markers)


def build_preprocessed_audio_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_stt_ready.wav")


def build_audio_filter(enable_denoise: bool = DEFAULT_ENABLE_DENOISE) -> str:
    filters = [
        "highpass=f=60",
        "lowpass=f=7600",
    ]
    if enable_denoise:
        filters.append("afftdn=nf=-25")
    filters.append("loudnorm=I=-16:LRA=7:TP=-1.5")
    return ",".join(filters)


def preprocess_audio_for_transcription(
    audio_path: Path,
    output_path: Path | None = None,
    audio_filter: str | None = None,
) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for preprocessing but was not found in PATH.")

    prepared_path = output_path or build_preprocessed_audio_path(audio_path)
    resolved_audio_filter = audio_filter or build_audio_filter()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-af",
        resolved_audio_filter,
        str(prepared_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        error_output = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg preprocessing failed: {error_output}")
    return prepared_path


def transcribe_with_runtime_fallback(
    audio_path: Path,
    config: WhisperTranscriptionConfig | None = None,
    print_fn: Callable[[str], None] | None = print,
    prepared_audio_path: Path | None = None,
    preprocess_audio: bool = True,
    require_wav_input: bool = False,
) -> tuple[object, object, str, Path]:
    resolved_config = config or WhisperTranscriptionConfig()
    model_id = resolved_config.model_path or resolved_config.model_size
    is_wav_input = audio_path.suffix.lower() == ".wav"
    if require_wav_input and not is_wav_input:
        raise ValueError(f"Expected a .wav input file, got: {audio_path}")

    if is_wav_input or not preprocess_audio:
        resolved_prepared_audio_path = audio_path
    else:
        resolved_prepared_audio_path = preprocess_audio_for_transcription(
            audio_path,
            output_path=prepared_audio_path,
            audio_filter=build_audio_filter(enable_denoise=resolved_config.enable_denoise),
        )
    transcribe_options = build_transcribe_options(resolved_config)
    preferred_device = resolve_runtime_device(resolved_config.device)

    if preferred_device == "cuda":
        try:
            model = build_model(replace(resolved_config, device="cuda"))
            if print_fn:
                print_fn(f"Using CUDA GPU with float16. Model: {model_id}")
                print_fn(f"Prepared audio: {resolved_prepared_audio_path}")
            segments, info = model.transcribe(str(resolved_prepared_audio_path), **transcribe_options)
            return segments, info, "cuda", resolved_prepared_audio_path
        except Exception as exc:
            if not should_retry_on_cpu(exc):
                raise
            if print_fn:
                print_fn(f"CUDA runtime unavailable, retrying on CPU int8: {exc}")

    model = build_model(replace(resolved_config, device="cpu"))
    if print_fn:
        print_fn(f"Using CPU int8. Model: {model_id}")
        print_fn(f"Prepared audio: {resolved_prepared_audio_path}")
    segments, info = model.transcribe(str(resolved_prepared_audio_path), **transcribe_options)
    return segments, info, "cpu", resolved_prepared_audio_path


def build_estimate_message(context: TranscriptionContext, processed_audio_seconds: float) -> str | None:
    if context.estimate_printed or not context.audio_duration:
        return None

    if processed_audio_seconds < ESTIMATE_THRESHOLD_SECONDS:
        return None

    progress_ratio = processed_audio_seconds / context.audio_duration
    if progress_ratio < ESTIMATE_MIN_PROGRESS_RATIO:
        return None

    wall_elapsed_seconds = time.perf_counter() - context.started_perf
    if wall_elapsed_seconds < ESTIMATE_MIN_WALL_SECONDS:
        return None

    estimated_total_seconds = max(
        wall_elapsed_seconds,
        context.audio_duration * DEFAULT_ESTIMATED_TRANSCRIPTION_RATIO,
    )
    estimated_end_time = context.started_at + timedelta(seconds=estimated_total_seconds)
    context.estimate_printed = True
    return (
        "Estimated total duration: "
        f"{format_duration(estimated_total_seconds)} "
        f"(ETA {format_datetime(estimated_end_time)})"
    )


def elapsed_seconds(context: TranscriptionContext) -> float:
    return time.perf_counter() - context.started_perf


def normalize_segment_text(raw_text: str) -> str:
    collapsed = re.sub(r"\s+", " ", raw_text).strip()
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    return collapsed


def process_segments(segments, context: TranscriptionContext, on_segment: Callable[[object], None]) -> None:
    del context
    for seg in segments:
        on_segment(seg)


def collect_transcript_segments(
    segments,
    context: TranscriptionContext,
    on_segment: Callable[[TranscriptSegment], None] | None = None,
) -> list[TranscriptSegment]:
    collected: list[TranscriptSegment] = []

    def handle_segment(seg: object) -> None:
        cleaned_text = normalize_segment_text(getattr(seg, "text", ""))
        if not cleaned_text:
            return

        normalized_segment = TranscriptSegment(
            start=float(getattr(seg, "start", 0.0) or 0.0),
            end=float(getattr(seg, "end", 0.0) or 0.0),
            text=cleaned_text,
        )
        collected.append(normalized_segment)
        if on_segment:
            on_segment(normalized_segment)

    process_segments(segments, context, handle_segment)
    return collected


def merge_transcript_segments(
    segments: list[TranscriptSegment],
    max_gap_seconds: float = DEFAULT_SEGMENT_MERGE_GAP_SECONDS,
    max_chars: int = DEFAULT_SEGMENT_MERGE_MAX_CHARS,
) -> list[str]:
    merged_blocks: list[str] = []
    current_parts: list[str] = []
    previous_end: float | None = None

    for segment in segments:
        if not current_parts:
            current_parts.append(segment.text)
            previous_end = segment.end
            continue

        current_text = " ".join(current_parts)
        gap_seconds = max(0.0, segment.start - (previous_end or segment.start))
        should_split = (
            gap_seconds > max_gap_seconds
            or len(current_text) + len(segment.text) + 1 > max_chars
            or current_parts[-1].endswith((".", "?", "!", "..."))
        )

        if should_split:
            merged_blocks.append(" ".join(current_parts).strip())
            current_parts = [segment.text]
        else:
            current_parts.append(segment.text)

        previous_end = segment.end

    if current_parts:
        merged_blocks.append(" ".join(current_parts).strip())

    return [block for block in merged_blocks if block]


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_timestamped_transcript(path: Path, segments: list[TranscriptSegment]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        for segment in segments:
            file_obj.write(
                f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {segment.text}\n"
            )


def write_plain_transcript(path: Path, blocks: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        for index, block in enumerate(blocks):
            if index:
                file_obj.write("\n\n")
            file_obj.write(block)
        if blocks:
            file_obj.write("\n")
