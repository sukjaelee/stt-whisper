from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from path_utils import PROJECT_ROOT, PROMPTS_DIR, resolve_prompt_path

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CORRECTION_MODEL = os.getenv("OPENAI_CORRECTION_MODEL", "gpt-4.1")
DEFAULT_CHUNK_CHARS = int(os.getenv("OPENAI_CORRECTION_MAX_CHARS", "6000"))
DEFAULT_INSTRUCTIONS_FILE = os.getenv(
    "OPENAI_CORRECTION_PROMPT_FILE",
    str(PROMPTS_DIR / "direct_correction_default.md"),
)


def resolve_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        from google.colab import userdata
    except ImportError:
        return ""

    try:
        api_key = str(userdata.get("OPENAI_API_KEY") or "").strip()
    except Exception:
        return ""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    return api_key


@dataclass
class CorrectionChunk:
    index: int
    text: str


def load_correction_instructions(path: str | Path | None = None) -> str:
    instruction_path = resolve_prompt_path(str(path or DEFAULT_INSTRUCTIONS_FILE))
    if not instruction_path.exists():
        raise RuntimeError(f"Correction instructions file not found: {instruction_path}")

    instructions = instruction_path.read_text(encoding="utf-8").strip()
    if not instructions:
        raise RuntimeError(f"Correction instructions file is empty: {instruction_path}")

    return instructions


def build_chunks(text_segments: Iterable[str], max_chars: int = DEFAULT_CHUNK_CHARS) -> list[CorrectionChunk]:
    chunks: list[CorrectionChunk] = []
    current_parts: list[str] = []
    current_length = 0

    for raw_segment in text_segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        segment_length = len(segment) + (1 if current_parts else 0)
        if current_parts and current_length + segment_length > max_chars:
            chunks.append(CorrectionChunk(index=len(chunks) + 1, text=" ".join(current_parts)))
            current_parts = [segment]
            current_length = len(segment)
            continue

        current_parts.append(segment)
        current_length += segment_length

    if current_parts:
        chunks.append(CorrectionChunk(index=len(chunks) + 1, text=" ".join(current_parts)))

    return chunks


def correct_text_chunks(
    chunks: list[CorrectionChunk],
    model: str = DEFAULT_CORRECTION_MODEL,
    instructions_file: str | Path | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run `uv sync` to install project dependencies."
        ) from exc

    api_key = resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment, .env file, or Colab Secrets."
        )

    instructions = load_correction_instructions(instructions_file)
    client = OpenAI(api_key=api_key)
    corrected_chunks: list[str] = []

    total_chunks = len(chunks)
    for chunk in chunks:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=(
                f"This is chunk {chunk.index} of {total_chunks} from a longer transcript. "
                "Correct only this chunk as a direct corrected transcript and do not infer missing surrounding content.\n\n"
                f"Original text:\n{chunk.text}"
            ),
        )
        corrected_text = response.output_text.strip()
        if corrected_text:
            corrected_chunks.append(corrected_text)

    return "\n\n".join(corrected_chunks).strip()
