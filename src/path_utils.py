from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def resolve_input_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    data_candidate = DATA_DIR / raw_path
    if data_candidate.exists():
        return data_candidate

    return candidate


def resolve_prompt_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    prompt_candidate = PROMPTS_DIR / raw_path
    if prompt_candidate.exists():
        return prompt_candidate

    return candidate