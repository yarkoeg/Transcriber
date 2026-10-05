"""Формирование выходных файлов: читаемый .txt и полный .json."""

import json
from pathlib import Path
from typing import Dict, List

from .types import Utterance


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def render_txt(utterances: List[Utterance], labels: Dict[str, str]) -> str:
    lines = []
    for utterance in utterances:
        name = labels.get(utterance.speaker, utterance.speaker)
        lines.append(f"{name} [{format_timestamp(utterance.start)}]: {utterance.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def build_json(
    utterances: List[Utterance],
    labels: Dict[str, str],
    meta: Dict[str, object],
) -> Dict[str, object]:
    return {
        "meta": meta,
        "speakers": [
            {"id": raw_id, "name": name} for raw_id, name in labels.items()
        ],
        "utterances": [
            {
                "speaker": labels.get(utterance.speaker, utterance.speaker),
                "speaker_id": utterance.speaker,
                "start": round(utterance.start, 3),
                "end": round(utterance.end, 3),
                "text": utterance.text,
                "words": [
                    {
                        "text": word.text,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                    }
                    for word in utterance.words
                ],
            }
            for utterance in utterances
        ],
    }


def render_txt_from_payload(payload: Dict[str, object]) -> str:
    """Пересобирает .txt из сохранённого .json — нужно при переименовании спикеров."""
    lines = []
    for utterance in payload.get("utterances", []):  # type: ignore[union-attr]
        lines.append(
            f"{utterance['speaker']} [{format_timestamp(utterance['start'])}]: {utterance['text']}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def write_outputs(
    utterances: List[Utterance],
    labels: Dict[str, str],
    meta: Dict[str, object],
    txt_path: Path,
    json_path: Path,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(render_txt(utterances, labels), encoding="utf-8")
    json_path.write_text(
        json.dumps(build_json(utterances, labels, meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


