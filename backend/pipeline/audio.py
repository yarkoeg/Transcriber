"""Подготовка аудио: любой формат -> 16 kHz mono PCM wav, который ждёт GigaAM."""

import json
import shutil
import subprocess
from pathlib import Path

from ..config import SAMPLE_RATE


class AudioError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _clean(message: str, src: Path) -> str:
    """Убирает из вывода ffmpeg внутренний путь — пользователю он ничего не говорит."""
    return message.replace(str(src), "").strip(" :\n")


def probe_duration(src: Path) -> float:
    """Длительность в секундах. Бросает AudioError, если файл не читается."""
    if not ffmpeg_available():
        raise AudioError("ffmpeg/ffprobe не найдены в PATH. Установите: brew install ffmpeg")

    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"Не удалось прочитать файл как аудио: {_clean(proc.stderr, src)[:400]}")

    try:
        duration = float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError("В файле не найдена аудиодорожка") from exc

    if duration <= 0:
        raise AudioError("Длительность аудио равна нулю")
    return duration


def to_wav16k(src: Path, dst: Path) -> Path:
    """Перекодирует в моно 16 kHz s16le. Идемпотентно перезаписывает dst."""
    if not ffmpeg_available():
        raise AudioError("ffmpeg не найден в PATH. Установите: brew install ffmpeg")

    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(src),
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg не смог перекодировать файл: {_clean(proc.stderr, src)[-400:]}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioError("ffmpeg отработал, но результат пустой")
    return dst
