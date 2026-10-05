"""Проверка GigaAM в изоляции, до всякого веба.

Отвечает на три вопроса:
  1. Скачивается ли модель и грузится ли она на выбранном устройстве.
  2. Приходят ли таймкоды слов (от этого зависит точность привязки спикеров).
  3. Есть ли в тексте пунктуация (её дают только версии v3_e2e_*).

    python scripts/smoke_test.py [путь_к_аудио]

Без аргумента пытается синтезировать тестовую фразу через штатный macOS `say`.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.pipeline import asr, audio  # noqa: E402

PHRASE = (
    "Привет! Это проверка распознавания речи. "
    "Сегодня четвёртое августа, температура двадцать три градуса. "
    "Всё работает как надо?"
)


def russian_voice() -> str | None:
    proc = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if "ru_RU" in line:
            return line.split()[0]
    return None


def make_sample(dst: Path) -> Path | None:
    voice = russian_voice()
    if voice is None:
        return None
    aiff = dst.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), PHRASE], check=True)
    return aiff


def main() -> int:
    tmp = config.DATA_DIR / "smoke"
    tmp.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        if not src.exists():
            print(f"Файл не найден: {src}")
            return 1
    else:
        src = make_sample(tmp / "sample")
        if src is None:
            print(
                "В системе нет русского голоса для `say`. Установите его в\n"
                "Системные настройки → Универсальный доступ → Устный контент → Системный голос,\n"
                "либо передайте свой файл: python scripts/smoke_test.py запись.m4a"
            )
            return 1
        print(f"Сгенерирован тестовый файл: {src}")

    if not audio.ffmpeg_available():
        print("Нет ffmpeg. Установите: brew install ffmpeg")
        return 1

    wav = audio.to_wav16k(src, tmp / "sample.wav")
    duration = audio.probe_duration(wav)
    print(f"Длительность: {duration:.1f} с")

    print(f"Загружаю {config.GIGAAM_MODEL} на {config.ASR_DEVICE}…")
    started = time.perf_counter()
    asr.load()
    print(f"Модель загружена за {time.perf_counter() - started:.1f} с (энкодер на {asr.device()})")

    started = time.perf_counter()
    segments = asr.transcribe(wav)
    elapsed = time.perf_counter() - started
    print(f"Распознавание: {elapsed:.1f} с (RTF {duration / elapsed:.1f}x реального времени)\n")

    if not segments:
        print("РЕЗУЛЬТАТ ПУСТОЙ — речь не найдена")
        return 1

    for segment in segments:
        print(f"[{segment.start:7.2f} – {segment.end:7.2f}] {segment.text}")
        if segment.words:
            preview = ", ".join(f"{w.text}@{w.start:.2f}" for w in segment.words[:8])
            print(f"    слова: {preview}{' …' if len(segment.words) > 8 else ''}")

    text = " ".join(segment.text for segment in segments)
    has_words = asr.has_word_timestamps(segments)
    has_punctuation = any(char in text for char in ".,!?")

    print("\n--- Проверки ---")
    print(f"таймкоды слов:  {'да' if has_words else 'НЕТ — привязка спикеров будет грубее'}")
    print(f"пунктуация:     {'да' if has_punctuation else 'НЕТ — возьмите модель v3_e2e_*'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
