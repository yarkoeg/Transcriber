"""CLI для пайплайна — позволяет проверять обработку без веб-сервера.

    python -m backend.pipeline --input запись.m4a
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from . import STAGE_NAMES, STAGES, run


def _format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return f"{total} с"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин {rest} с" if rest else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин"


def _print_timings(stage_seconds: Dict[str, float]) -> None:
    width = max(len(name) for name in STAGE_NAMES.values())
    # Порядок берём из STAGES: словарь таймингов упорядочен по факту выполнения,
    # а пропущенной стадии диаризации в нём просто нет.
    for stage in STAGES:
        seconds = stage_seconds.get(stage)
        if seconds is not None:
            print(f"  {STAGE_NAMES[stage]:<{width}}  {seconds:6.1f} с")


def main() -> int:
    parser = argparse.ArgumentParser(description="Транскрибация аудио через GigaAM")
    parser.add_argument("--input", "-i", required=True, type=Path, help="путь к аудиофайлу")
    parser.add_argument("--name", "-n", help="имя для выходных файлов (по умолчанию — имя входного)")
    parser.add_argument("--speakers", "-s", type=int, help="точное число говорящих, если известно")
    parser.add_argument("--min-speakers", type=int, help="нижняя граница числа говорящих")
    parser.add_argument("--max-speakers", type=int, help="верхняя граница числа говорящих")
    parser.add_argument(
        "--diarization",
        action=argparse.BooleanOptionalAction,
        help="разделять ли по говорящим (по умолчанию — DIARIZATION_ENABLED из .env)",
    )
    args = parser.parse_args()

    # Без этого логи диаризации (сколько интервалов, сколько говорящих) в CLI
    # не видны — а именно через CLI пайплайн и проверяют.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )

    if not args.input.exists():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 1

    def on_progress(stage: str, progress: float) -> None:
        print(f"[{progress * 100:5.1f}%] {stage}", file=sys.stderr)

    try:
        result = run(
            args.input,
            out_stem=args.name or args.input.stem,
            on_progress=on_progress,
            num_speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarization=args.diarization,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    rtf: Optional[float] = result.get("realtime_factor")  # type: ignore[assignment]
    print(
        f"\nГотово за {_format_elapsed(float(result['pipeline_seconds']))}"
        + (f" (RTF {rtf}×)" if rtf else "")
        + f" — {result['utterance_count']} реплик, {result['speaker_count']} говорящих"
    )
    _print_timings(result.get("stage_seconds") or {})  # type: ignore[arg-type]
    print(f"  {result['txt_path']}")
    print(f"  {result['json_path']}")
    print(Path(str(result["txt_path"])).read_text(encoding="utf-8")[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
