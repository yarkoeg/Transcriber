"""Полный пайплайн транскрибации: аудио -> текст с разделением по говорящим."""

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from .. import config
from . import asr, audio, diarize, merge, render
from .types import Turn

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]

# Доля общего прогресса на стадию — по замерам на M4 (mps): диаризация
# занимает около трёх четвертей времени, распознавание — четверть.
STAGES = {
    "decoding": (0.0, 0.02),
    "diarizing": (0.02, 0.72),
    "transcribing": (0.72, 0.98),
    "merging": (0.98, 1.0),
}

# Человекочитаемые названия стадий для логов и CLI. У фронтенда своя копия
# (STAGE_LABELS в app.js) — он живёт без сборки и импортировать отсюда не может.
STAGE_NAMES = {
    "decoding": "подготовка аудио",
    "diarizing": "разделение говорящих",
    "transcribing": "распознавание",
    "merging": "сборка",
}


def _noop(stage: str, progress: float) -> None:
    pass


@contextmanager
def _timed(name: str, timings: Dict[str, float]) -> Iterator[None]:
    """monotonic, а не time(): перевод системных часов не должен ломать замер."""
    started = time.monotonic()
    try:
        yield
    finally:
        timings[name] = round(time.monotonic() - started, 2)


def run(
    src: Path,
    out_stem: str,
    wav_path: Optional[Path] = None,
    on_progress: ProgressCallback = _noop,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarization: Optional[bool] = None,
) -> Dict[str, object]:
    """Обрабатывает один файл и пишет .txt и .json в config.RESULT_DIR.

    diarization=None — как в .env (DIARIZATION_ENABLED); веб передаёт выбор
    пользователя для конкретной задачи. Возвращает метаданные результата,
    включая пути к файлам.
    """
    src = Path(src)
    diarization_enabled = config.DIARIZATION_ENABLED if diarization is None else diarization
    wav_path = wav_path or (config.WAV_DIR / f"{out_stem}.wav")

    run_started = time.monotonic()
    timings: Dict[str, float] = {}
    # Первая задача после старта процесса платит за загрузку моделей (десятки
    # секунд). Без этого флага её низкая скорость выглядит необъяснимым выбросом.
    cold_start = not asr.is_loaded() or (
        diarization_enabled and not diarize.is_loaded()
    )

    on_progress("decoding", STAGES["decoding"][0])
    with _timed("decoding", timings):
        duration = audio.probe_duration(src)
        audio.to_wav16k(src, wav_path)
    on_progress("decoding", STAGES["decoding"][1])

    turns: List[Turn] = []
    if diarization_enabled:
        low, high = STAGES["diarizing"]
        on_progress("diarizing", low)
        # Ключ diarizing появляется в таймингах, только если стадия реально шла:
        # ноль был бы неотличим от «отработала мгновенно».
        with _timed("diarizing", timings):
            turns = diarize.diarize(
                wav_path,
                # None из веба и CLI означает «взять из .env», а не «без ограничений».
                num_speakers=num_speakers if num_speakers is not None else config.NUM_SPEAKERS,
                min_speakers=min_speakers if min_speakers is not None else config.MIN_SPEAKERS,
                max_speakers=max_speakers if max_speakers is not None else config.MAX_SPEAKERS,
                on_progress=lambda fraction: on_progress(
                    "diarizing", low + (high - low) * fraction
                ),
            )
    low, high = STAGES["transcribing"]
    if not diarization_enabled:
        # Пропущенная стадия не должна съедать полосу: иначе прогресс прыгает
        # с 2% сразу на 72%.
        low = STAGES["decoding"][1]
    on_progress("transcribing", low)
    with _timed("transcribing", timings):
        segments = asr.transcribe(
            wav_path,
            on_progress=lambda fraction: on_progress("transcribing", low + (high - low) * fraction),
        )
    if not segments:
        raise ValueError("В записи не найдено речи")
    on_progress("merging", STAGES["merging"][0])

    with _timed("merging", timings):
        utterances, labels = merge.merge(segments, turns)

    # Фиксируем до сборки meta: сам meta пишется внутри write_outputs, и время
    # записи файлов сюда уже не попадёт — оно учтено в worker.processing_seconds.
    pipeline_seconds = round(time.monotonic() - run_started, 2)
    realtime_factor = round(duration / pipeline_seconds, 1) if pipeline_seconds > 0 else None

    meta: Dict[str, object] = {
        "source_file": src.name,
        "duration": round(duration, 2),
        "model": config.GIGAAM_MODEL,
        "asr_device": asr.device(),
        "diarization": config.DIARIZATION_MODEL if diarization_enabled else None,
        "diarize_device": diarize.device() if diarization_enabled else None,
        # Явный флаг и число интервалов: без них «один говорящий» неотличим
        # от «диаризация была выключена».
        "diarization_enabled": diarization_enabled,
        "diarization_turns": len(turns),
        "word_timestamps": asr.has_word_timestamps(segments),
        "speaker_count": len(labels),
        "pipeline_seconds": pipeline_seconds,
        "stage_seconds": timings,
        # Как в scripts/benchmark.py: во сколько раз быстрее реального времени.
        "realtime_factor": realtime_factor,
        "cold_start": cold_start,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    txt_path = config.RESULT_DIR / f"{out_stem}.txt"
    json_path = config.RESULT_DIR / f"{out_stem}.json"
    render.write_outputs(utterances, labels, meta, txt_path, json_path)

    on_progress("merging", STAGES["merging"][1])

    log.info(
        "Задача %s: %.1f с на %.1f с аудио (RTF %s×), %d реплик, %d говорящих, стадии %s%s",
        out_stem,
        pipeline_seconds,
        duration,
        realtime_factor,
        len(utterances),
        len(labels),
        timings,
        " [холодный старт]" if cold_start else "",
    )

    return {
        **meta,
        "txt_path": str(txt_path),
        "json_path": str(json_path),
        "wav_path": str(wav_path),
        "utterance_count": len(utterances),
    }
