"""Фоновая очередь обработки.

Задачи выполняются строго последовательно и в одном потоке: модели занимают
1–2.5 ГБ каждая, а безвентиляторный Air на параллельной нагрузке уходит
в троттлинг быстрее, чем успевает что-то выиграть.
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from . import config, db, pipeline

log = logging.getLogger(__name__)

_queue: "queue.Queue[str]" = queue.Queue()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def submit(job_id: str) -> None:
    _queue.put(job_id)


def queue_size() -> int:
    return _queue.qsize()


def _process(job_id: str) -> None:
    job = db.get(job_id)
    if job is None:
        return

    started = time.monotonic()
    wav_path = config.WAV_DIR / f"{job_id}.wav"
    db.update(
        job_id,
        status="running",
        stage="decoding",
        progress=0.0,
        error=None,
        # Путь пишем до обработки, а не по успеху: иначе wav упавшей или
        # прерванной задачи остаётся на диске сиротой, и удаление его не находит.
        wav_path=str(wav_path),
        started_at=db.now(),
        # Сбрасываем итоги прошлой попытки, иначе на повторной постановке
        # в очередь висело бы чужое время.
        processing_seconds=None,
        finished_at=None,
    )

    def on_progress(stage: str, progress: float) -> None:
        db.update(job_id, stage=stage, progress=round(progress, 4))

    try:
        result = pipeline.run(
            Path(job["source_path"]),
            out_stem=job_id,
            wav_path=wav_path,
            on_progress=on_progress,
            # NULL у задач, созданных до появления переключателя, — берём из .env.
            diarization=None if job.get("diarize") is None else bool(job["diarize"]),
        )
    except Exception as exc:  # noqa: BLE001 - любая ошибка должна попасть в UI
        elapsed = round(time.monotonic() - started, 2)
        # «Упало сразу» и «упало на четвёртой минуте» — разные диагнозы.
        log.exception("Задача %s упала через %.1f с", job_id, elapsed)
        db.update(
            job_id,
            status="failed",
            stage=None,
            error=str(exc),
            processing_seconds=elapsed,
            finished_at=db.now(),
        )
        return

    db.update(
        job_id,
        status="done",
        stage="done",
        progress=1.0,
        duration=result["duration"],
        processing_seconds=round(time.monotonic() - started, 2),
        txt_path=result["txt_path"],
        json_path=result["json_path"],
        wav_path=result["wav_path"],
        speaker_count=result["speaker_count"],
        utterance_count=result["utterance_count"],
        meta=result,
        finished_at=db.now(),
    )
    # Исходник после обработки не нужен: плеер играет wav, а это часто видео
    # на сотни мегабайт. У упавших задач он остаётся — по нему видно, что не так.
    Path(job["source_path"]).unlink(missing_ok=True)


def _loop() -> None:
    while not _stop.is_set():
        try:
            job_id = _queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            _process(job_id)
        finally:
            _queue.task_done()


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="transcribe-worker", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)
