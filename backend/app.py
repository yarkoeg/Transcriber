"""FastAPI-приложение: загрузка файлов, очередь, выдача результатов."""

import json
import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db, worker
from .pipeline import asr, audio, diarize, render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("transcriber")

CHUNK = 1024 * 1024

# mimetypes на macOS отдаёт для .m4a «audio/mp4a-latm» — Safari такое не играет.
SOURCE_MEDIA_TYPES = {
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}
MAX_UPLOAD_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024


def _startup_checks() -> None:
    """Три самых частых затыка стоит назвать вслух, а не показывать трейсбеком."""
    if not audio.ffmpeg_available():
        log.warning("ffmpeg не найден в PATH — обработка любого файла упадёт. brew install ffmpeg")
    if not config.HF_TOKEN:
        log.warning(
            "HF_TOKEN не задан. Он нужен и для VAD внутри GigaAM, и для диаризации. "
            "Заполните .env по образцу .env.example"
        )
    log.info(
        "Модель: %s | ASR: %s | диаризация: %s (%s)",
        config.GIGAAM_MODEL,
        asr.device(),
        config.DIARIZATION_MODEL if config.DIARIZATION_ENABLED else "выключена",
        diarize.device(),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    interrupted = db.reset_interrupted()
    if interrupted:
        log.warning("Помечено упавшими незавершённых задач: %d", interrupted)
    _startup_checks()
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="Transcriber", lifespan=lifespan)


# --- API ----------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ffmpeg": audio.ffmpeg_available(),
        "hf_token": bool(config.HF_TOKEN),
        "model": config.GIGAAM_MODEL,
        "asr_device": asr.device(),
        "asr_loaded": asr.is_loaded(),
        "diarization_enabled": config.DIARIZATION_ENABLED,
        "diarization_model": config.DIARIZATION_MODEL,
        "diarize_device": diarize.device(),
        "diarization_loaded": diarize.is_loaded(),
        # Ограничения на число говорящих видно снаружи: иначе «почему опять
        # один спикер» решается гаданием.
        "num_speakers": config.NUM_SPEAKERS,
        "min_speakers": config.MIN_SPEAKERS,
        "max_speakers": config.MAX_SPEAKERS,
        "diarization_telemetry": config.DIARIZATION_TELEMETRY,
        "queued": worker.queue_size(),
    }


def _save_upload(src: BinaryIO, dst: Path) -> int:
    written = 0
    with dst.open("wb") as out:
        while chunk := src.read(CHUNK):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Файл больше {config.MAX_UPLOAD_MB} МБ")
            out.write(chunk)
    return written


@app.post("/api/jobs", status_code=201)
async def create_job(
    file: UploadFile = File(...),
    # Не передано — как в .env (DIARIZATION_ENABLED).
    diarize: Optional[bool] = Form(None),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(400, "Файл без имени")

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix or ".bin"
    dst = config.UPLOAD_DIR / f"{job_id}{suffix}"

    # Копирование в пуле потоков: синхронная запись сотен мегабайт в обработчике
    # подвешивала бы event loop, и поллинг прогресса замирал на время загрузки.
    try:
        written = await run_in_threadpool(_save_upload, file.file, dst)
    except HTTPException:
        dst.unlink(missing_ok=True)
        raise

    if written == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "Пустой файл")

    db.create(job_id, file.filename, str(dst), diarize)
    worker.submit(job_id)
    log.info(
        "Задача %s принята: %s (%.1f МБ), разделение говорящих: %s",
        job_id,
        file.filename,
        written / 1024 / 1024,
        "как в .env" if diarize is None else ("да" if diarize else "нет"),
    )
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    return {"jobs": db.list_all()}


def _require_job(job_id: str) -> Dict[str, Any]:
    job = db.get(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    return _require_job(job_id)


def _load_result(job: Dict[str, Any]) -> Dict[str, Any]:
    if job["status"] != "done" or not job.get("json_path"):
        raise HTTPException(409, "Результат ещё не готов")
    path = Path(job["json_path"])
    if not path.exists():
        raise HTTPException(410, "Файл результата удалён с диска")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> Dict[str, Any]:
    return _load_result(_require_job(job_id))


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, fmt: str = "txt") -> FileResponse:
    job = _require_job(job_id)
    if fmt not in ("txt", "json"):
        raise HTTPException(400, "fmt должен быть txt или json")
    path_str = job.get(f"{fmt}_path")
    if job["status"] != "done" or not path_str or not Path(path_str).exists():
        raise HTTPException(409, "Результат ещё не готов")

    stem = Path(job["filename"]).stem
    return FileResponse(
        path_str,
        media_type="application/json" if fmt == "json" else "text/plain; charset=utf-8",
        filename=f"{stem}.{fmt}",
    )


@app.get("/api/jobs/{job_id}/audio")
def audio_file(job_id: str) -> FileResponse:
    job = _require_job(job_id)
    wav = job.get("wav_path")
    if wav and Path(wav).exists():
        return FileResponse(wav, media_type="audio/wav")
    # wav могли удалить руками — тогда играем исходник, браузер понимает m4a/mp3/webm.
    source = job.get("source_path")
    if source and Path(source).exists():
        suffix = Path(source).suffix.lower()
        media_type = SOURCE_MEDIA_TYPES.get(suffix) or mimetypes.guess_type(source)[0]
        return FileResponse(source, media_type=media_type or "application/octet-stream")
    raise HTTPException(404, "Аудио недоступно")


@app.patch("/api/jobs/{job_id}/speakers")
def rename_speakers(job_id: str, names: Dict[str, str] = Body(..., embed=True)) -> Dict[str, Any]:
    """names: {"Спикер 1": "Иван"}. Переписывает и .json, и .txt."""
    job = _require_job(job_id)
    payload = _load_result(job)

    for speaker in payload.get("speakers", []):
        speaker["name"] = names.get(speaker["name"], speaker["name"])
    for utterance in payload.get("utterances", []):
        utterance["speaker"] = names.get(utterance["speaker"], utterance["speaker"])

    Path(job["json_path"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(job["txt_path"]).write_text(
        render.render_txt_from_payload(payload), encoding="utf-8"
    )
    return payload


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, bool]:
    job = _require_job(job_id)
    if job["status"] == "running":
        raise HTTPException(409, "Задача сейчас обрабатывается")

    for key in ("source_path", "wav_path", "txt_path", "json_path"):
        path_str = job.get(key)
        if path_str:
            Path(path_str).unlink(missing_ok=True)

    db.delete(job_id)
    return {"deleted": True}


# --- Статика ------------------------------------------------------------

class NoCacheStatic(StaticFiles):
    """Заставляет браузер сверяться с сервером на каждый запрос.

    Без этого правка frontend/app.js не видна до ручной очистки кэша —
    для локального инструмента лишняя ловушка, а экономить тут нечего.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if config.FRONTEND_DIR.exists():
    app.mount("/", NoCacheStatic(directory=config.FRONTEND_DIR, html=True), name="frontend")
else:  # pragma: no cover
    @app.get("/")
    def index() -> JSONResponse:
        return JSONResponse({"error": "frontend/ не найден"}, status_code=500)
