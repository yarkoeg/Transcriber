"""Настройки приложения. Всё переопределяется через переменные окружения / .env."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip())
    except ValueError:
        return default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Пустое значение, мусор и бессмысленные числа (<= 0) дают default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# --- Пути ---------------------------------------------------------------

DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
WAV_DIR = DATA_DIR / "wav"
RESULT_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "app.db"
FRONTEND_DIR = ROOT / "frontend"

for _dir in (UPLOAD_DIR, WAV_DIR, RESULT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --- Модели -------------------------------------------------------------

# Только v3_e2e_* дают пунктуацию и нормализацию текста.
GIGAAM_MODEL = _env("GIGAAM_MODEL", "v3_e2e_rnnt")

# cpu | mps. На MPS считаются VAD и энкодер (в fp32), декодер всегда на CPU —
# см. asr.py. На M4 запись 13.7 мин распознаётся за 25 с против 247 с на CPU,
# текст побайтно тот же. Без MPS откатывается на CPU.
ASR_DEVICE = _env("ASR_DEVICE", "mps")

# Та же картина: 75 с против 512 с, разметка совпадает.
DIARIZE_DEVICE = _env("DIARIZE_DEVICE", "mps")

# В pyannote.audio 4.x откатиться на speaker-diarization-3.1 нельзя: SpeakerDiarization
# всё равно тянет PLDA из community-1, так что доступ нужен именно к нему.
DIARIZATION_MODEL = _env("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
DIARIZATION_ENABLED = _env_bool("DIARIZATION_ENABLED", True)

# Эксклюзивная разметка не содержит перекрывающейся речи — ровно то, что нужно
# merge.py: слово не попадает сразу в два turn'а. Выключать, только если
# привязка слов к говорящим окажется хуже обычной разметки.
DIARIZATION_EXCLUSIVE = _env_bool("DIARIZATION_EXCLUSIVE", True)

HF_TOKEN = _env("HF_TOKEN", "")

# Число говорящих. Пусто — pyannote решает сам (кластеризация VBx это умеет).
# NUM_SPEAKERS перекрывает MIN/MAX: так устроен сам pyannote.
NUM_SPEAKERS = _env_int("NUM_SPEAKERS")
MIN_SPEAKERS = _env_int("MIN_SPEAKERS")
MAX_SPEAKERS = _env_int("MAX_SPEAKERS")

# Противоречивые границы гасим здесь: pyannote бросит ValueError уже внутри
# обработки, когда файл прогнан через ffmpeg и потрачены минуты.
if MIN_SPEAKERS and MAX_SPEAKERS and MIN_SPEAKERS > MAX_SPEAKERS:
    MIN_SPEAKERS = MAX_SPEAKERS = None

# pyannote.audio 4.x по умолчанию отправляет длительность записи и запрошенное
# число говорящих на otel.pyannote.ai. Локальный инструмент наружу ничего слать
# не должен. Переменную читает сам pyannote — важно выставить её до его импорта.
DIARIZATION_TELEMETRY = _env_bool("DIARIZATION_TELEMETRY", False)
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", str(DIARIZATION_TELEMETRY).lower())


# --- Параметры склейки --------------------------------------------------

# Максимальное расстояние (сек) до ближайшего speaker turn, на котором мы
# ещё готовы приписать слово этому спикеру.
SPEAKER_SNAP_WINDOW = _env_float("SPEAKER_SNAP_WINDOW", 0.5)

# Пауза (сек), по которой разрываем реплику одного и того же спикера.
UTTERANCE_GAP = _env_float("UTTERANCE_GAP", 1.0)


# --- Прочее -------------------------------------------------------------

SAMPLE_RATE = 16000
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "2048"))
