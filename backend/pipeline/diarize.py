"""Диаризация говорящих через pyannote. В самом GigaAM её нет — только VAD."""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import config
from .types import Turn

log = logging.getLogger(__name__)

_pipeline = None
_device: Optional[str] = None
_lock = threading.Lock()


class DiarizationError(RuntimeError):
    pass


def _gated_hint() -> str:
    """Подсказка про gated-модели — со ссылкой на ту модель, которая грузится.

    Захардкоженное имя уводит не туда: пользователь примет условия на странице
    модели, которую приложение даже не запрашивает.
    """
    return (
        "Проверьте, что HF_TOKEN задан в .env и что вы приняли условия использования "
        "на обеих страницах:\n"
        f"  https://huggingface.co/{config.DIARIZATION_MODEL} — диаризация\n"
        "  https://huggingface.co/pyannote/segmentation-3.0 — VAD внутри GigaAM"
    )


def _is_access_denied(exc: BaseException) -> bool:
    """401/403 где-нибудь в цепочке исключений huggingface_hub.

    Без этой проверки отсутствие сети получает подсказку про принятие условий
    и отправляет разбираться не туда.
    """
    current: Optional[BaseException] = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) in (401, 403):
            return True
        current = current.__cause__ or current.__context__
    return False


def device() -> str:
    """Устройство, на котором диаризация реально считается.

    MPS по умолчанию, но его может не быть (Intel Mac, старый macOS) — тогда
    откатываемся на CPU, а не падаем на первой же задаче.
    """
    global _device
    if _device is None:
        _device = config.DIARIZE_DEVICE
        if _device == "mps":
            import torch

            if not torch.backends.mps.is_available():
                log.warning("MPS недоступен — диаризация пойдёт на CPU, это в разы медленнее")
                _device = "cpu"
    return _device


def load() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _lock:
        if _pipeline is not None:
            return _pipeline

        if not config.HF_TOKEN:
            raise DiarizationError(
                "Не задан HF_TOKEN — без него не скачать модель диаризации.\n" + _gated_hint()
            )

        import torch
        from pyannote.audio import Pipeline

        started = time.monotonic()
        try:
            pipeline = Pipeline.from_pretrained(config.DIARIZATION_MODEL, token=config.HF_TOKEN)
        except Exception as exc:  # noqa: BLE001
            if _is_access_denied(exc):
                raise DiarizationError(
                    f"Hugging Face не даёт доступ к {config.DIARIZATION_MODEL}.\n" + _gated_hint()
                ) from exc
            raise DiarizationError(
                f"Не удалось загрузить {config.DIARIZATION_MODEL}: {exc}"
            ) from exc

        if pipeline is None:
            # from_pretrained возвращает None при отказе в доступе, без исключения.
            raise DiarizationError(
                f"Hugging Face не отдал модель {config.DIARIZATION_MODEL}.\n" + _gated_hint()
            )

        _pipeline = pipeline.to(torch.device(device()))
        log.info(
            "Модель диаризации %s загружена за %.1f с (устройство %s, legacy=%s, кластеризация=%s)",
            config.DIARIZATION_MODEL,
            time.monotonic() - started,
            device(),
            getattr(_pipeline, "legacy", "?"),
            getattr(_pipeline, "klustering", "?"),
        )

    return _pipeline


def is_loaded() -> bool:
    return _pipeline is not None


def _extract_annotation(output: Any) -> Tuple[Any, str]:
    """Достаёт разметку из ответа pyannote.

    В 4.x pipeline() возвращает DiarizeOutput с несколькими полями, в 3.x и при
    legacy=True — сразу Annotation. Предпочитаем эксклюзивный вариант: в нём нет
    перекрывающейся речи, а значит слово не попадает сразу в двух говорящих.
    Проверяем по наличию itertracks, чтобы не тащить сюда импорт pyannote.core.
    """
    preferred = ("exclusive_speaker_diarization", "speaker_diarization")
    if not config.DIARIZATION_EXCLUSIVE:
        preferred = preferred[::-1]

    for name in preferred:
        candidate = getattr(output, name, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate, name

    if hasattr(output, "itertracks"):
        return output, "annotation"

    raise DiarizationError(
        f"pyannote вернул {type(output).__name__} — из него не достать разметку говорящих. "
        "Похоже, изменился формат ответа pyannote.audio; проверьте версию библиотеки."
    )


# Шаги pyannote в порядке исполнения и их доля в стадии диаризации.
# Веса приблизительные: сегментация и эмбеддинги съедают почти всё время.
_HOOK_STEPS = (
    ("segmentation", 0.40),
    ("speaker_counting", 0.05),
    ("embeddings", 0.45),
    ("discrete_diarization", 0.10),
)


def _make_hook(on_progress: Callable[[float], None]) -> Callable[..., None]:
    """Переводит вызовы pyannote-хука в долю 0..1 внутри стадии диаризации.

    Троттлинг здесь обязателен: на эмбеддингах хук зовётся на каждый батч, а это
    тысячи вызовов, каждый из которых в вебе оборачивается записью в SQLite.
    """
    bases: Dict[str, Tuple[float, float]] = {}
    offset = 0.0
    for name, weight in _HOOK_STEPS:
        bases[name] = (offset, weight)
        offset += weight

    state = {"value": 0.0, "emitted_at": 0.0}

    def hook(
        step_name: str,
        step_artifact: Any = None,
        file: Any = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ) -> None:
        base_weight = bases.get(step_name)
        if base_weight is None:
            return  # незнакомый шаг — прогресс не двигаем
        base, weight = base_weight
        inner = 1.0 if not total else min(1.0, (completed or 0) / total)
        # max — потому что сегментация приходит дважды: с прогрессом изнутри
        # Inference.slide и финальным вызовом уже без него.
        value = min(1.0, max(state["value"], base + weight * inner))

        now = time.monotonic()
        finished = inner >= 1.0
        if not finished and value - state["value"] < 0.01 and now - state["emitted_at"] < 1.0:
            return
        state["value"], state["emitted_at"] = value, now
        on_progress(value)

    return hook


def diarize(
    wav_path: Path,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[Turn]:
    """Возвращает интервалы речи по говорящим, отсортированные по времени."""
    pipeline = load()

    limits: Dict[str, int] = {}
    if num_speakers is not None:
        limits["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            limits["min_speakers"] = min_speakers
        if max_speakers is not None:
            limits["max_speakers"] = max_speakers

    kwargs: Dict[str, Any] = dict(limits)
    if on_progress is not None:
        kwargs["hook"] = _make_hook(on_progress)

    log.info(
        "Диаризация %s начата, ограничения: %s",
        wav_path.name,
        ", ".join(f"{key}={value}" for key, value in limits.items()) or "нет",
    )
    started = time.monotonic()
    try:
        output = pipeline(str(wav_path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(f"Ошибка диаризации: {exc}") from exc
    elapsed = time.monotonic() - started

    annotation, source = _extract_annotation(output)
    turns = [
        Turn(start=float(segment.start), end=float(segment.end), speaker=str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))

    speakers = {turn.speaker for turn in turns}
    log.info(
        "Диаризация завершена за %.1f с: %d интервалов, %d говорящих (%s), разметка %s",
        elapsed,
        len(turns),
        len(speakers),
        ", ".join(sorted(speakers)) or "—",
        source,
    )
    if not turns:
        log.warning("pyannote не нашёл ни одного интервала речи — весь текст уйдёт одному спикеру")
    elif len(speakers) == 1:
        log.warning(
            "pyannote нашёл одного говорящего. Если в записи их больше — "
            "задайте MIN_SPEAKERS=2 (или NUM_SPEAKERS) в .env"
        )
    return turns
