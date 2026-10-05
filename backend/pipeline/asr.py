"""Обёртка над GigaAM. Модель грузится лениво и живёт в памяти процесса.

Длинные записи распознаются собственным циклом, а не `transcribe_longform`:
тот на любом устройстве, кроме CPU, гонит энкодер под autocast(float16), и текст
расходится с CPU. Здесь энкодер на MPS считается в fp32 — текст побайтно
совпадает с CPU, а распознавание идёт в ~10 раз быстрее. Декодер остаётся
на CPU: он последовательный и занимает ~1% времени, ускорять там нечего.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

from .. import config
from .types import Segment, Word

log = logging.getLogger(__name__)

# Столько VAD-сегментов (по ~22 с) энкодер берёт за раз — как в transcribe_longform.
BATCH_SIZE = 16

_model = None
_lock = threading.Lock()


class ASRError(RuntimeError):
    pass


def _ensure_hf_token() -> None:
    """GigaAM тянет VAD (pyannote/segmentation-3.0) и читает токен из HF_TOKEN."""
    if config.HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", config.HF_TOKEN)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", config.HF_TOKEN)


def resolve_device(requested: str) -> str:
    """mps, если просили и он есть; иначе cpu — с предупреждением, а не падением."""
    if requested != "mps":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    log.warning("MPS недоступен — распознавание пойдёт на CPU, это в разы медленнее")
    return "cpu"


def load_model(name: str, device: str) -> Any:
    """Загружает GigaAM: декодер всегда на CPU, препроцессор и энкодер — на device."""
    _ensure_hf_token()
    device = resolve_device(device)
    if device == "mps":
        # Часть операций Conformer может не иметь реализации на MPS.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import gigaam

    try:
        # Грузим на CPU: так model.forward никогда не включит autocast, а на MPS
        # переезжают только тяжёлые части. fp16_encoder по умолчанию True —
        # на CPU/MPS это ломает инференс.
        model = gigaam.load_model(name, device="cpu", fp16_encoder=False, use_flash=False)
        model.preprocessor.to(device)
        model.encoder.to(device)
    except Exception as exc:  # noqa: BLE001 - хотим внятное сообщение наверх
        raise ASRError(f"Не удалось загрузить модель {name} на устройстве {device}: {exc}") from exc
    return model


def load() -> Any:
    """Возвращает модель из настроек, при необходимости скачав веса."""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is None:
            _model = load_model(config.GIGAAM_MODEL, config.ASR_DEVICE)
    return _model


def is_loaded() -> bool:
    return _model is not None


def device() -> str:
    """Устройство, на котором реально считается энкодер."""
    if _model is None:
        return resolve_device(config.ASR_DEVICE)
    return _encoder_device(_model).type


def _encoder_device(model: Any) -> Any:
    return next(model.encoder.parameters()).device


def _make_segment(text: str, start: float, end: float, raw_words: Any) -> Optional[Segment]:
    text = text.strip()
    if not text:
        return None

    # Таймкоды GigaAM отсчитываются от начала сегмента.
    words = [
        Word(text=w.text.strip(), start=round(w.start + start, 3), end=round(w.end + start, 3))
        for w in raw_words or []
        if w.text.strip()
    ]
    if not words:
        # Фолбэк: слов с таймкодами нет — равномерно раскладываем слова
        # по интервалу сегмента, чтобы диаризация всё же смогла к ним привязаться.
        tokens = text.split()
        step = (end - start) / len(tokens)
        words = [
            Word(text=token, start=start + i * step, end=start + (i + 1) * step)
            for i, token in enumerate(tokens)
        ]
    return Segment(text=text, start=start, end=end, words=words)


def transcribe_with(
    model: Any,
    wav_path: Path,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[Segment]:
    """Расшифровывает 16 kHz wav любой длины, с таймкодами слов.

    Повторяет transcribe_longform, но без autocast и с прогрессом по батчам.
    `.transcribe()` самого GigaAM для этого не годится — он работает до 25 с.
    """
    import torch
    from gigaam.utils import AudioDataset
    from gigaam.vad_utils import segment_audio_file

    device = _encoder_device(model)
    progress = on_progress or (lambda fraction: None)

    raw: List[tuple] = []
    try:
        # VAD тоже под inference_mode, как в самом GigaAM: его VAD-пайплайн глобальный,
        # и если переносить его между устройствами то в этом режиме, то вне его,
        # pyannote падает с «Inference tensors do not track version counter».
        with torch.inference_mode():
            # VAD — та же pyannote-сегментация, на MPS она в ~7 раз быстрее.
            chunks, boundaries = segment_audio_file(
                str(wav_path), config.SAMPLE_RATE, device=device
            )
            progress(0.15)
            if not chunks:
                return []

            dataset = AudioDataset(chunks, tokenizer=None)
            for first in range(0, len(dataset), BATCH_SIZE):
                batch = [dataset[i] for i in range(first, min(first + BATCH_SIZE, len(dataset)))]
                wav_pad, wav_lens = AudioDataset.collate(batch)
                features, feature_lens = model.preprocessor(wav_pad.to(device), wav_lens.to(device))
                encoded, encoded_lens = model.encoder(features, feature_lens)
                raw.extend(model._decode(encoded.cpu(), encoded_lens.cpu(), wav_lens, True))
                progress(0.15 + 0.85 * len(raw) / len(dataset))
    except Exception as exc:  # noqa: BLE001
        raise ASRError(f"Ошибка распознавания: {exc}") from exc

    segments = [
        segment
        for (text, words), (start, end) in zip(raw, boundaries)
        if (segment := _make_segment(text, start, end, words)) is not None
    ]
    for index, segment in enumerate(segments):
        for word in segment.words:
            word.segment = index
    return segments


def transcribe(
    wav_path: Path, on_progress: Optional[Callable[[float], None]] = None
) -> List[Segment]:
    return transcribe_with(load(), wav_path, on_progress)


def has_word_timestamps(segments: List[Segment]) -> bool:
    return any(seg.words for seg in segments)
