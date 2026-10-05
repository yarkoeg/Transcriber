"""Замер скорости на этой машине: что ставить в ASR_DEVICE и GIGAAM_MODEL.

    python scripts/benchmark.py запись.m4a

Печатает RTF (во сколько раз быстрее реального времени) и пиковую память.
Меряется тот же путь, что в приложении (asr.transcribe_with), а не
transcribe_longform самого GigaAM: тот на MPS включает fp16 и портит текст.
Совпадение текста с CPU видно по md5 — у корректной связки он тот же.
"""

import gc
import hashlib
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from backend import config  # noqa: E402
from backend.pipeline import audio  # noqa: E402

MODELS = ["v3_e2e_rnnt", "v3_e2e_ctc"]


def peak_memory_mb() -> float:
    # На macOS ru_maxrss отдаётся в байтах, в отличие от Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def devices() -> List[str]:
    import torch

    available = ["cpu"]
    if torch.backends.mps.is_available():
        available.append("mps")
    return available


def bench_asr(model_name: str, device: str, wav: Path, duration: float) -> Tuple[str, str]:
    from backend.pipeline import asr

    started = time.perf_counter()
    try:
        model = asr.load_model(model_name, device)
    except Exception as exc:  # noqa: BLE001
        return "ошибка загрузки", str(exc)[:120]
    load_time = time.perf_counter() - started

    started = time.perf_counter()
    try:
        segments = asr.transcribe_with(model, wav)
    except Exception as exc:  # noqa: BLE001
        return "ошибка инференса", str(exc)[:120]
    elapsed = time.perf_counter() - started
    digest = hashlib.md5(" ".join(seg.text for seg in segments).encode()).hexdigest()[:8]

    del model
    gc.collect()

    return (
        f"{elapsed:6.1f} с   RTF {duration / elapsed:5.1f}x",
        f"текст md5 {digest}, загрузка {load_time:.1f} с, пик RSS {peak_memory_mb():.0f} МБ",
    )


def bench_diarization(wav: Path, duration: float) -> Tuple[str, str]:
    if not config.HF_TOKEN:
        return "пропущено", "не задан HF_TOKEN"

    from backend.pipeline import diarize

    started = time.perf_counter()
    try:
        turns = diarize.diarize(wav)
    except Exception as exc:  # noqa: BLE001
        return "ошибка", str(exc)[:120]
    elapsed = time.perf_counter() - started
    speakers = len({turn.speaker for turn in turns})
    return (
        f"{elapsed:6.1f} с   RTF {duration / elapsed:5.1f}x",
        f"{speakers} говорящих, {len(turns)} интервалов, устройство {diarize.device()}",
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Файл не найден: {src}")
        return 1

    # Временная папка, а не data/: wav часового файла весит больше 100 МБ.
    with tempfile.TemporaryDirectory() as tmp:
        return run(src, audio.to_wav16k(src, Path(tmp) / "bench.wav"))


def run(src: Path, wav: Path) -> int:
    duration = audio.probe_duration(wav)
    print(f"Файл: {src.name}, {duration / 60:.1f} мин\n")

    print(f"{'конфигурация':<28}{'время':<26}примечание")
    print("-" * 96)

    for device in devices():
        for model_name in MODELS:
            label, note = bench_asr(model_name, device, wav, duration)
            print(f"{model_name + ' / ' + device:<28}{label:<26}{note}")

    label, note = bench_diarization(wav, duration)
    print(f"{'диаризация':<28}{label:<26}{note}")

    print(
        "\nВыберите самую быструю приемлемую по качеству строку и пропишите\n"
        "GIGAAM_MODEL и ASR_DEVICE в .env"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
