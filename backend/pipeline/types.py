"""Общие структуры данных пайплайна."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    # Номер VAD-сегмента, из которого пришло слово. Границы сегментов —
    # это найденные моделью паузы, по ним удобно разбивать текст на абзацы.
    segment: int = 0


@dataclass
class Segment:
    """Сегмент, как его отдаёт GigaAM (нарезка по VAD)."""

    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)


@dataclass
class Turn:
    """Интервал речи одного говорящего, как его отдаёт pyannote."""

    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    """Реплика: подряд идущие слова одного говорящего."""

    speaker: str
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
