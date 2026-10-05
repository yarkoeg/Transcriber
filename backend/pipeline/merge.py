"""Склейка результата ASR и диаризации в реплики по говорящим.

GigaAM и pyannote работают независимо и режут аудио по-разному, поэтому
единственная общая система координат — время. Привязываем каждое слово
к говорящему по его середине, затем собираем подряд идущие слова в реплики.
"""

from bisect import bisect_right
from typing import Dict, List, Optional

from ..config import SPEAKER_SNAP_WINDOW, UTTERANCE_GAP
from .types import Segment, Turn, Utterance, Word

SINGLE_SPEAKER = "SPEAKER_00"


def _find_speaker(turns: List[Turn], starts: List[float], t: float) -> Optional[str]:
    """Говорящий, чей интервал накрывает момент t, иначе ближайший в пределах окна."""
    if not turns:
        return None

    # Кандидаты вокруг точки: последний turn, начавшийся не позже t, и следующий.
    idx = bisect_right(starts, t) - 1

    for i in (idx, idx + 1):
        if 0 <= i < len(turns) and turns[i].start <= t <= turns[i].end:
            return turns[i].speaker

    best_speaker, best_distance = None, float("inf")
    for i in (idx - 1, idx, idx + 1, idx + 2):
        if not 0 <= i < len(turns):
            continue
        turn = turns[i]
        distance = 0.0 if turn.start <= t <= turn.end else min(abs(t - turn.start), abs(t - turn.end))
        if distance < best_distance:
            best_speaker, best_distance = turn.speaker, distance

    if best_speaker is not None and best_distance <= SPEAKER_SNAP_WINDOW:
        return best_speaker
    return None


def assign_speakers(segments: List[Segment], turns: List[Turn]) -> List[Word]:
    """Проставляет каждому слову говорящего. Возвращает плоский список слов."""
    starts = [turn.start for turn in turns]
    words: List[Word] = []
    previous_speaker: Optional[str] = None

    for segment in segments:
        for word in segment.words:
            midpoint = (word.start + word.end) / 2
            speaker = _find_speaker(turns, starts, midpoint)
            if speaker is None:
                # Слово попало в паузу диаризации — наследуем предыдущего
                # говорящего, это почти всегда продолжение той же реплики.
                speaker = previous_speaker or (turns[0].speaker if turns else SINGLE_SPEAKER)
            word.speaker = speaker
            previous_speaker = speaker
            words.append(word)

    return words


def group_utterances(words: List[Word]) -> List[Utterance]:
    """Собирает подряд идущие слова одного говорящего в реплики.

    Реплика рвётся при смене говорящего, при длинной паузе и на границе
    VAD-сегмента. Последнее важно для читаемости: без него монолог на
    несколько минут превращается в один абзац-простыню.
    """
    utterances: List[Utterance] = []
    current: List[Word] = []

    def flush() -> None:
        if not current:
            return
        utterances.append(
            Utterance(
                speaker=current[0].speaker or SINGLE_SPEAKER,
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(word.text for word in current),
                words=list(current),
            )
        )
        current.clear()

    for word in words:
        if current:
            previous = current[-1]
            changed_speaker = word.speaker != previous.speaker
            long_pause = word.start - previous.end > UTTERANCE_GAP
            new_segment = word.segment != previous.segment
            if changed_speaker or long_pause or new_segment:
                flush()
        current.append(word)

    flush()
    return utterances


def label_speakers(utterances: List[Utterance]) -> Dict[str, str]:
    """SPEAKER_xx -> «Спикер N» в порядке первого появления в записи.

    Нумеровать по ID из pyannote нельзя: они не соответствуют порядку речи,
    и «Спикер 2» может заговорить первым.
    """
    mapping: Dict[str, str] = {}
    for utterance in utterances:
        if utterance.speaker not in mapping:
            mapping[utterance.speaker] = f"Спикер {len(mapping) + 1}"
    return mapping


def merge(segments: List[Segment], turns: List[Turn]) -> tuple[List[Utterance], Dict[str, str]]:
    """Полный проход: слова -> говорящие -> реплики -> человекочитаемые метки."""
    if not turns:
        # Диаризация выключена или ничего не нашла — весь текст одному говорящему.
        for segment in segments:
            for word in segment.words:
                word.speaker = SINGLE_SPEAKER

        words = [word for segment in segments for word in segment.words]
    else:
        words = assign_speakers(segments, turns)

    utterances = group_utterances(words)
    return utterances, label_speakers(utterances)
