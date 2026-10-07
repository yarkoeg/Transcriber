# Сторонние компоненты и их лицензии

Код этого репозитория распространяется под MIT (см. [LICENSE](LICENSE)).

Веса моделей и библиотеки в репозиторий не входят: они скачиваются при установке и первом запуске и остаются под лицензиями своих авторов.

## Модели

| Модель | Для чего | Лицензия | Условия |
| --- | --- | --- | --- |
| [GigaAM](https://github.com/salute-developers/GigaAM) (`v3_e2e_rnnt`, `v3_e2e_ctc` и др.) | распознавание речи | MIT | — |
| [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | VAD внутри GigaAM | MIT | доступ после принятия условий на странице модели |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | разделение по говорящим | CC-BY-4.0 | доступ после принятия условий; при публикации результатов нужна атрибуция pyannote |

## Основные библиотеки

| Библиотека | Лицензия |
| --- | --- |
| [gigaam](https://github.com/salute-developers/GigaAM) | MIT |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT |
| [PyTorch](https://github.com/pytorch/pytorch), torchaudio | BSD-3-Clause |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT |
| [FFmpeg](https://ffmpeg.org/legal.html) | LGPL-2.1+ / GPL (зависит от сборки), ставится отдельно через `brew` |

Полный список транзитивных зависимостей и их лицензий: `.venv/bin/pip-licenses` (пакет `pip-licenses`) после установки.
