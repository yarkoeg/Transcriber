"""Окно приложения для macOS: сервер в фоновом потоке и интерфейс во встроенном WebKit.

Запускается из Транскрибация.app (собирает scripts/build_mac_app.py) или напрямую:
    .venv/bin/python -m backend.desktop

Закрытие окна останавливает сервер. Если на порту уже работает сервер
Транскрибации (запущенный руками или ярлыком), окно подключается к нему и
второй не поднимает: очередь живёт в памяти процесса, а новый процесс при старте
пометил бы чужие задачи в работе упавшими (db.reset_interrupted).
"""

import json
import logging
import os
import socket
import threading
import time
import urllib.request
from html import escape
from typing import Optional

import uvicorn
import webview

from . import db

log = logging.getLogger("transcriber.desktop")

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
STARTUP_TIMEOUT = 120  # с: первый импорт torch после перезагрузки бывает долгим

LOCALIZATION = {
    "global.quitConfirmation": (
        "Идёт обработка записи. Если закрыть приложение, она прервётся, "
        "и запись придётся загрузить заново."
    ),
    "global.ok": "ОК",
    "global.quit": "Закрыть",
    "global.cancel": "Отмена",
    "global.saveFile": "Сохранить файл",
    "cocoa.menu.about": "О программе",
    "cocoa.menu.services": "Службы",
    "cocoa.menu.view": "Вид",
    "cocoa.menu.edit": "Правка",
    "cocoa.menu.hide": "Скрыть",
    "cocoa.menu.hideOthers": "Скрыть остальные",
    "cocoa.menu.showAll": "Показать все",
    "cocoa.menu.quit": "Завершить",
    "cocoa.menu.fullscreen": "Во весь экран",
    "cocoa.menu.cut": "Вырезать",
    "cocoa.menu.copy": "Скопировать",
    "cocoa.menu.paste": "Вставить",
    "cocoa.menu.selectAll": "Выбрать всё",
}

PAGE = """<!doctype html><meta charset="utf-8">
<style>
  body {{ font: 15px -apple-system, system-ui, sans-serif; margin: 0; height: 100vh;
         display: grid; place-items: center; text-align: center;
         color: #14171c; background: #ffffff; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #e8eaee; background: #14171c; }} }}
  p {{ max-width: 34em; line-height: 1.5; margin: 0 24px; }}
  .muted {{ color: #6b7280; margin-top: 8px; }}
</style>
<div><p>{title}</p><p class="muted">{hint}</p></div>
"""


def _page(title: str, hint: str = "") -> str:
    return PAGE.format(title=escape(title), hint=escape(hint))


def _log_hint() -> str:
    path = os.environ.get("TRANSCRIBER_LOG")
    return f"Подробности — в журнале {path}" if path else "Подробности — в выводе терминала."


def _our_server_answers() -> bool:
    """На порту отвечает именно Транскрибация, а не что-то постороннее."""
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=1) as resp:
            return "asr_device" in json.load(resp)
    except Exception:  # noqa: BLE001 - любой отказ значит «не наш»
        return False


def _port_busy() -> bool:
    with socket.socket() as sock:
        return sock.connect_ex((HOST, PORT)) == 0


def _has_active_jobs() -> bool:
    return any(job["status"] in ("queued", "running") for job in db.list_all())


class Server:
    """uvicorn в фоновом потоке, чтобы главный поток достался AppKit."""

    def __init__(self) -> None:
        config = uvicorn.Config("backend.app:app", host=HOST, port=PORT)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="uvicorn", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_started(self) -> bool:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._server.started:
                return True
            if not self._thread.is_alive():
                return False
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        self._server.should_exit = True
        # Идущую задачу дождаться нельзя — она может длиться минуты. Воркер
        # в daemon-потоке умрёт вместе с процессом, а при следующем старте
        # db.reset_interrupted пометит задачу прерванной.
        self._thread.join(timeout=10)


def _connect(window: webview.Window, server: Optional[Server]) -> None:
    """Выполняется в отдельном потоке, когда окно уже на экране."""
    if server is None:
        window.load_url(URL)
        return

    server.start()
    if server.wait_started():
        window.load_url(URL)
    else:
        window.load_html(_page("Сервер не запустился.", _log_hint()))


def main() -> None:
    server: Optional[Server] = None
    first_page = _page("Запускается…")

    if _our_server_answers():
        log.info("Сервер уже работает на %s — подключаюсь к нему", URL)
    elif _port_busy():
        first_page = _page(
            f"Порт {PORT} занят другой программой.",
            "Закройте её и откройте Транскрибацию снова.",
        )
    else:
        server = Server()

    window = webview.create_window(
        "Транскрибация",
        html=first_page,
        width=1200,
        height=820,
        min_size=(760, 520),
    )

    def on_closing() -> None:
        # Спрашиваем, только если закрытие действительно что-то прервёт.
        # pywebview смотрит на confirm_close сразу после этого обработчика.
        window.confirm_close = server is not None and _has_active_jobs()

    window.events.closing += on_closing

    webview.settings["ALLOW_DOWNLOADS"] = True
    # Внешние ссылки — в обычный браузер, а не поверх интерфейса.
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True

    try:
        webview.start(
            _connect,
            (window, server),
            # Иначе localStorage (выбор «с разделением / сплошной текст»)
            # стирается при каждом запуске.
            private_mode=False,
            localization=LOCALIZATION,
        )
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
