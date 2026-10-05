"""Хранилище задач на SQLite. Нужно, чтобы результаты переживали перезапуск."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    status          TEXT NOT NULL,
    stage           TEXT,
    progress        REAL NOT NULL DEFAULT 0,
    duration        REAL,
    error           TEXT,
    created_at      TEXT NOT NULL,
    finished_at     TEXT,
    txt_path        TEXT,
    json_path       TEXT,
    wav_path        TEXT,
    speaker_count   INTEGER,
    utterance_count INTEGER,
    meta            TEXT,
    started_at      TEXT,
    processing_seconds REAL,
    diarize         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at DESC);
"""

# Колонки, добавленные после первого релиза. CREATE TABLE IF NOT EXISTS их
# не добавит в уже созданную базу, а ADD COLUMN IF NOT EXISTS в SQLite нет —
# поэтому сверяемся с PRAGMA. Полноценный механизм миграций пока не нужен.
MIGRATIONS = (
    ("started_at", "TEXT"),
    ("processing_seconds", "REAL"),
    # NULL — «как в .env»: так ведут себя задачи, созданные до переключателя.
    ("diarize", "INTEGER"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    with _lock:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for column, declaration in MIGRATIONS:
            if column not in existing:
                # Имена и типы — литералы из MIGRATIONS, не пользовательский ввод.
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")


def reset_interrupted() -> int:
    """Задачи, оставшиеся в running после падения процесса, помечаем упавшими."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
            "WHERE status IN ('running', 'queued')",
            ("Обработка прервана перезапуском сервера", now()),
        )
        return cursor.rowcount


def create(job_id: str, filename: str, source_path: str, diarize: Optional[bool] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, source_path, status, stage, progress, created_at, diarize) "
            "VALUES (?, ?, ?, 'queued', NULL, 0, ?, ?)",
            (job_id, filename, source_path, now(), None if diarize is None else int(diarize)),
        )


def update(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    if "meta" in fields and not isinstance(fields["meta"], (str, type(None))):
        fields["meta"] = json.dumps(fields["meta"], ensure_ascii=False)
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            (*fields.values(), job_id),
        )


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    job = dict(row)
    if job.get("meta"):
        try:
            job["meta"] = json.loads(job["meta"])
        except json.JSONDecodeError:
            job["meta"] = None
    return job


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_all(limit: int = 200) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def delete(job_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0
