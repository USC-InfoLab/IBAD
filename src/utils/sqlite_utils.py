"""Shared SQLite connection helper."""

import sqlite3
from pathlib import Path


def open_sqlite(path, setup_sqls=(), *, timeout: int = 30, wal: bool = False):
    """Open (or create) a SQLite database and run any setup DDL statements.

    Creates parent directories automatically. WAL mode is set before DDL
    so concurrent readers are not blocked during schema initialisation.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    for sql in setup_sqls:
        conn.execute(sql)
    if setup_sqls:
        conn.commit()
    return conn
