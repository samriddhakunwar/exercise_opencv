"""
history.py
----------
Workout history: one row per session in a SQLite database
(``config.HISTORY_DB``, default ``data/history.db``).

The per-session CSVs in ``workouts/`` are kept as detailed logs; this table
is the long-term record used for progress charts, "last time" text and
personal-record detection.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import List, Optional

import config
from exercises import EXERCISES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,      -- ISO timestamp (local time)
    exercise    TEXT    NOT NULL,      -- key of exercises.EXERCISES
    reps        INTEGER NOT NULL,
    sets        INTEGER NOT NULL,
    duration    REAL    NOT NULL,      -- seconds
    calories    REAL    NOT NULL,
    avg_metric  REAL,                  -- avg angle / depth / knee height
    best_metric REAL,                  -- deepest angle, longest plank, …
    record      REAL    NOT NULL       -- value compared for "New record!"
)
"""


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open (creating if needed) the history database."""
    path = Path(db_path or config.HISTORY_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def add_session(
    exercise: str,
    counter,
    duration: float,
    db_path: Optional[str] = None,
    date: Optional[datetime.datetime] = None,
) -> None:
    """Append one row summarising *counter*'s session.

    Args:
        exercise: Exercise key.
        counter:  The ExerciseBase whose session just ended.
        duration: Session length in seconds.
        db_path:  Override the database path (tests).
        date:     Override the timestamp (tests).
    """
    stats = counter.stats
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (date, exercise, reps, sets, duration, calories,"
            " avg_metric, best_metric, record) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (date or datetime.datetime.now()).isoformat(timespec="seconds"),
                exercise,
                stats.total_reps,
                stats.sets_completed,
                round(duration, 1),
                round(stats.calories, 2),
                counter.avg_metric(),
                counter.best_metric(),
                counter.record_value(),
            ),
        )
    conn.close()


def last_session(exercise: str, db_path: Optional[str] = None) -> Optional[sqlite3.Row]:
    """Most recent session row for *exercise* (None if there isn't one)."""
    conn = connect(db_path)
    row = conn.execute(
        "SELECT * FROM sessions WHERE exercise = ? ORDER BY date DESC, id DESC LIMIT 1",
        (exercise,),
    ).fetchone()
    conn.close()
    return row


def personal_record(exercise: str, db_path: Optional[str] = None) -> Optional[float]:
    """Best ``record`` value so far for *exercise* (None if no history)."""
    conn = connect(db_path)
    value = conn.execute(
        "SELECT MAX(record) FROM sessions WHERE exercise = ?", (exercise,)
    ).fetchone()[0]
    conn.close()
    return value


def all_sessions(db_path: Optional[str] = None) -> List[sqlite3.Row]:
    """Every session, oldest first."""
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM sessions ORDER BY date, id").fetchall()
    conn.close()
    return rows


def last_time_text(exercise: str, db_path: Optional[str] = None) -> str:
    """'Last time: …' line for the countdown screen ('' if no history)."""
    row = last_session(exercise, db_path)
    if row is None:
        return ""
    try:
        detail = EXERCISES[exercise].last_fmt.format(**dict(row))
    except (TypeError, ValueError, KeyError):   # NULL metric in the row
        detail = f"{row['reps']} reps"
    return f"Last time: {detail}"
