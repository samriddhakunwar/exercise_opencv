"""
clock.py
--------
Single time source for exercise counters.

Defaults to wall-clock ``time.time``.  When processing a video file the app
switches it to the video's own timeline (frame index / fps) so rep
durations, plank holds and pace are correct even when frames are processed
faster or slower than real time.
"""

from __future__ import annotations

import time
from typing import Callable

_source: Callable[[], float] = time.time


def now() -> float:
    """Current time in seconds from the active source."""
    return _source()


def use(source: Callable[[], float]) -> None:
    """Replace the time source (e.g. a video timestamp function)."""
    global _source
    _source = source


def use_wall_clock() -> None:
    """Restore the default wall-clock source."""
    use(time.time)
