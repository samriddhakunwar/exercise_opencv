"""
exercises.py
------------
Single registry of every supported exercise.

Adding an exercise = writing its ExerciseBase subclass + one entry in
``EXERCISES``.  The dict order is the order the E key cycles through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import config
from exercise_base import ExerciseBase
from squat_counter import SquatCounter
from pushup_counter import PushUpCounter
from high_knee_counter import HighKneeCounter
from lunge_counter import LungeCounter
from curl_counter import CurlCounter
from jumping_jack_counter import JumpingJackCounter
from plank_counter import PlankCounter


@dataclass(frozen=True)
class ExerciseSpec:
    """Display metadata + factory for one exercise.

    Attributes:
        name:        Upper-case display name shown on the HUD.
        factory:     Zero-argument callable returning a fresh counter.
        angle_label: Label next to the live angle / metric.
        target_reps: Reps per set (per side for curls / lunges; holds for plank).
        target_sets: Total sets.
        bar_label:   Label above the right-side progress bar.
        best_label:  What history's best_metric means (progress.py).
        best_fmt:    Format for a best_metric value.
        last_fmt:    "Last time" text, formatted with a history row's columns.
    """
    name:        str
    factory:     Callable[[], ExerciseBase]
    angle_label: str
    target_reps: int
    target_sets: int
    bar_label:   str = "DEPTH"
    best_label:  str = "Deepest angle"
    best_fmt:    str = "{:.0f}°"
    last_fmt:    str = "{reps} reps, avg depth {avg_metric:.0f}°"


EXERCISES: Dict[str, ExerciseSpec] = {
    "squat": ExerciseSpec(
        "SQUAT", SquatCounter, "Knee",
        config.TARGET_REPS, config.TARGET_SETS,
    ),
    "pushup": ExerciseSpec(
        "PUSH-UP", PushUpCounter, "Elbow",
        config.TARGET_PUSHUPS, config.TARGET_PUSHUP_SETS,
        last_fmt="{reps} reps, avg elbow {avg_metric:.0f}°",
    ),
    "highknee": ExerciseSpec(
        "HIGH KNEES", HighKneeCounter, "Knee Ht",
        config.HIGH_KNEE_TARGET_REPS, config.HIGH_KNEE_TARGET_SETS,
        bar_label="HEIGHT",
        best_label="Highest knee lift", best_fmt="{:.0%}",
        last_fmt="{reps} reps, avg knee height {avg_metric:.0%}",
    ),
    "lunge": ExerciseSpec(
        "LUNGE", LungeCounter, "Front Knee",
        config.LUNGE_TARGET_REPS, config.LUNGE_TARGET_SETS,
    ),
    "curl": ExerciseSpec(
        "BICEP CURL", CurlCounter, "Elbow",
        config.CURL_TARGET_REPS, config.CURL_TARGET_SETS,
        bar_label="CURL",
        best_label="Tightest curl",
        last_fmt="{reps} reps, avg elbow {avg_metric:.0f}°",
    ),
    "jumpingjack": ExerciseSpec(
        "JUMPING JACKS", JumpingJackCounter, "Arms",
        config.JJ_TARGET_REPS, config.JJ_TARGET_SETS,
        bar_label="ARMS",
        best_label="Highest arm angle",
        last_fmt="{reps} reps, avg arm angle {avg_metric:.0f}°",
    ),
    "plank": ExerciseSpec(
        "PLANK", PlankCounter, "Body",
        1, config.PLANK_TARGET_SETS,
        bar_label="HOLD",
        best_label="Longest hold", best_fmt="{:.1f}s",
        last_fmt="{sets} sets, longest hold {best_metric:.0f}s",
    ),
}
