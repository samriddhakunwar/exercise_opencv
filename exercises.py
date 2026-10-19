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
        bar_label:   Label above the right-side progress bar.
        best_label:  What history's best_metric means (progress.py).
        best_fmt:    Format for a best_metric value.
        last_fmt:    "Last time" text, formatted with a history row's columns.
    """
    name:        str
    factory:     Callable[[], ExerciseBase]
    angle_label: str
    bar_label:   str = "DEPTH"
    best_label:  str = "Deepest angle"
    best_fmt:    str = "{:.0f}°"
    last_fmt:    str = "{reps} reps, avg depth {avg_metric:.0f}°"


EXERCISES: Dict[str, ExerciseSpec] = {
    "squat": ExerciseSpec(
        "SQUAT", SquatCounter, "Knee",
    ),
    "pushup": ExerciseSpec(
        "PUSH-UP", PushUpCounter, "Elbow",
        last_fmt="{reps} reps, avg elbow {avg_metric:.0f}°",
    ),
    "highknee": ExerciseSpec(
        "HIGH KNEES", HighKneeCounter, "Knee Ht",
        bar_label="HEIGHT",
        best_label="Highest knee lift", best_fmt="{:.0%}",
        last_fmt="{reps} reps, avg knee height {avg_metric:.0%}",
    ),
    "lunge": ExerciseSpec(
        "LUNGE", LungeCounter, "Front Knee",
    ),
    "curl": ExerciseSpec(
        "BICEP CURL", CurlCounter, "Elbow",
        bar_label="CURL",
        best_label="Tightest curl",
        last_fmt="{reps} reps, avg elbow {avg_metric:.0f}°",
    ),
    "jumpingjack": ExerciseSpec(
        "JUMPING JACKS", JumpingJackCounter, "Arms",
        bar_label="ARMS",
        best_label="Highest arm angle",
        last_fmt="{reps} reps, avg arm angle {avg_metric:.0f}°",
    ),
    "plank": ExerciseSpec(
        "PLANK", PlankCounter, "Body",
        bar_label="HOLD",
        best_label="Longest hold", best_fmt="{:.1f}s",
        last_fmt="{sets} sets, longest hold {best_metric:.0f}s",
    ),
}
