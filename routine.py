"""
routine.py
----------
Workout routines: an ordered list of exercise and rest steps, defined in
``routines.json``::

    {"beginner": {"name": "Beginner", "steps": [
        {"exercise": "squat", "sets": 2, "reps": 10},
        {"rest": 30},
        {"exercise": "plank", "sets": 2, "seconds": 20}
    ]}}

``RoutineRunner`` walks the steps and collects one result per exercise
step; the app drives it (``python app.py --routine beginner``).
"""

from __future__ import annotations

import csv
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from exercises import EXERCISES


@dataclass(frozen=True)
class Step:
    """One routine step: an exercise block, or a rest when ``exercise`` is None."""
    exercise: Optional[str]   = None
    sets:     Optional[int]   = None
    reps:     Optional[int]   = None
    seconds:  Optional[float] = None
    rest:     float           = 0.0

    @property
    def is_rest(self) -> bool:
        return self.exercise is None


@dataclass(frozen=True)
class Routine:
    """A named sequence of steps."""
    key:   str
    name:  str
    steps: Tuple[Step, ...]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _positive(value: Any, what: str, where: str) -> Any:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{where}: '{what}' must be a positive number, got {value!r}")
    return value


def _parse_step(raw: Dict[str, Any], where: str) -> Step:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected an object")
    if "rest" in raw:
        if set(raw) != {"rest"}:
            raise ValueError(f"{where}: a rest step only takes 'rest'")
        return Step(rest=float(_positive(raw["rest"], "rest", where)))

    key = raw.get("exercise")
    if key not in EXERCISES:
        raise ValueError(f"{where}: unknown exercise {key!r} "
                         f"(choose from {', '.join(EXERCISES)})")
    unknown = set(raw) - {"exercise", "sets", "reps", "seconds"}
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
    timed = key == "plank"
    if timed and "reps" in raw:
        raise ValueError(f"{where}: plank uses 'seconds', not 'reps'")
    if not timed and "seconds" in raw:
        raise ValueError(f"{where}: {key} uses 'reps', not 'seconds'")
    return Step(
        exercise=key,
        sets=int(_positive(raw["sets"], "sets", where)) if "sets" in raw else None,
        reps=int(_positive(raw["reps"], "reps", where)) if "reps" in raw else None,
        seconds=float(_positive(raw["seconds"], "seconds", where)) if "seconds" in raw else None,
    )


def load_routines(path: Optional[str] = None) -> Dict[str, Routine]:
    """Load and validate every routine in *path* (default ``config.ROUTINES_FILE``).

    Raises:
        ValueError: with the routine / step that is wrong.
    """
    data = json.loads(Path(path or config.ROUTINES_FILE).read_text(encoding="utf-8"))
    routines = {}
    for key, body in data.items():
        steps = tuple(
            _parse_step(raw, f"routine '{key}' step {i}")
            for i, raw in enumerate(body.get("steps", []), start=1)
        )
        if not any(not s.is_rest for s in steps):
            raise ValueError(f"routine '{key}' has no exercise steps")
        routines[key] = Routine(key, body.get("name", key), steps)
    return routines


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class RoutineRunner:
    """Tracks progress through a routine and the result of each exercise."""

    def __init__(self, routine: Routine) -> None:
        self.routine = routine
        self.index   = 0
        self.results: List[Dict[str, Any]] = []
        self.started = datetime.datetime.now()

    @property
    def step(self) -> Optional[Step]:
        """Current step (None when the routine is finished)."""
        steps = self.routine.steps
        return steps[self.index] if self.index < len(steps) else None

    @property
    def done(self) -> bool:
        return self.step is None

    def advance(self) -> Optional[Step]:
        """Move to the next step and return it (None = routine finished)."""
        self.index += 1
        return self.step

    def next_exercise(self) -> Optional[str]:
        """Key of the next exercise step after the current one."""
        for step in self.routine.steps[self.index + 1:]:
            if not step.is_rest:
                return step.exercise
        return None

    def progress_text(self) -> str:
        """e.g. "FULL BODY 2/5" — position among the exercise steps."""
        exercise_steps = [i for i, s in enumerate(self.routine.steps) if not s.is_rest]
        done = sum(1 for i in exercise_steps if i < self.index)
        current = min(done + 1, len(exercise_steps))
        return f"{self.routine.name.upper()} {current}/{len(exercise_steps)}"

    def record(self, exercise: str, counter, duration: float) -> None:
        """Store the outcome of an exercise step."""
        stats = counter.stats
        self.results.append({
            "exercise":    exercise,
            "reps":        stats.total_reps,
            "sets":        stats.sets_completed,
            "target_sets": counter.TARGET_SETS,
            "duration":    duration,
            "calories":    stats.calories,
            "avg":         counter.avg_metric(),
            "best":        counter.best_metric(),
        })

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def summary_lines(self) -> List[Tuple[str, str]]:
        """(label, value) pairs for the summary screen / console."""
        lines = []
        for r in self.results:
            spec = EXERCISES[r["exercise"]]
            if r["exercise"] == "plank":
                best = "-" if r["best"] is None else spec.best_fmt.format(r["best"])
                value = f"{r['sets']}/{r['target_sets']} sets, best {best}"
            else:
                value = f"{r['reps']} reps, {r['sets']}/{r['target_sets']} sets"
            lines.append((spec.name.title(), value))
        total_time = sum(r["duration"] for r in self.results)
        lines.append(("Total Time", f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"))
        lines.append(("Calories", f"{sum(r['calories'] for r in self.results):.1f} kcal"))
        return lines

    def write_summary(self, out_dir: str = config.STATS_DIR) -> Path:
        """Write one CSV covering every exercise step of the routine."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = self.started.strftime("%Y%m%d_%H%M%S")
        path = out / f"routine_{self.routine.key}_{ts}.csv"
        rows: List[List[Any]] = [
            ["Routine", self.routine.name],
            ["Date", self.started.strftime("%Y-%m-%d %H:%M")],
            ["Completed", "yes" if self.done else "no (stopped early)"],
            [],
            ["Step", "Exercise", "Reps", "Sets", "Target Sets", "Duration (s)",
             "Calories", "Avg Metric", "Best Metric"],
        ]
        for i, r in enumerate(self.results, start=1):
            rows.append([
                i, EXERCISES[r["exercise"]].name, r["reps"], r["sets"], r["target_sets"],
                f"{r['duration']:.1f}", f"{r['calories']:.1f}",
                "" if r["avg"] is None else f"{r['avg']:.2f}",
                "" if r["best"] is None else f"{r['best']:.2f}",
            ])
        rows.append(["Total", "", sum(r["reps"] for r in self.results),
                     sum(r["sets"] for r in self.results), "",
                     f"{sum(r['duration'] for r in self.results):.1f}",
                     f"{sum(r['calories'] for r in self.results):.1f}", "", ""])
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        return path
