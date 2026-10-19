"""
calibration.py
--------------
Per-user threshold calibration.

Flow (driven by the app, one metric value per frame)::

    STAND ─(CALIBRATION_STAND_SECONDS of body in view)─► REPS
    REPS  ─(CALIBRATION_REPS full reps, or timeout)────► DONE / FAILED

The metric is the exercise's primary value (knee angle, elbow angle, knee
lift …).  The resting value is the median of the STAND phase; the extreme is
the furthest value reached during REPS.

Each default threshold was designed for a nominal (rest, extreme) range.
Calibration maps it linearly onto the user's measured range and clamps the
result to ``default ± max_shift``.  The mapping is monotonic and so is the
clamp, so threshold ordering (depth < exit < standing …) is preserved and a
bad calibration can't make counting impossible.  A measured range smaller
than ``min_range`` is rejected outright.

Saved per exercise in ``config.CALIBRATION_FILE`` (data/calibration.json)
and applied to ``config`` on startup; missing file → config.py defaults.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


@dataclass(frozen=True)
class CalibrationSpec:
    """How to calibrate one exercise.

    Attributes:
        instruction: What to do during the STAND phase.
        nominal:     (rest, extreme) metric range the defaults assume.
        thresholds:  config names rescaled onto the user's range.
        max_shift:   Max change from each default (clamp).
        min_range:   Smallest acceptable |extreme - rest|.
    """
    instruction: str
    nominal:     Tuple[float, float]
    thresholds:  Tuple[str, ...]
    max_shift:   float
    min_range:   float


SPECS: Dict[str, CalibrationSpec] = {
    "squat": CalibrationSpec(
        "Stand up straight", (175.0, 80.0),
        ("STANDING_ANGLE_MIN", "FULL_EXT_ANGLE", "BAD_DEPTH_ANGLE", "SQUAT_DEPTH_ANGLE"),
        max_shift=20.0, min_range=50.0),
    "pushup": CalibrationSpec(
        "Hold the top of a push-up", (170.0, 80.0),
        ("PUSHUP_TOP_ANGLE", "PUSHUP_FULL_EXT_ANGLE", "PUSHUP_LOWER_LIMIT",
         "PUSHUP_BOTTOM_ANGLE"),
        max_shift=20.0, min_range=40.0),
    "lunge": CalibrationSpec(
        "Stand tall in a split stance", (175.0, 85.0),
        ("LUNGE_STANDING_ANGLE", "LUNGE_BOTTOM_EXIT_ANGLE", "LUNGE_DEPTH_ANGLE"),
        max_shift=20.0, min_range=40.0),
    "curl": CalibrationSpec(
        "Let your arms hang straight", (165.0, 35.0),
        ("CURL_EXTENDED_ANGLE", "CURL_TOP_EXIT_ANGLE", "CURL_TOP_ANGLE"),
        max_shift=20.0, min_range=60.0),
    "highknee": CalibrationSpec(
        "Stand still", (-0.5, 0.9),
        ("HIGH_KNEE_MIN_HEIGHT_RATIO",),
        max_shift=0.65, min_range=0.3),
}

# Snapshot of config.py defaults, taken before any calibration is applied
DEFAULTS: Dict[str, float] = {
    name: getattr(config, name) for spec in SPECS.values() for name in spec.thresholds
}


# ---------------------------------------------------------------------------
# Threshold maths
# ---------------------------------------------------------------------------

def _clamp(name: str, value: float, max_shift: float) -> float:
    default = DEFAULTS[name]
    return max(default - max_shift, min(default + max_shift, value))


def derive_thresholds(exercise: str, rest: float, extreme: float) -> Optional[Dict[str, float]]:
    """Map the default thresholds onto the user's (rest, extreme) range.

    Returns:
        {config name: value}, or None if the range is too small.
    """
    spec = SPECS[exercise]
    if abs(extreme - rest) < spec.min_range:
        return None
    nom_rest, nom_ext = spec.nominal
    scale = (extreme - rest) / (nom_ext - nom_rest)
    return {
        name: round(_clamp(name, rest + (DEFAULTS[name] - nom_rest) * scale, spec.max_shift), 3)
        for name in spec.thresholds
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load(path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Read saved calibrations ({} if the file is missing or unreadable)."""
    try:
        return json.loads(Path(path or config.CALIBRATION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(exercise: str, thresholds: Dict[str, float], path: Optional[str] = None) -> None:
    """Store *exercise*'s thresholds, keeping other exercises' entries."""
    file = Path(path or config.CALIBRATION_FILE)
    data = load(str(file))
    data[exercise] = thresholds
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply(data: Dict[str, Dict[str, float]]) -> List[str]:
    """Set calibrated values on ``config``; returns the exercises applied.

    Only known threshold names are accepted and values are re-clamped, so a
    hand-edited file can't push thresholds out of their safe range.
    """
    applied = []
    for exercise, values in data.items():
        spec = SPECS.get(exercise)
        if spec is None or not isinstance(values, dict):
            continue
        for name, value in values.items():
            if name in spec.thresholds and isinstance(value, (int, float)):
                setattr(config, name, _clamp(name, float(value), spec.max_shift))
        applied.append(exercise)
    return applied


# ---------------------------------------------------------------------------
# Guided calibration state machine
# ---------------------------------------------------------------------------

class Calibrator:
    """Collects metric samples and derives thresholds for one exercise.

    Call ``update(metric, now)`` every frame (metric None = no body found).
    Read ``message`` for the HUD, drain ``cues`` for audio, and check
    ``stage`` for "done" / "failed"; ``result`` holds the thresholds.
    """

    def __init__(self, exercise: str) -> None:
        self.exercise = exercise
        self.spec     = SPECS[exercise]
        self.stage    = "stand"
        self.result: Optional[Dict[str, float]] = None
        self.cues: List[str] = [f"{self.spec.instruction} for "
                                f"{config.CALIBRATION_STAND_SECONDS:.0f} seconds"]
        self.message  = self.spec.instruction.upper()
        self._rest_samples: List[float] = []
        self._held      = 0.0              # Seconds of STAND with a body in view
        self._last_now: Optional[float] = None
        self._reps_start = 0.0
        self._rest      = 0.0
        self._extreme   = 0.0
        self._reps      = 0
        self._away      = False

    @property
    def finished(self) -> bool:
        """True once calibration succeeded or failed."""
        return self.stage in ("done", "failed")

    def update(self, metric: Optional[float], now: float) -> None:
        """Advance with this frame's metric value."""
        dt = 0.0 if self._last_now is None else now - self._last_now
        self._last_now = now
        if self.finished:
            return
        if metric is None:
            self.message = "STEP INTO FRAME"
            return

        if self.stage == "stand":
            self._rest_samples.append(metric)
            self._held += dt
            left = config.CALIBRATION_STAND_SECONDS - self._held
            self.message = f"{self.spec.instruction.upper()}  {max(0, left):.0f}s"
            if left <= 0:
                self._rest    = statistics.median(self._rest_samples)
                self._extreme = self._rest
                self._reps_start = now
                self.stage    = "reps"
                self.cues.append(f"Now do {config.CALIBRATION_REPS} slow, full reps")
            return

        # ---- REPS ---------------------------------------------------------
        direction = 1 if self.spec.nominal[1] > self.spec.nominal[0] else -1
        if direction * (metric - self._extreme) > 0:
            self._extreme = metric
        excursion = abs(metric - self._rest)
        if not self._away and excursion > 0.8 * self.spec.min_range:
            self._away = True
        elif self._away and excursion < 0.25 * self.spec.min_range:
            self._away = False
            self._reps += 1
            self.cues.append(str(self._reps))
        self.message = f"SLOW FULL REPS  {self._reps} / {config.CALIBRATION_REPS}"

        if (self._reps >= config.CALIBRATION_REPS
                or now - self._reps_start > config.CALIBRATION_TIMEOUT):
            self._finish()

    def _finish(self) -> None:
        self.result = derive_thresholds(self.exercise, self._rest, self._extreme)
        if self.result is None:
            self.stage   = "failed"
            self.message = "CALIBRATION FAILED - USING DEFAULTS"
            self.cues.append("Calibration failed. Using default settings.")
        else:
            self.stage   = "done"
            self.message = "CALIBRATION SAVED!"
            self.cues.append("Calibration saved")
