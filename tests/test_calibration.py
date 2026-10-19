"""Calibration maths, persistence and the guided Calibrator FSM."""

import json

import pytest

import calibration
import config
from calibration import Calibrator, derive_thresholds
from joint_factory import hold, ramp


@pytest.fixture(autouse=True)
def restore_config(monkeypatch):
    """apply() mutates config — let monkeypatch restore every threshold."""
    for name in calibration.DEFAULTS:
        monkeypatch.setattr(config, name, getattr(config, name))


@pytest.mark.parametrize("exercise", list(calibration.SPECS))
def test_nominal_range_reproduces_defaults(exercise):
    spec = calibration.SPECS[exercise]
    result = derive_thresholds(exercise, *spec.nominal)
    assert result == {n: pytest.approx(calibration.DEFAULTS[n]) for n in spec.thresholds}


def test_shallow_user_gets_easier_depth():
    t = derive_thresholds("squat", rest=170, extreme=100)
    assert t["SQUAT_DEPTH_ANGLE"] > config.SQUAT_DEPTH_ANGLE
    assert t["SQUAT_DEPTH_ANGLE"] >= 100                # user's deepest angle still reaches it
    assert t["SQUAT_DEPTH_ANGLE"] < t["BAD_DEPTH_ANGLE"] < t["FULL_EXT_ANGLE"] < t["STANDING_ANGLE_MIN"] + 1e-9


def test_extreme_input_is_clamped():
    t = derive_thresholds("squat", rest=180, extreme=10)
    for name, value in t.items():
        assert abs(value - calibration.DEFAULTS[name]) <= 20 + 1e-9


def test_too_small_range_is_rejected():
    assert derive_thresholds("squat", rest=170, extreme=140) is None
    assert derive_thresholds("highknee", rest=-0.4, extreme=-0.2) is None


@pytest.mark.parametrize("rest, extreme", [(150, 60), (178, 95), (160, 30), (180, 105)])
def test_ordering_preserved_for_any_range(rest, extreme):
    t = derive_thresholds("pushup", rest, extreme)
    assert (t["PUSHUP_BOTTOM_ANGLE"] < t["PUSHUP_LOWER_LIMIT"]
            < t["PUSHUP_FULL_EXT_ANGLE"] < t["PUSHUP_TOP_ANGLE"])


def test_high_knee_threshold_is_reachable():
    t = derive_thresholds("highknee", rest=-0.4, extreme=0.3)
    assert -0.4 < t["HIGH_KNEE_MIN_HEIGHT_RATIO"] < 0.3


def test_save_load_apply_roundtrip(tmp_path):
    path = str(tmp_path / "cal.json")
    calibration.save("squat", {"SQUAT_DEPTH_ANGLE": 108.0}, path)
    calibration.save("curl", {"CURL_TOP_ANGLE": 60.0}, path)
    data = calibration.load(path)
    assert set(data) == {"squat", "curl"}
    assert calibration.apply(data) == ["squat", "curl"]
    assert config.SQUAT_DEPTH_ANGLE == 108.0 and config.CURL_TOP_ANGLE == 60.0


def test_apply_reclamps_and_ignores_unknown_names(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({
        "squat": {"SQUAT_DEPTH_ANGLE": 5.0, "CAMERA_INDEX": 9},
        "plank": {"PLANK_TARGET_SECONDS": 1},
    }))
    calibration.apply(calibration.load(str(path)))
    assert config.SQUAT_DEPTH_ANGLE == calibration.DEFAULTS["SQUAT_DEPTH_ANGLE"] - 20
    assert config.CAMERA_INDEX == 0
    assert config.PLANK_TARGET_SECONDS == 30


def test_missing_or_corrupt_file_falls_back(tmp_path):
    assert calibration.load(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert calibration.load(str(bad)) == {}


# ---------------------------------------------------------------------------
# Calibrator FSM (10 fps synthetic metric)
# ---------------------------------------------------------------------------

def run(cal, values, t0=0.0, dt=0.1):
    t = t0
    for v in values:
        t += dt
        cal.update(v, t)
    return t


def test_guided_calibration_succeeds():
    cal = Calibrator("squat")
    rep = ramp(172, 95, 15) + ramp(95, 172, 15)
    run(cal, hold(172, 35) + rep + rep)
    assert cal.stage == "done"
    assert cal.result["SQUAT_DEPTH_ANGLE"] == pytest.approx(
        derive_thresholds("squat", 172, 95)["SQUAT_DEPTH_ANGLE"])
    assert "Calibration saved" in cal.cues


def test_no_body_does_not_advance_stand_phase():
    cal = Calibrator("squat")
    run(cal, [None] * 50)
    assert cal.stage == "stand" and cal.message == "STEP INTO FRAME"


def test_no_reps_times_out_and_fails():
    cal = Calibrator("squat")
    run(cal, hold(172, 35) + hold(170, int(config.CALIBRATION_TIMEOUT * 10) + 5))
    assert cal.stage == "failed" and cal.result is None


def test_high_knee_calibration_direction():
    cal = Calibrator("highknee")
    rep = hold(-0.4, 5) + hold(0.35, 5) + hold(-0.4, 5)
    run(cal, hold(-0.4, 35) + rep + rep)
    assert cal.stage == "done"
    assert -0.4 < cal.result["HIGH_KNEE_MIN_HEIGHT_RATIO"] < 0.35
