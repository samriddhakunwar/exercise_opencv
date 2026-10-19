"""Routine loading / validation, the runner, and a full routine on video."""

import csv
import json

import pytest

import config
import history
import routine as routines
from joint_factory import hold, pushup_pose, ramp, squat_pose
from test_app_video import FPS, run_video  # noqa: F401  (fixture)


def write(tmp_path, data) -> str:
    path = tmp_path / "routines.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_sample_routines_are_valid():
    loaded = routines.load_routines("routines.json")
    assert {"beginner", "fullbody", "cardio"} <= set(loaded)
    beginner = loaded["beginner"].steps
    assert beginner[0] == routines.Step("squat", sets=2, reps=10)
    assert beginner[1].is_rest and beginner[1].rest == 30


@pytest.mark.parametrize("step, message", [
    ({"exercise": "burpee"}, "unknown exercise"),
    ({"exercise": "squat", "reps": -3}, "positive"),
    ({"exercise": "plank", "reps": 10}, "plank uses 'seconds'"),
    ({"exercise": "squat", "seconds": 30}, "uses 'reps'"),
    ({"exercise": "squat", "rep": 10}, "unknown field"),
    ({"rest": 30, "exercise": "squat"}, "only takes 'rest'"),
])
def test_invalid_steps_are_reported(tmp_path, step, message):
    path = write(tmp_path, {"bad": {"steps": [step]}})
    with pytest.raises(ValueError, match=message):
        routines.load_routines(path)


def test_routine_needs_an_exercise(tmp_path):
    with pytest.raises(ValueError, match="no exercise steps"):
        routines.load_routines(write(tmp_path, {"lazy": {"steps": [{"rest": 10}]}}))


def test_runner_walks_steps():
    r = routines.load_routines("routines.json")["beginner"]
    run = routines.RoutineRunner(r)
    assert run.progress_text() == "BEGINNER 1/3"
    assert run.next_exercise() == "pushup"
    run.advance()                                   # rest
    assert run.step.is_rest and run.progress_text() == "BEGINNER 2/3"
    run.advance(); run.advance(); run.advance()     # push-up, rest, plank
    assert run.step.exercise == "plank" and run.next_exercise() is None
    assert run.advance() is None and run.done


def test_routine_runs_end_to_end_on_video(run_video, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "ROUTINES_FILE", write(tmp_path, {"mini": {
        "name": "Mini",
        "steps": [
            {"exercise": "squat", "sets": 1, "reps": 2},
            {"rest": 2},
            {"exercise": "plank", "sets": 1, "seconds": 3},
        ],
    }}))
    rep = hold(175) + ramp(175, 85) + hold(85) + ramp(85, 175) + hold(175)
    poses = ([squat_pose(a) for a in rep * 2]
             + [squat_pose(175)] * int(2.5 * FPS)          # rest (+ a little spare)
             + [pushup_pose(170)] * (5 * FPS))             # plank
    tracker = run_video(None, poses, routine="mini")

    assert tracker._routine.done
    results = tracker._routine.results
    assert [r["exercise"] for r in results] == ["squat", "plank"]
    assert results[0]["reps"] == 2 and results[1]["sets"] == 1

    # Squat reps beyond the target don't belong to the plank step
    assert history.last_session("squat")["reps"] == 2
    assert history.last_session("plank")["sets"] == 1

    files = list((tmp_path / config.STATS_DIR).glob("routine_mini_*.csv"))
    assert len(files) == 1
    rows = list(csv.reader(open(files[0], encoding="utf-8-sig")))
    assert rows[0] == ["Routine", "Mini"] and rows[2] == ["Completed", "yes"]
    assert [row[1] for row in rows[5:7]] == ["SQUAT", "PLANK"]
