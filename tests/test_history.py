"""History database, progress summary and charts."""

import datetime

import pytest

import history
import progress
from high_knee_counter import HighKneeCounter
from plank_counter import PlankCounter
from squat_counter import SquatCounter

DAY = datetime.timedelta(days=1)
TODAY = datetime.date(2026, 10, 19)


def squat_session(depths):
    c = SquatCounter()
    for d in depths:
        c._accept_rep(2.0, d, c.CALORIES_PER_REP, c.TARGET_REPS, c.TARGET_SETS)
    return c


def plank_session(holds):
    c = PlankCounter()
    c.stats.rep_durations.extend(holds)
    c.stats.depths.extend([175.0] * len(holds))
    return c


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "history.db")


def at(days_ago: int) -> datetime.datetime:
    return datetime.datetime.combine(TODAY - days_ago * DAY, datetime.time(18, 0))


def test_add_and_read_back(db):
    history.add_session("squat", squat_session([90, 100]), 60, db, date=at(1))
    history.add_session("squat", squat_session([80, 90, 100]), 60, db, date=at(0))
    last = history.last_session("squat", db)
    assert last["reps"] == 3 and last["avg_metric"] == pytest.approx(90)
    assert last["best_metric"] == 80                       # deepest angle
    assert history.personal_record("squat", db) == 3       # most reps
    assert history.personal_record("pushup", db) is None
    assert history.last_time_text("squat", db) == "Last time: 3 reps, avg depth 90°"
    assert history.last_time_text("pushup", db) == ""


def test_plank_record_is_longest_hold(db):
    history.add_session("plank", plank_session([30.0, 42.5]), 120, db)
    assert history.personal_record("plank", db) == 42.5
    assert history.last_time_text("plank", db) == "Last time: 0 sets, longest hold 42s"


def test_null_metric_falls_back_to_reps(db):
    history.add_session("highknee", HighKneeCounter(), 10, db)   # no lifts → NULL metrics
    assert history.last_time_text("highknee", db) == "Last time: 0 reps"


@pytest.mark.parametrize("days_ago, expected", [
    ([0, 1, 2], 3),
    ([1, 2], 2),          # nothing today yet — streak still alive
    ([0, 2, 3], 1),       # gap yesterday
    ([2, 3], 0),          # missed yesterday — broken
    ([], 0),
])
def test_current_streak(days_ago, expected):
    assert progress.current_streak([TODAY - d * DAY for d in days_ago], TODAY) == expected


def test_summary_personal_bests_respect_direction(db):
    history.add_session("squat", squat_session([95, 100]), 60, db, date=at(2))
    history.add_session("squat", squat_session([85, 90, 92]), 60, db, date=at(1))
    hk = HighKneeCounter()
    hk.total_lifts, hk.max_knee_height, hk.avg_knee_height = 10, 0.9, 0.8
    history.add_session("highknee", hk, 60, db, date=at(1))
    hk.max_knee_height = 0.7
    history.add_session("highknee", hk, 60, db, date=at(0))
    history.add_session("plank", plank_session([31.0]), 60, db, date=at(1))
    history.add_session("plank", plank_session([25.0]), 60, db, date=at(0))

    text = progress.summary_text(history.all_sessions(db), TODAY)
    assert "Current streak  : 3 day(s)" in text
    assert "SQUAT         : 3 reps, deepest angle 85°" in text
    assert "HIGH KNEES    : 0 reps, highest knee lift 90%" in text
    assert "PLANK         : longest hold 31.0s" in text


def test_save_charts_writes_all_three(db, tmp_path):
    for i in range(3):
        history.add_session("squat", squat_session([90 - i * 3] * 5), 60, db, date=at(i * 7))
        history.add_session("plank", plank_session([30.0 + i * 5]), 60, db, date=at(i * 7))
    paths = progress.save_charts(history.all_sessions(db), str(tmp_path / "charts"))
    assert sorted(p.name for p in paths) == ["plank_hold.png", "reps_per_week.png", "squat_depth.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_empty_history_summary():
    assert "No workouts" in progress.summary_text([])
