"""Timer tests for PlankCounter using a fake clock (10 fps)."""

import pytest

import config
import plank_counter
from exercise_base import FormIssue
from plank_counter import PlankCounter, PlankState
from joint_factory import FakeClock, feed_timed, mirror, pushup_pose, squat_pose

FPS = 10


def plank(seconds: float, hip_offset: float = 0.0):
    """Plank frames (straight arms, horizontal body) for *seconds*."""
    return [pushup_pose(170, hip_offset=hip_offset)] * int(seconds * FPS)


def standing(seconds: float):
    return [squat_pose(175)] * int(seconds * FPS)


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(plank_counter.time, "time", c)
    return c


def test_hold_reaching_target_completes_set(clock):
    c = PlankCounter()
    results = feed_timed(c, plank(config.PLANK_TARGET_SECONDS + 1), clock)
    assert len(results) == 1 and results[0].counted
    assert c.stats.sets_completed == 1
    assert c.state == PlankState.HOLDING        # timer keeps running past target


def test_sag_pauses_timer_and_cues_hips_up(clock):
    c = PlankCounter()
    feed_timed(c, plank(10), clock)
    c.pop_cues()
    feed_timed(c, plank(5, hip_offset=0.25), clock)
    assert c.state == PlankState.PAUSED
    assert "Hips up" in c.pop_cues()
    _, issue, _ = c.process_frame(pushup_pose(170, hip_offset=0.25), (720, 1280))
    assert issue == FormIssue.HIPS_UP
    feed_timed(c, plank(10), clock)
    assert c.hold_seconds == pytest.approx(20, abs=0.3)


def test_pike_cues_hips_down(clock):
    c = PlankCounter()
    feed_timed(c, plank(2), clock)
    feed_timed(c, plank(1, hip_offset=-0.25), clock)
    assert "Hips down" in c.pop_cues()


def test_sag_detected_when_facing_other_way(clock):
    c = PlankCounter()
    feed_timed(c, [mirror(j) for j in plank(2, hip_offset=0.25)], clock)
    assert c.state == PlankState.PAUSED
    assert "Hips up" in c.pop_cues()


def test_leaving_early_logs_attempt_without_completing_set(clock):
    c = PlankCounter()
    feed_timed(c, plank(12) + standing(2), clock)
    assert c.stats.sets_completed == 0
    assert c.attempts == [pytest.approx(12, abs=0.3)]
    feed_timed(c, plank(3), clock)
    assert c.hold_seconds == pytest.approx(3, abs=0.3)      # fresh hold


def test_brief_break_within_grace_keeps_hold(clock):
    c = PlankCounter()
    feed_timed(c, plank(5) + standing(config.PLANK_BREAK_GRACE / 2) + plank(5), clock)
    assert c.attempts == []
    assert c.hold_seconds == pytest.approx(10, abs=0.3)


def test_all_sets_complete_session(clock):
    c = PlankCounter()
    for _ in range(config.PLANK_TARGET_SETS):
        feed_timed(c, plank(config.PLANK_TARGET_SECONDS + 3) + standing(2), clock)
        if c.session_complete:
            break
    assert c.session_complete
    assert len(c.stats.rep_durations) == config.PLANK_TARGET_SETS
    assert c.longest_hold >= config.PLANK_TARGET_SECONDS
    assert c.stats.calories > 0


def test_not_in_position_prompts_user(clock):
    c = PlankCounter()
    _, issue, _ = c.process_frame(squat_pose(175), (720, 1280))
    assert issue == FormIssue.GET_IN_PLANK


def test_session_end_records_running_hold(clock):
    c = PlankCounter()
    feed_timed(c, plank(8), clock)
    c.on_session_end()
    assert c.attempts == [pytest.approx(8, abs=0.3)]
    assert c.record_value() == pytest.approx(8, abs=0.3)
