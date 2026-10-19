"""State-machine tests for LungeCounter (side view, per-leg sets)."""

import config
from exercise_base import FormIssue
from lunge_counter import LungeCounter
from joint_factory import feed, hold, lunge_pose, mirror, ramp


def lunge(front: str = "left", bottom: float = 90, **pose_kw):
    """One lunge with *front* leg forward."""
    angles = hold(175) + ramp(175, bottom) + hold(bottom) + ramp(bottom, 175) + hold(175)
    return [lunge_pose(a, front=front, **pose_kw) for a in angles]


def test_left_front_lunge_counts_left():
    c = LungeCounter()
    results = feed(c, lunge("left"))
    assert [r.counted for r in results] == [True]
    assert c.side_reps == {"left": 1, "right": 0}


def test_right_front_lunge_counts_right():
    c = LungeCounter()
    feed(c, lunge("right"))
    assert c.side_reps == {"left": 0, "right": 1}


def test_front_leg_detected_when_facing_left():
    c = LungeCounter()
    feed(c, [mirror(j) for j in lunge("right")])
    assert c.side_reps == {"left": 0, "right": 1}


def test_holding_at_bottom_counts_once():
    c = LungeCounter()
    angles = hold(175) + ramp(175, 90) + hold(90, 60) + ramp(90, 175) + hold(175)
    feed(c, [lunge_pose(a) for a in angles])
    assert c.stats.total_reps == 1


def test_shallow_lunge_is_rejected():
    c = LungeCounter()
    results = feed(c, lunge(bottom=130))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.GO_LOWER


def test_knee_past_toes_is_rejected():
    c = LungeCounter()
    results = feed(c, lunge(knee_forward=100))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.KNEE_OVER_TOES
    assert "Knee behind toes" in c.pop_cues()


def test_set_needs_target_on_both_legs():
    c = LungeCounter()
    for _ in range(config.LUNGE_TARGET_REPS):
        feed(c, lunge("left"))
    assert c.stats.sets_completed == 0
    for _ in range(config.LUNGE_TARGET_REPS):
        feed(c, lunge("right"))
    assert c.stats.sets_completed == 1
    assert c.stats.current_set == 2
