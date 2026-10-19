"""State-machine tests for JumpingJackCounter using synthetic front-view poses."""

import config
from exercise_base import FormIssue
from jumping_jack_counter import JackState, JumpingJackCounter
from joint_factory import feed, hold, jumping_jack_pose, ramp


def jack(arms_top: float = 175, feet_top: float = 1.8, hold_open: int = 6):
    """One jumping jack: closed → open → closed."""
    arms = hold(10, 6) + ramp(10, arms_top, 5) + hold(arms_top, hold_open) + ramp(arms_top, 10, 5) + hold(10, 6)
    feet = hold(0.5, 6) + ramp(0.5, feet_top, 5) + hold(feet_top, hold_open) + ramp(feet_top, 0.5, 5) + hold(0.5, 6)
    return [jumping_jack_pose(a, f) for a, f in zip(arms, feet)]


def test_clean_jack_counts_once():
    c = JumpingJackCounter()
    results = feed(c, jack())
    assert [r.counted for r in results] == [True]
    assert c.state == JackState.CLOSED


def test_holding_open_counts_once():
    c = JumpingJackCounter()
    feed(c, jack(hold_open=60))
    assert c.stats.total_reps == 1


def test_arms_without_feet_is_rejected():
    c = JumpingJackCounter()
    results = feed(c, jack(feet_top=0.6))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.FEET_WIDER


def test_feet_without_arms_is_rejected():
    c = JumpingJackCounter()
    results = feed(c, jack(arms_top=80))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.ARMS_OVERHEAD


def test_jitter_near_open_threshold_does_not_double_count():
    c = JumpingJackCounter()
    open_jitter = []
    for i in range(30):   # feet wobble across JJ_FEET_OPEN_RATIO while arms stay up
        open_jitter.append(jumping_jack_pose(175, 1.15 if i % 2 else 1.3))
    poses = jack()[:16] + open_jitter + jack()[-11:]
    feed(c, poses)
    assert c.stats.total_reps == 1


def test_sets_roll_over_at_target():
    c = JumpingJackCounter()
    for _ in range(config.JJ_TARGET_REPS):
        feed(c, jack())
    assert c.stats.sets_completed == 1
    assert c.stats.reps_in_current_set == 0
