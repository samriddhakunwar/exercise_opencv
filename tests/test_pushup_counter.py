"""State-machine tests for PushUpCounter using synthetic side-view poses."""

import config
from exercise_base import FormIssue
from pushup_counter import PushUpCounter, PushUpState
from joint_factory import feed, hold, pushup_pose, ramp


def pushup_rep(bottom: float = 80.0, **pose_kw):
    """One push-up: arms locked → lower to *bottom* → lock out."""
    angles = hold(170) + ramp(170, bottom) + hold(bottom) + ramp(bottom, 170) + hold(170)
    return [pushup_pose(a, **pose_kw) for a in angles]


def test_clean_rep_counts_once():
    c = PushUpCounter()
    results = feed(c, pushup_rep())
    assert len(results) == 1 and results[0].counted
    assert c.stats.total_reps == 1
    assert c.state == PushUpState.TOP


def test_shallow_rep_does_not_count():
    c = PushUpCounter()
    results = feed(c, pushup_rep(bottom=115))
    assert c.stats.total_reps == 0
    assert [r.form_issue for r in results] == [FormIssue.GO_LOWER]
    assert c.state == PushUpState.TOP


def test_small_dip_is_ignored():
    c = PushUpCounter()
    assert feed(c, pushup_rep(bottom=150)) == []
    assert c.state == PushUpState.TOP


def test_holding_at_bottom_counts_once():
    c = PushUpCounter()
    angles = hold(170) + ramp(170, 80) + hold(80, 60) + ramp(80, 170) + hold(170, 30)
    feed(c, [pushup_pose(a) for a in angles])
    assert c.stats.total_reps == 1


def test_sagging_hips_rejects_rep():
    c = PushUpCounter()
    results = feed(c, pushup_rep(hip_offset=0.25))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.STRAIGHTEN


def test_front_view_is_gated():
    c = PushUpCounter()
    issues = set()
    for joints in pushup_rep(upper_arm_dir=90.0):     # elbow straight below shoulder
        _, issue, _ = c.process_frame(joints, (720, 1280))
        issues.add(issue)
    assert c.stats.total_reps == 0
    assert FormIssue.MOVE_SIDEWAYS in issues


def test_sets_roll_over_at_target():
    c = PushUpCounter()
    for _ in range(config.TARGET_PUSHUPS):
        feed(c, pushup_rep())
    assert c.stats.sets_completed == 1
    assert c.stats.current_set == 2
    assert c.stats.reps_in_current_set == 0
