"""State-machine tests for SquatCounter using synthetic side-view poses."""

import config
from exercise_base import FormIssue
from squat_counter import SquatCounter, SquatState
from joint_factory import FRAME, feed, hold, ramp, squat_pose


def squat_rep(bottom: float = 85.0, torso_lean: float = 0.0):
    """One squat: stand → descend to *bottom* → stand."""
    angles = hold(175) + ramp(175, bottom) + hold(bottom) + ramp(bottom, 175) + hold(175)
    return [squat_pose(a, torso_lean) for a in angles]


def test_clean_rep_counts_once():
    c = SquatCounter()
    results = feed(c, squat_rep())
    assert len(results) == 1 and results[0].counted
    assert c.stats.total_reps == 1
    assert c.state == SquatState.STANDING
    assert c.stats.depths[0] <= config.SQUAT_DEPTH_ANGLE


def test_shallow_rep_does_not_count():
    c = SquatCounter()
    results = feed(c, squat_rep(bottom=125))
    assert c.stats.total_reps == 0
    assert [r.form_issue for r in results] == [FormIssue.GO_LOWER]
    assert c.state == SquatState.STANDING


def test_small_dip_is_ignored():
    c = SquatCounter()
    assert feed(c, squat_rep(bottom=150)) == []
    assert c.state == SquatState.STANDING


def test_holding_and_bouncing_at_bottom_counts_once():
    c = SquatCounter()
    bottom = hold(85, 60) + ramp(85, 115, 5) + ramp(115, 85, 5) + hold(85, 30)
    angles = hold(175) + ramp(175, 85) + bottom + ramp(85, 175) + hold(175, 30)
    feed(c, [squat_pose(a) for a in angles])
    assert c.stats.total_reps == 1


def test_leaning_torso_rejects_rep():
    c = SquatCounter()
    results = feed(c, squat_rep(torso_lean=60))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.BAD_BACK


def test_sets_roll_over_at_target():
    c = SquatCounter()
    for _ in range(config.TARGET_REPS):
        feed(c, squat_rep())
    assert c.stats.sets_completed == 1
    assert c.stats.current_set == 2
    assert c.stats.reps_in_current_set == 0
    feed(c, squat_rep())
    assert c.stats.reps_in_current_set == 1


def test_workout_completes_after_all_sets():
    c = SquatCounter()
    for _ in range(config.TARGET_REPS * config.TARGET_SETS):
        feed(c, squat_rep())
    assert c.session_complete


def test_missing_body_reports_issue():
    c = SquatCounter()
    result, issue, angle = c.process_frame(None, FRAME)
    assert result is None and issue == FormIssue.BODY_NOT_FOUND and angle == 0.0


def test_reset_clears_progress():
    c = SquatCounter()
    feed(c, squat_rep())
    c.reset()
    assert c.stats.total_reps == 0 and c.state == SquatState.STANDING
