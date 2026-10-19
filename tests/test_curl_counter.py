"""State-machine tests for CurlCounter (per-arm FSMs, per-side sets)."""

import config
from exercise_base import FormIssue
from curl_counter import CurlCounter
from joint_factory import curl_pose, feed, hold, ramp


def curl(side: str = "left", top: float = 40, bottom: float = 165, **pose_kw):
    """One curl on *side* (the other arm stays extended)."""
    angles = hold(bottom, 6) + ramp(bottom, top) + hold(top, 6) + ramp(top, bottom) + hold(bottom, 6)
    other = 165
    if side == "left":
        return [curl_pose(a, other, **pose_kw) for a in angles]
    if side == "right":
        return [curl_pose(other, a, **pose_kw) for a in angles]
    return [curl_pose(a, a, **pose_kw) for a in angles]        # both arms


def test_left_curl_counts_left_only():
    c = CurlCounter()
    results = feed(c, curl("left"))
    assert [r.counted for r in results] == [True]
    assert c.side_reps == {"left": 1, "right": 0}
    assert c.rep_announcement() == "Left 1"


def test_simultaneous_curls_count_both_arms():
    c = CurlCounter()
    feed(c, curl("both"))
    assert c.side_reps == {"left": 1, "right": 1}
    assert c.stats.total_reps == 2


def test_partial_curl_is_rejected():
    c = CurlCounter()
    results = feed(c, curl("left", top=100))
    assert c.stats.total_reps == 0
    assert results and results[-1].form_issue == FormIssue.CURL_HIGHER


def test_not_extending_at_bottom_is_rejected():
    c = CurlCounter()
    angles = hold(165, 6) + ramp(165, 40) + ramp(40, 110) + ramp(110, 40) + ramp(40, 165) + hold(165, 6)
    results = feed(c, [curl_pose(a, 165) for a in angles])
    assert FormIssue.EXTEND_ARM in [r.form_issue for r in results]
    assert c.side_reps["left"] == 1          # the final, fully-extended rep counts


def test_elbow_swinging_out_is_rejected():
    c = CurlCounter()
    poses = curl("left")
    swung = [curl_pose(a, 165, upper_arm_tilt=50) for a in ramp(165, 40)]
    results = feed(c, poses[:10] + swung + poses[-16:])
    assert c.stats.total_reps == 0
    assert FormIssue.ELBOW_DRIFT in [r.form_issue for r in results]
    assert "Keep your elbow still" in c.pop_cues()


def test_elbow_rising_is_rejected():
    c = CurlCounter()
    angles = hold(165, 6) + ramp(165, 40)
    poses = [curl_pose(a, 165) for a in angles[:12]]          # rep under way
    poses += [curl_pose(a, 165, elbow_rise=60) for a in angles[12:]]  # 60/120 = 0.5 rise
    poses += [curl_pose(a, 165) for a in ramp(40, 165) + hold(165, 6)]
    feed(c, poses)
    assert c.stats.total_reps == 0


def test_invisible_arm_is_ignored():
    c = CurlCounter()
    feed(c, curl("both", right_arm_visibility=0.1))
    assert c.side_reps == {"left": 1, "right": 0}


def test_set_needs_target_on_both_sides():
    c = CurlCounter()
    for _ in range(config.CURL_TARGET_REPS + 2):          # extra left reps
        feed(c, curl("left"))
    assert c.stats.sets_completed == 0
    assert c.side_reps["left"] == config.CURL_TARGET_REPS
    for _ in range(config.CURL_TARGET_REPS):
        feed(c, curl("right"))
    assert c.stats.sets_completed == 1
    assert c.side_reps == {"left": 0, "right": 0}
    assert c.stats.reps_in_current_set == 0
    assert c.side_total["left"] == config.CURL_TARGET_REPS + 2
