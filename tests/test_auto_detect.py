"""Automatic exercise detection from synthetic pose streams (30 fps)."""

import pytest

import config
from auto_detect import ExerciseClassifier
from joint_factory import (high_knee_pose, hold, jumping_jack_pose, lunge_pose,
                           pushup_pose, ramp, squat_pose)

FPS = 30


def stream(cycle, seconds: float = 4.0):
    """Repeat *cycle* (list of poses) for about *seconds*."""
    n = int(seconds * FPS)
    return [cycle[i % len(cycle)] for i in range(n)]


def classify(poses, clf=None):
    clf = clf or ExerciseClassifier()
    for i, joints in enumerate(poses):
        clf.update(joints, i / FPS)
    return clf


SQUAT = [squat_pose(a) for a in ramp(175, 85, 15) + ramp(85, 175, 15)]
LUNGE = [lunge_pose(a) for a in ramp(175, 90, 15) + ramp(90, 175, 15)]
PUSHUP = [pushup_pose(a) for a in ramp(170, 80, 15) + ramp(80, 170, 15)]
PLANK = [pushup_pose(170)]
JACK = [jumping_jack_pose(a, f) for a, f in zip(ramp(10, 175, 8) + ramp(175, 10, 8),
                                                 ramp(0.5, 1.8, 8) + ramp(1.8, 0.5, 8))]
HIGH_KNEE = ([high_knee_pose(h, -0.5) for h in hold(-0.5, 4) + hold(0.9, 6)]
             + [high_knee_pose(-0.5, h) for h in hold(-0.5, 4) + hold(0.9, 6)])


@pytest.mark.parametrize("cycle, expected", [
    (SQUAT, "squat"),
    (LUNGE, "lunge"),
    (PUSHUP, "pushup"),
    (PLANK, "plank"),
    (JACK, "jumpingjack"),
    (HIGH_KNEE, "highknee"),
])
def test_detects_each_exercise(cycle, expected):
    clf = classify(stream(cycle))
    assert clf.detected == expected
    assert clf.confidence >= config.AUTO_MIN_AGREEMENT


def test_needs_stable_time_before_detecting():
    clf = classify(stream(SQUAT, seconds=config.AUTO_STABLE_SECONDS * 0.5))
    assert clf.detected is None


def test_flickering_labels_do_not_switch():
    # Alternate 0.5 s of plank with 0.5 s of standing squats — never stable
    half = int(0.5 * FPS)
    poses = (PLANK * half + SQUAT[:half]) * 6
    assert classify(poses).detected is None


def test_switches_after_change_on_continuous_timeline():
    clf = ExerciseClassifier()
    poses = stream(SQUAT) + stream(PLANK)
    seen = []
    for i, j in enumerate(poses):
        seen.append(clf.update(j, i / FPS))
    assert seen[len(stream(SQUAT)) - 1] == "squat"
    assert seen[-1] == "plank"


def test_standing_still_detects_nothing():
    assert classify(stream([squat_pose(175)])).detected is None


def test_no_body_detects_nothing():
    assert classify([None] * 120).detected is None
