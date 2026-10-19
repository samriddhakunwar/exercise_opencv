"""State-machine tests for HighKneeCounter using synthetic front-view poses."""

import config
from high_knee_counter import HighKneeCounter, HighKneeMode
from joint_factory import feed, high_knee_pose, hold

DOWN = -0.5      # standing knee height
UP   = 0.9       # above HIGH_KNEE_MIN_HEIGHT_RATIO


def lift(side: str, height: float = UP):
    """One knee lift on *side* followed by a return to standing."""
    seq = hold(DOWN, 8) + hold(height, 8) + hold(DOWN, 8)
    if side == "left":
        return [high_knee_pose(h, DOWN) for h in seq]
    return [high_knee_pose(DOWN, h) for h in seq]


def test_left_then_right_is_one_rep():
    c = HighKneeCounter()
    feed(c, lift("left"))
    assert c.total_lifts == 1 and c.stats.total_reps == 0
    feed(c, lift("right"))
    assert c.total_lifts == 2 and c.stats.total_reps == 1


def test_same_leg_twice_does_not_complete_rep():
    c = HighKneeCounter()
    feed(c, lift("left") + lift("left"))
    assert c.total_lifts == 1
    assert c.stats.total_reps == 0


def test_low_lift_does_not_count():
    c = HighKneeCounter()
    feed(c, lift("left", height=0.3) + lift("right", height=0.3))
    assert c.total_lifts == 0


def test_holding_knee_up_counts_once():
    c = HighKneeCounter()
    seq = hold(DOWN, 8) + hold(UP, 60) + hold(DOWN, 8)
    feed(c, [high_knee_pose(h, DOWN) for h in seq])
    assert c.total_lifts == 1


def test_cardio_mode_counts_every_lift():
    c = HighKneeCounter(mode=HighKneeMode.CARDIO, cardio_duration=60)
    c.on_session_start()
    feed(c, lift("left") + lift("right") + lift("left"))
    assert c.stats.total_reps == 3
    assert not c.session_complete


def test_sets_roll_over_at_target():
    c = HighKneeCounter()
    for _ in range(config.HIGH_KNEE_TARGET_REPS):
        feed(c, lift("left") + lift("right"))
    assert c.stats.sets_completed == 1
    assert c.stats.current_set == 2
    assert c.stats.reps_in_current_set == 0
