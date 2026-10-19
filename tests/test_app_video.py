"""End-to-end: app.py --video --headless with a fake pose detector.

The real MediaPipe model is swapped for a detector that replays synthetic
poses, so this exercises the whole loop (video read → counter → audio
dispatch → CSV + history save → summary) without needing a person on video.
"""

import cv2
import numpy as np
import pytest

import app
import clock
import config
import history
from joint_factory import hold, pushup_pose, ramp, squat_pose

FPS = 30


class FakeDetector:
    """Stands in for PoseDetector, returning one scripted pose per frame."""

    poses: list = []

    def __init__(self):
        self._i = 0

    def process(self, frame):
        joints = self.poses[self._i] if self._i < len(self.poses) else None
        self._i += 1
        return joints, frame

    def release(self):
        pass


@pytest.fixture
def run_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "HISTORY_DB", str(tmp_path / "history.db"))
    monkeypatch.setattr(app, "PoseDetector", FakeDetector)

    def run(exercise, poses):
        FakeDetector.poses = poses
        path = str(tmp_path / "clip.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 36))
        for _ in poses:
            writer.write(np.zeros((36, 64, 3), np.uint8))
        writer.release()
        tracker = app.FitnessTrackerApp(exercise=exercise, video=path, headless=True)
        tracker.run()
        return tracker

    yield run
    clock.use_wall_clock()


def test_squats_from_video_are_counted_and_saved(run_video, tmp_path):
    rep = hold(175) + ramp(175, 85) + hold(85) + ramp(85, 175) + hold(175)
    tracker = run_video("squat", [squat_pose(a) for a in rep * 2])

    assert tracker._counter.stats.total_reps == 2
    assert "Reps     : 2" in tracker.summary()
    assert len(list((tmp_path / config.STATS_DIR).glob("squat_session_*.csv"))) == 1
    row = history.last_session("squat")
    assert row["reps"] == 2
    assert row["duration"] == pytest.approx(len(rep) * 2 / FPS, abs=0.1)   # video time


def test_plank_hold_uses_video_time(run_video):
    seconds = config.PLANK_TARGET_SECONDS + 2
    tracker = run_video("plank", [pushup_pose(170)] * (seconds * FPS))
    # Processing takes far less than 32 s of wall time — the set only completes
    # if the hold is timed on the video's clock.
    assert tracker._counter.stats.sets_completed == 1
    assert history.last_session("plank")["best_metric"] == pytest.approx(seconds, abs=0.2)


def test_parse_args():
    args = app.parse_args(["--video", "x.mp4", "--exercise", "curl", "--camera", "2",
                           "--no-audio", "--headless"])
    assert (args.video, args.exercise, args.camera, args.no_audio, args.headless) == \
        ("x.mp4", "curl", 2, True, True)
    with pytest.raises(SystemExit):
        app.parse_args(["--exercise", "burpee"])
