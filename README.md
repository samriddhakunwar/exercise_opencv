# 🏋️ AI Fitness Tracker

A real-time, webcam-based exercise tracker built with Python, OpenCV and MediaPipe Pose.
It counts reps (or times holds), checks your form, coaches you out loud, and keeps a
history of every session so you can see your progress.

<!-- Demo GIF: record one and save it as docs/demo.gif (see "Recording a demo GIF" below) -->
![Demo](docs/demo.gif)

**Exercises:** Squats · Push-ups · High Knees · Lunges · Bicep Curls · Jumping Jacks · Plank

---

## ✨ Features

| Category | Details |
|---|---|
| **7 exercises** | Squats, push-ups, high knees, lunges, bicep curls, jumping jacks, plank (timed) |
| **Form checks** | Depth, lock-out, back angle, hip sag / pike, knee past toes, elbow drift, full extension, knee height, pace |
| **Per-side counting** | Curls and lunges count left and right separately; a set completes when both sides hit the target |
| **Live HUD** | Reps or hold timer, L/R counts, set progress, depth bar, coaching message, FPS, timer, calories |
| **Audio coaching** | TTS rep counts, form cues ("Hips up", "Knee behind toes"), set / workout announcements, "New record!" |
| **History & progress** | Every session is logged to SQLite; `progress.py` prints totals, streak and personal bests and draws charts |
| **Calibration** | Press **C** to fit the thresholds to *your* range of motion; saved per exercise |
| **Routines** | Multi-exercise workouts with rest timers from `routines.json` (`--routine fullbody`) |
| **Auto-detect** | Press **A** and the app works out which exercise you're doing and switches counters |
| **Video input** | Analyse a recorded video (`--video clip.mp4`), optionally headless for scripts |
| **Exports** | Per-session CSVs, combined routine CSVs, screenshots, video recording |

---

## 📋 Requirements

- Python 3.10+
- A webcam (or a video file)
- Windows / macOS / Linux

---

## 🚀 Installation

```bash
# 1. Clone the project
git clone https://github.com/samriddhakunwar/exercise_opencv.git
cd exercise_opencv

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

---

## 🖥️ Command-line usage

```bash
python app.py                                   # webcam, starts with squats
python app.py --exercise plank                  # pick the starting exercise
python app.py --camera 1                        # use another webcam
python app.py --no-audio                        # no spoken feedback
python app.py --calibrate --exercise squat      # calibrate first, then train
python app.py --routine beginner                # run a routine from routines.json
python app.py --auto                            # detect the exercise automatically

python app.py --video clip.mp4 --exercise pushup             # analyse a video in a window
python app.py --video clip.mp4 --exercise pushup --headless  # no window, print final stats
```

| Flag | Meaning |
|---|---|
| `--exercise KEY` | `squat`, `pushup`, `highknee`, `lunge`, `curl`, `jumpingjack`, `plank` |
| `--video PATH` | Process a video file instead of the webcam. No countdown; the end of the file ends (and saves) the session. Reps, holds and pace are timed on the video's own timeline. |
| `--headless` | No window; prints a session summary at the end. Implies `--no-audio`. |
| `--camera N` | Override `CAMERA_INDEX` |
| `--no-audio` | Disable text-to-speech |
| `--calibrate` | Run calibration for the starting exercise before counting |
| `--routine NAME` | Run a routine (`beginner`, `fullbody`, `cardio`, or your own) |
| `--auto` | Automatic exercise detection (ignored during a routine) |

Headless output looks like:

```
=== SESSION SUMMARY ===
Exercise : PUSH-UP
Reps     : 12
Sets     : 0 / 3
Duration : 45.2 s
Calories : 3.5 kcal
Avg Elbow: 88.4
Deepest angle : 79°
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Reset the current session (what you did so far is saved first) |
| `E` | Next exercise — or, during a routine, skip to the next step |
| `C` | Calibrate the current exercise (press again to cancel) |
| `A` | Toggle automatic exercise detection |
| `S` | Save a screenshot |
| `V` | Start / stop video recording (`recordings/`) |
| `Q` / `Esc` | Quit (the session is saved) |

---

## 🧍 Camera placement

| Exercise | View | Notes |
|---|---|---|
| Squat | Side | Whole body in frame |
| Push-up | Side | The app asks you to "MOVE SIDEWAYS" if it sees you head-on |
| Plank | Side | Shoulders, hips and ankles visible |
| Lunge | Side | Face either direction; the front leg is detected automatically |
| High knees | Front | Hips and knees visible |
| Jumping jacks | Front | Leave room above your head and to the sides |
| Bicep curls | Front | Both arms visible from shoulder to wrist |

---

## 🧠 How detection works

Every exercise is a small finite-state machine (FSM) fed one frame at a time.
Angles come from MediaPipe landmarks, smoothed with an EMA landmark filter and a
moving-average angle filter. All thresholds live in `config.py`.

### Squat — knee angle (hip → knee → ankle)
```
STANDING ─(< 155°)─► GOING_DOWN ─(≤ 100°)─► BOTTOM ─(> 120°)─► GOING_UP ─(≥ 155°)─► STANDING  ← rep evaluated
```
Checks: depth, full extension at the top, torso lean < `TORSO_LEAN_THRESHOLD`.

### Push-up — elbow angle (shoulder → elbow → wrist)
```
TOP ─(< 155°)─► LOWERING ─(≤ 100°)─► BOTTOM ─(> 125°)─► PUSHING_UP ─(≥ 155°)─► TOP  ← rep evaluated
```
Checks: depth, lock-out, straight body (hip distance from the shoulder→ankle line), side view.

### High knees — per-leg knee lift
Two leg FSMs (`READY → LIFTING → UP → LOWERING → DOWN`) that must alternate; a left + right
cycle is one rep (rep mode), or every lift counts (cardio mode). Pace is shown in steps/min.

### Lunge — front-knee angle, counted per leg
Same shape as the squat FSM. The front leg is the one whose ankle is further in the
direction you face (nose vs shoulders). The rep is rejected if the front knee travels past
the toes by more than `LUNGE_KNEE_TOE_TOLERANCE` × shin length.

### Bicep curl — elbow angle per arm
```
DOWN ─(< 150°)─► CURLING ─(≤ 55°)─► UP ─(> 75°)─► LOWERING ─(≥ 150°)─► DOWN  ← rep evaluated
```
Rejected reps: not curled to the top, curling again before the arm straightens, or the
elbow drifting (upper arm > `CURL_ELBOW_DRIFT_ANGLE` from vertical, or the elbow rising).

### Jumping jacks — arms + feet with hysteresis
Open = both wrists above the nose **and** ankles wider than `JJ_FEET_OPEN_RATIO` × shoulder width.
Closed = wrists below the shoulders **and** feet under `JJ_FEET_CLOSED_RATIO`. The gap between the
two thresholds stops jitter from double-counting. One rep = closed → open → closed.

### Plank — timed hold
The timer runs while the shoulder → hip → ankle line is straight. Hip sag or pike *pauses* it
and coaches "Hips up" / "Hips down". A set completes when one continuous hold reaches
`PLANK_TARGET_SECONDS`; the timer keeps running as a bonus until you get up. Leaving the
position early is logged as an attempt.

---

## 📈 History & progress

Every session is saved twice:

- a detailed CSV in `workouts/<exercise>_session_YYYYMMDD_HHMMSS.csv` (rep-by-rep), and
- one row in **`data/history.db`** (SQLite): date, exercise, reps, sets, duration, calories,
  average metric and best metric (deepest angle, longest plank, …).

```bash
python progress.py               # summary + charts
python progress.py --no-charts   # summary only
```

```
WORKOUT PROGRESS
========================================
Sessions        : 24
Total reps      : 812
Total time      : 96 min
Calories        : 151 kcal
Current streak  : 4 day(s)

PERSONAL BESTS
----------------------------------------
SQUAT         : 29 reps, deepest angle 80°
PLANK         : longest hold 48.0s
```

Charts are written to `data/charts/`: `reps_per_week.png`, `squat_depth.png`, `plank_hold.png`.

In the app, the countdown screen shows your last session for the exercise
("Last time: 15 reps, avg depth 92°"), and beating your record (most reps; longest hold for
plank) shows **NEW RECORD!** and says so.

---

## 🎯 Calibration

Default thresholds suit an "average" range of motion. Calibration fits them to yours:

1. Press **C** (or start with `--calibrate`).
2. Hold the start position for 3 seconds (stand straight / top of the push-up / arms hanging).
3. Do **2 slow, full-range reps**.

The app records your resting and deepest values and rescales the exercise's thresholds onto
that range. Every threshold is clamped to its default ± a safe margin, ordering is preserved,
and a calibration with too little movement is rejected — so a bad calibration can't make
counting impossible. Results are saved per exercise in `data/calibration.json` and loaded on
startup; delete the file to go back to the defaults.

Supported: squat, push-up, lunge, bicep curl, high knees. (Plank is timed, and jumping-jack
thresholds are already relative to your shoulder width.)

---

## 📋 Routines

Routines live in `routines.json`. Three samples are included: `beginner`, `fullbody`, `cardio`.

```json
"beginner": {
  "name": "Beginner",
  "steps": [
    {"exercise": "squat",  "sets": 2, "reps": 10},
    {"rest": 30},
    {"exercise": "pushup", "sets": 2, "reps": 8},
    {"rest": 30},
    {"exercise": "plank",  "sets": 2, "seconds": 20}
  ]
}
```

`python app.py --routine beginner` switches exercises automatically, shows a rest countdown
with "NEXT: PUSH-UP", announces each transition, and at the end saves one combined
`workouts/routine_<name>_<timestamp>.csv` (also if you quit early). Curls and lunges use
`reps` per side; plank uses `seconds`. Mistakes in the file (unknown exercise, `reps` on a
plank, …) are reported with the routine and step number.

---

## 🤖 Automatic exercise detection

Press **A** (or `--auto`). Each frame is labelled from the last 1.5 s of pose features:

| Pose | Exercise |
|---|---|
| Body horizontal, elbows bending | Push-up |
| Body horizontal, elbows still | Plank |
| Upright, wrists above head + feet wide (front view) | Jumping jacks |
| Upright, one knee up at a time, both sides | High knees |
| Upright, knees bending, split stance | Lunge |
| Upright, knees bending, feet together | Squat |

A label must hold for 1.5 s with 80 % agreement before the app switches (no flicker). The HUD
badge shows the current guess and confidence, e.g. `AUTO: SQUAT 92%`. Reps during the
detection window of a new exercise aren't counted. Curls aren't auto-detected.

---

## ⚙️ Configuration (`config.py`)

Everything tunable is in `config.py`: camera, MediaPipe settings, every exercise's
thresholds and targets, calorie estimates, calibration / auto-detect parameters, file
locations and UI toggles. Examples:

```python
TARGET_REPS = 15              # Squats per set
SQUAT_DEPTH_ANGLE = 100.0     # Knee angle that counts as a proper squat
PLANK_TARGET_SECONDS = 30     # Hold that completes a plank set
CURL_TARGET_REPS = 12         # Per arm
AUTO_STABLE_SECONDS = 1.5     # Auto-detect: how long a label must hold
ENABLE_RECORDING = False      # True = record every session automatically
HISTORY_DB = "data/history.db"
```

---

## 🧪 Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite (~140 tests, a few seconds) needs no webcam. Counters are fed synthetic poses built
from joint angles (`tests/joint_factory.py`), and the end-to-end tests run the real app loop
on a generated video with a scripted pose detector — covering video timing, CSV / history
output, calibration, routines and auto-detect.

---

## 🎬 Recording a demo GIF

1. Run the app, press **V** to start recording (saved in `recordings/`), do a few reps and
   press **V** again. *(Any screen recorder works too — Windows `Win + Alt + R`,
   macOS `Cmd + Shift + 5`.)*
2. Convert ~5–10 s of it to a GIF with [ffmpeg](https://ffmpeg.org/):
   ```bash
   ffmpeg -ss 2 -t 8 -i recordings/session_XXXX.mp4 \
     -vf "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
     -loop 0 docs/demo.gif
   ```
3. Keep it under ~5 MB so it loads quickly on GitHub, then commit `docs/demo.gif`.

---

## 📁 Project structure

```
exercise_opencv/
├── app.py                  ← Entry point: CLI, main loop, keys, audio, CSV export
├── exercises.py            ← Exercise registry (key → counter + HUD metadata)
├── exercise_base.py        ← ExerciseBase, SidedExerciseBase, SessionStats, FormIssue, JointData
├── squat_counter.py        ← Squat FSM
├── pushup_counter.py       ← Push-up FSM
├── high_knee_counter.py    ← High-knee dual-leg FSM (rep / cardio modes)
├── lunge_counter.py        ← Lunge FSM (per leg)
├── curl_counter.py         ← Bicep-curl FSM (per arm)
├── jumping_jack_counter.py ← Jumping-jack FSM
├── plank_counter.py        ← Plank hold timer
├── pose_detector.py        ← MediaPipe Pose wrapper + landmark smoothing
├── angle_utils.py          ← Geometry helpers (angles, smoothing, body-line deviation)
├── ui.py                   ← OpenCV HUD renderer
├── history.py              ← SQLite session history
├── progress.py             ← Progress report + charts
├── calibration.py          ← Per-user threshold calibration
├── routine.py              ← Routine loading + runner
├── auto_detect.py          ← Rule-based exercise classifier
├── clock.py                ← Time source (wall clock, or video time)
├── _mp_compat.py           ← Works around a protobuf/TensorFlow import clash
├── config.py               ← All tunable constants
├── routines.json           ← Sample routines
├── tests/                  ← pytest suite
├── requirements.txt / requirements-dev.txt
│
├── workouts/               ← (generated) per-session and routine CSVs
├── data/                   ← (generated) history.db, calibration.json, charts/
├── screenshots/            ← (generated) S-key screenshots
└── recordings/             ← (generated) V-key recordings
```

---

## 🏗️ Architecture

```
ExerciseBase                   ← abstract base (exercise_base.py)
  ├── SquatCounter, PushUpCounter, HighKneeCounter, JumpingJackCounter, PlankCounter
  └── SidedExerciseBase        ← per-side sets
        ├── LungeCounter
        └── CurlCounter

FitnessTrackerApp              ← detection → counter → UI → audio → history
  ├── PoseDetector             ← MediaPipe wrapper (JointData per frame)
  ├── UIRenderer               ← RenderData → HUD
  ├── AudioFeedback            ← threaded TTS
  ├── StatsLogger / history    ← CSV + SQLite
  ├── Calibrator               ← C key
  ├── RoutineRunner            ← --routine
  └── ExerciseClassifier       ← A key
```

### Adding an exercise

1. Subclass `ExerciseBase` (or `SidedExerciseBase`) and implement `process_frame` and `reset`.
   Optional hooks: `hud_fields`, `csv_rows`, `best_metric`, `record_value`, `rep_announcement`, …
2. Put its thresholds in `config.py` and any new form problems in `FormIssue`.
3. Add **one** `ExerciseSpec` entry to `EXERCISES` in `exercises.py`.

The E-key cycle, HUD, CSV export, history, routines and `--exercise` pick it up automatically.

---

## 🔮 Roadmap

- [ ] Browser version using MediaPipe's JavaScript pose model
- [ ] More exercises: burpees, mountain climbers, butt kicks
- [ ] Settings panel (targets, high-knee rep / cardio mode)
- [ ] Multi-person tracking

---

## 📝 License

MIT — feel free to use, modify, and distribute.
