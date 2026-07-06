# 🏋️ AI Fitness Tracker

A **production-quality**, real-time multi-exercise fitness tracker built with Python, OpenCV, and MediaPipe Pose.
Supports **Squats**, **Push-ups**, and **High Knees** — switch between them live with a single keystroke.
Track reps, validate form, monitor workout progress, and receive live audio coaching — all from your webcam.

---

## ✨ Features

| Category | Details |
|---|---|
| **Real-time detection** | MediaPipe Pose at 25–30 FPS |
| **Squat counting** | Knee-angle finite-state machine with anti-double-count guard |
| **Push-up counting** | Elbow-angle FSM with side-view gate and body-alignment check |
| **High Knee counting** | Dual-leg alternating FSM with pace tracking and cardio/rep modes |
| **Form validation** | Per-exercise: depth, extension, alignment, knee height, pace cues |
| **Live HUD** | Rep counter, set progress, depth/height bar, coaching message, FPS, timer |
| **Pace tracking** | Current & average steps-per-minute; colour-coded pace indicator |
| **Cardio mode** | Time-based High Knee sessions (30 / 45 / 60 s) with countdown display |
| **Audio coaching** | TTS via pyttsx3 ("One", "Halfway There", "Go Lower", "Lift Above Hip", etc.) |
| **Exercise switching** | Press **E** to cycle Squat → Push-Up → High Knees mid-session |
| **Session stats** | Total reps/lifts, calories, fastest/slowest rep, avg depth, pace |
| **CSV export** | Auto-saves to `workouts/<exercise>_session_YYYYMMDD_HHMMSS.csv` |
| **Screenshot** | Press **S** to capture the current frame |
| **Recording** | Press **V** to toggle session video recording |
| **Keyboard controls** | Space (pause) · R (reset) · E (switch exercise) · S (screenshot) · V (record) · Q (quit) |
| **Countdown** | Configurable countdown before the session (and after each exercise switch) |

---

## 📋 Requirements

- Python 3.10+
- A webcam (built-in or USB)
- Windows / macOS / Linux

---

## 🚀 Installation

```bash
# 1. Clone or download the project
git clone <repo-url>
cd fitness-tracker

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

## 📁 Project Structure

```
fitness-tracker/
│
├── app.py               ← Entry point, main loop, keyboard/audio/CSV
├── pose_detector.py     ← MediaPipe Pose wrapper + landmark smoother
├── squat_counter.py     ← ExerciseBase + SquatCounter FSM + shared types
├── pushup_counter.py    ← PushUpCounter FSM (elbow-angle, body alignment)
├── high_knee_counter.py ← HighKneeCounter FSM (dual-leg, pace, cardio/rep)
├── angle_utils.py       ← Geometry helpers (angles, smoothing, alignment)
├── ui.py                ← OpenCV HUD renderer (panels, overlays, summary)
├── config.py            ← All configurable constants
│
├── workouts/            ← Auto-created; CSV session exports go here
├── screenshots/         ← Auto-created; S-key screenshots go here
├── recordings/          ← Auto-created (if recording enabled)
│
├── requirements.txt
└── README.md
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Space` | Pause / Resume |
| `R` | Reset current session |
| `E` | Cycle exercise (Squat → Push-Up → High Knees) |
| `S` | Save screenshot |
| `V` | Toggle video recording |
| `Q` / `Esc` | Quit |

---

## ⚙️ Customising Settings (`config.py`)

All behaviour is controlled through `config.py`.  No other file needs to be edited.

```python
# ── Squat angle thresholds ─────────────────────────────────────────────
STANDING_ANGLE_MIN  = 155.0   # Knee angle considered "standing"
SQUAT_DEPTH_ANGLE   = 100.0   # Qualifies as a proper squat
FULL_EXT_ANGLE      = 150.0   # Min angle to be counted as "standing"

# ── Push-up angle thresholds ───────────────────────────────────────────
PUSHUP_TOP_ANGLE      = 155.0  # Elbow angle considered "arms locked out"
PUSHUP_BOTTOM_ANGLE   = 100.0  # Elbow angle qualifying as proper depth
PUSHUP_FULL_EXT_ANGLE = 140.0  # Minimum elbow angle to count as extended

# ── High Knee thresholds ───────────────────────────────────────────────
HIGH_KNEE_MIN_HEIGHT_RATIO   = 0.72   # Knee lift ratio (1.0 = hip level)
HIGH_KNEE_SUPPORT_KNEE_ANGLE = 150.0  # Min angle on the supporting leg
HIGH_KNEE_MIN_LIFT_DELTA     = 0.05   # Min δ-motion before lift counts
HIGH_KNEE_MIN_SPEED          = 30.0   # spm below which pace is "too slow"
HIGH_KNEE_GOOD_SPEED         = 60.0   # spm above which pace is "good"
HIGH_KNEE_MAX_SPEED          = 180.0  # spm above which pace is "too fast"

# ── Workout targets ────────────────────────────────────────────────────
TARGET_REPS              = 15    # Squats per set
TARGET_SETS              = 3
TARGET_PUSHUPS           = 20    # Push-ups per set
TARGET_PUSHUP_SETS       = 3
HIGH_KNEE_TARGET_REPS    = 20    # Left-right cycles per set
HIGH_KNEE_TARGET_SETS    = 3
HIGH_KNEE_CARDIO_DURATIONS = [30, 45, 60]   # Cardio mode durations (s)

# ── Camera ─────────────────────────────────────────────────────────────
CAMERA_INDEX = 0              # Change if you have multiple cameras

# ── Audio ──────────────────────────────────────────────────────────────
AUDIO_ENABLED = True
AUDIO_COOLDOWN = 1.5          # Seconds between TTS messages

# ── UI toggles ─────────────────────────────────────────────────────────
SHOW_FPS             = True
SHOW_PROGRESS_BAR    = True
DRAW_LANDMARKS       = True
SHOW_KNEE_HEIGHT_BAR = True   # Height bar in High Knee mode

# ── Stats ──────────────────────────────────────────────────────────────
SAVE_STATS_CSV = True
USER_WEIGHT_KG = 70.0         # Used for calorie estimate
```

---

## 🧠 How Detection Works

### Squat Detection

| Stage | Description |
|---|---|
| **Landmarks** | Hip → Knee → Ankle (better-visible leg selected automatically) |
| **Angle** | Interior knee angle via dot-product / cosine rule |
| **Smoothing** | EMA landmark smoother + moving-average angle smoother |

#### Squat FSM
```
STANDING ─(angle < 155°)────────► GOING_DOWN
GOING_DOWN ─(angle ≤ 100°)──────► BOTTOM
BOTTOM ─(angle > 120°)──────────► GOING_UP
GOING_UP ─(angle ≥ 155°)────────► STANDING  ← rep evaluated here
```

#### Squat form checks (all must pass)
| Check | Description |
|---|---|
| Squat depth | `min_angle ≤ SQUAT_DEPTH_ANGLE` |
| Hip below knee | Hip y ≥ knee y at the bottom |
| Full extension | Angle ≥ `FULL_EXT_ANGLE` at the top |
| Back straight | Torso lean < `TORSO_LEAN_THRESHOLD` degrees |

---

### Push-Up Detection

| Stage | Description |
|---|---|
| **Landmarks** | Shoulder → Elbow → Wrist (more-visible arm selected automatically) |
| **Side-view gate** | Elbow-vs-shoulder horizontal offset check (profile view required) |
| **Body alignment** | True perpendicular hip deviation from shoulder→ankle line |

#### Push-Up FSM
```
TOP ─(angle < 155°)──────────────► LOWERING
LOWERING ─(angle ≤ 100°)─────────► BOTTOM
BOTTOM ─(angle > 125°)───────────► PUSHING_UP
PUSHING_UP ─(angle ≥ 155°)───────► TOP  ← rep evaluated here
```

#### Push-up form checks (all must pass)
| Check | Description |
|---|---|
| Sufficient depth | `min_angle ≤ PUSHUP_BOTTOM_ANGLE` |
| Full arm extension | Elbow angle ≥ `PUSHUP_FULL_EXT_ANGLE` at the top |
| Body straight | Hip perpendicular deviation ≤ `PUSHUP_HIP_SAG_THRESHOLD` |
| Side view | Elbow-shoulder x-offset ratio ≥ 0.10 |

---

### High Knee Detection

Tracked landmarks: **Left/Right Hip, Knee, Ankle, Shoulder**

| Stage | Description |
|---|---|
| **Primary metric** | Normalised knee-lift ratio relative to hip height |
| **Motion gate** | Minimum upward Δ required before a lift is registered |
| **Support check** | Opposite-leg knee angle verifies weight-bearing stance |
| **Smoothing** | Per-leg moving-average smoother reduces jitter |

#### High Knee FSM (per leg — two independent FSMs run in parallel)
```
READY ─(knee rises > δ)──────────► LIFTING
LIFTING ─(lift ≥ threshold)──────► UP        ← lift counted here
        ─(drops before threshold)─► READY     (discard jitter)
UP ─(lift drops < threshold - 0.05)─► LOWERING
LOWERING ─(near standing)────────► DOWN
DOWN ─(other leg is DOWN)────────► READY     (alternate)
```

A full **left + right** cycle = **1 repetition** in Rep mode.
In **Cardio mode** every knee-lift increments the total counter.

#### High Knee form checks
| Check | Description |
|---|---|
| Knee height | `lift ≥ HIGH_KNEE_MIN_HEIGHT_RATIO` (relative to hip) |
| Pace lower bound | `current_pace ≥ HIGH_KNEE_MIN_SPEED` spm |
| Pace upper bound | `current_pace ≤ HIGH_KNEE_MAX_SPEED` spm |

#### Pace display colours
| Colour | Condition |
|---|---|
| 🟢 Green | pace ≥ `HIGH_KNEE_GOOD_SPEED` |
| 🟡 Yellow | pace ≥ `HIGH_KNEE_MIN_SPEED` but below good |
| 🔴 Red | pace < `HIGH_KNEE_MIN_SPEED` |

---

## 🏗️ Architecture

```
ExerciseBase           ← abstract base (squat_counter.py)
  ├── SquatCounter     ← knee-angle FSM
  ├── PushUpCounter    ← elbow-angle FSM (pushup_counter.py)
  └── HighKneeCounter  ← dual-leg marching FSM (high_knee_counter.py)

FitnessTrackerApp      ← orchestrates detection → counter → UI → audio
  ├── PoseDetector     ← MediaPipe wrapper
  ├── UIRenderer       ← OpenCV HUD (exercise-aware)
  ├── AudioFeedback    ← TTS dispatcher
  └── StatsLogger      ← CSV export
```

Adding a new exercise:
1. Subclass `ExerciseBase`, implement `process_frame` and `reset`.
2. Add it to `_EXERCISE_CYCLE` and `_EXERCISE_META` in `app.py`.
3. Add config constants in `config.py`.

All shared code (stats, audio, CSV logging, countdown, pause/resume, screenshot, recording) is inherited and requires **zero duplication**.

---

## 📊 Session Statistics (CSV example)

**High Knee session:**
```
Metric,Value
Exercise,HIGH KNEES
Date,2026-07-06 22:00
Total Reps,60
Sets Completed,3
Duration (s),120.0
Calories,12.0
Total Lifts,120
Avg Knee Height,84%
Max Knee Height,97%
Avg Pace (spm),60.0
Peak Pace (spm),78.5

Rep #,Duration (s),Min Knee Ht
1,1.98,0.741
2,2.05,0.762
...
```

**Squat / Push-up sessions** include `Avg Angle`, `Fastest Rep`, `Slowest Rep` instead.

---

## 🖼️ Screenshots

> _Place your screenshots here after running the application._

---

## 🔮 Future Roadmap

### Exercises (same architecture, new FSMs)
- [ ] **LungeCounter** — front-knee-angle FSM
- [ ] **ButtKickCounter** — heel-to-hip-distance FSM
- [ ] **JumpingJackCounter** — limb-spread FSM
- [ ] **MountainClimberCounter** — alternating knee-to-chest FSM
- [ ] **SkaterCounter** — lateral weight-shift FSM
- [ ] **BurpeeCounter** — multi-phase compound FSM
- [ ] **CurlCounter** — elbow flexion FSM
- [ ] **PlankCounter** — body-alignment timer

### App improvements
- [ ] GUI settings panel with mode selection (Rep / Cardio)
- [ ] Live rep-speed and pace graph overlay
- [ ] Cloud sync / leaderboard
- [ ] Multi-person tracking
- [ ] Voice-commanded workout programs
- [ ] Workout program builder (circuit / HIIT)

---

## 📝 License

MIT — feel free to use, modify, and distribute.
