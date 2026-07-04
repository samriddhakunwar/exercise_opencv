# 🏋️ AI Fitness Tracker

A **production-quality**, real-time multi-exercise fitness tracker built with Python, OpenCV, and MediaPipe Pose.
Supports **Squats** and **Push-ups** — switch between them live with a single keystroke.
Track reps, validate form, monitor workout progress, and receive live audio coaching — all from your webcam.

---

## ✨ Features

| Category | Details |
|---|---|
| **Real-time detection** | MediaPipe Pose at 25–30 FPS |
| **Squat counting** | Knee-angle finite-state machine with anti-double-count guard |
| **Push-up counting** | Elbow-angle FSM with side-view gate and body-alignment check |
| **Form validation** | Per-exercise: depth, full extension, back/body alignment, knee cave, hip sag/pike |
| **Live HUD** | Rep counter, set progress, depth bar, coaching message, FPS, timer |
| **Audio coaching** | TTS via pyttsx3 ("One", "Halfway There", "Go Lower", "Lock Out Your Arms", etc.) |
| **Exercise switching** | Press **E** to cycle Squat ↔ Push-up mid-session |
| **Session stats** | Total reps, calories, fastest/slowest rep, average depth |
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
├── app.py              ← Entry point, main loop, keyboard/audio/CSV
├── pose_detector.py    ← MediaPipe Pose wrapper + landmark smoother
├── squat_counter.py    ← ExerciseBase + SquatCounter FSM + shared data types
├── pushup_counter.py   ← PushUpCounter FSM (elbow-angle, body alignment)
├── angle_utils.py      ← Geometry helpers (angles, smoothing, alignment)
├── ui.py               ← OpenCV HUD renderer (panels, overlays, summary)
├── config.py           ← All configurable constants
│
├── workouts/           ← Auto-created; CSV session exports go here
├── screenshots/        ← Auto-created; S-key screenshots go here
├── recordings/         ← Auto-created (if recording enabled)
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
| `E` | Switch exercise (Squat ↔ Push-Up) |
| `S` | Save screenshot |
| `V` | Toggle video recording |
| `Q` / `Esc` | Quit |

---

## ⚙️ Customising Settings (`config.py`)

All behaviour is controlled through `config.py`.  No other file needs to be edited.

```python
# ── Squat angle thresholds ────────────────────────────────────────────
STANDING_ANGLE_MIN  = 155.0   # Knee angle considered "standing"
SQUAT_DEPTH_ANGLE   = 100.0   # Qualifies as a proper squat
FULL_EXT_ANGLE      = 150.0   # Min angle to be counted as "standing"

# ── Push-up angle thresholds ──────────────────────────────────────────
PUSHUP_TOP_ANGLE      = 155.0  # Elbow angle considered "arms locked out"
PUSHUP_BOTTOM_ANGLE   = 100.0  # Elbow angle qualifying as proper depth
PUSHUP_FULL_EXT_ANGLE = 140.0  # Minimum elbow angle to count as extended

# ── Workout targets ───────────────────────────────────────────────────
TARGET_REPS        = 15    # Squats per set
TARGET_SETS        = 3
TARGET_PUSHUPS     = 20    # Push-ups per set
TARGET_PUSHUP_SETS = 3

# ── Camera ────────────────────────────────────────────────────────────
CAMERA_INDEX = 0              # Change if you have multiple cameras

# ── Audio ─────────────────────────────────────────────────────────────
AUDIO_ENABLED = True
AUDIO_COOLDOWN = 1.5          # Seconds between TTS messages

# ── UI toggles ────────────────────────────────────────────────────────
SHOW_FPS            = True
SHOW_PROGRESS_BAR   = True
DRAW_LANDMARKS      = True

# ── Stats ─────────────────────────────────────────────────────────────
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

## 🏗️ Architecture

```
ExerciseBase          ← abstract base (squat_counter.py)
  ├── SquatCounter    ← knee-angle FSM
  └── PushUpCounter   ← elbow-angle FSM (pushup_counter.py)

FitnessTrackerApp     ← orchestrates detection → counter → UI → audio
  ├── PoseDetector    ← MediaPipe wrapper
  ├── UIRenderer      ← OpenCV HUD (exercise-aware)
  ├── AudioFeedback   ← TTS dispatcher
  └── StatsLogger     ← CSV export
```

Adding a new exercise:
1. Subclass `ExerciseBase`, implement `process_frame` and `reset`.
2. Add it to `_EXERCISE_CYCLE` and `_EXERCISE_META` in `app.py`.
3. Add config constants in `config.py`.

---

## 📊 Session Statistics (CSV example)

```
Metric,Value
Exercise,PUSH-UP
Date,2026-07-04 13:00
Total Reps,40
Sets Completed,2
Duration (s),312.5
Avg Elbow Angle (°),88.4
Fastest Rep (s),1.82
Slowest Rep (s),3.11
Calories,11.6

Rep #,Duration (s),Min Elbow Angle (°)
1,2.10,91.2
2,1.82,85.6
...
```

---

## 🖼️ Screenshots

> _Place your screenshots here after running the application._

---

## 🔮 Future Improvements

- [ ] **LungeCounter** — front-knee-angle FSM
- [ ] **CurlCounter** — wrist/elbow angle FSM
- [ ] **PlankCounter** — body-alignment timer
- [ ] **JumpingJackCounter** — limb-spread FSM
- [ ] GUI settings panel (Tkinter / PyQt)
- [ ] Live rep-speed graph
- [ ] Cloud sync / leaderboard
- [ ] Multi-person tracking
- [ ] Voice-commanded workout programs

All exercise counters inherit from `ExerciseBase` in `squat_counter.py`, making extension straightforward.

---

## 📝 License

MIT — feel free to use, modify, and distribute.
