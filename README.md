# 🏋️ Squat Tracker

A **production-quality**, real-time bodyweight squat counter built with Python, OpenCV, and MediaPipe Pose.
Track reps, validate form, monitor workout progress, and receive live audio coaching — all from your webcam.

---

## ✨ Features

| Category | Details |
|---|---|
| **Real-time detection** | MediaPipe Pose at 25–30 FPS |
| **Squat counting** | Finite-state machine with anti-double-count guard |
| **Form validation** | Depth check, hip-below-knee, full extension, back angle, knee cave |
| **Live HUD** | Rep counter, set progress, depth bar, coaching message, FPS, timer |
| **Audio coaching** | TTS via pyttsx3 ("One", "Halfway There", "Go Lower", etc.) |
| **Session stats** | Total reps, calories, fastest/slowest rep, average depth |
| **CSV export** | Auto-saves to `workouts/session_YYYYMMDD_HHMMSS.csv` |
| **Screenshot** | Press **S** to capture the current frame |
| **Recording** | Press **V** to toggle session video recording |
| **Keyboard controls** | Space (pause) · R (reset) · Q (quit) · S (screenshot) |
| **Countdown** | Configurable countdown before the session starts |

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
cd squat-counter

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
squat-counter/
│
├── app.py              ← Entry point, main loop, keyboard/audio/CSV
├── pose_detector.py    ← MediaPipe Pose wrapper + landmark smoother
├── squat_counter.py    ← FSM counter + form validation + session stats
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
| `S` | Save screenshot |
| `V` | Toggle video recording |
| `Q` / `Esc` | Quit |

---

## ⚙️ Customising Settings (`config.py`)

All behaviour is controlled through `config.py`.  No other file needs to be edited.

```python
# Squat angle thresholds
STANDING_ANGLE_MIN = 155.0   # Degrees — considered "standing"
SQUAT_DEPTH_ANGLE  = 100.0   # Degrees — qualifies as a proper squat

# Workout targets
TARGET_REPS = 15
TARGET_SETS = 3

# Camera
CAMERA_INDEX = 0              # Change if you have multiple cameras

# Audio
AUDIO_ENABLED = True
AUDIO_COOLDOWN = 1.5          # Seconds between TTS messages

# UI toggles
SHOW_FPS            = True
SHOW_PROGRESS_BAR   = True
DRAW_LANDMARKS      = True

# Stats
SAVE_STATS_CSV = True
USER_WEIGHT_KG = 70.0         # Used for calorie estimate
```

---

## 🧠 How Squat Detection Works

### 1. Landmark extraction
MediaPipe Pose outputs 33 body landmarks.  The detector extracts:
**Hip → Knee → Ankle** for whichever leg has the highest visibility score
(supports both left and right sides automatically).

### 2. Knee angle
The interior angle at the knee is calculated using the **dot-product / cosine rule**:

```
angle = arccos( BA · BC / (|BA| · |BC|) )
```

where A = hip, B = knee, C = ankle.

### 3. Smoothing
An **Exponential Moving Average** (EMA) smoother removes per-frame jitter on landmarks,
and a **moving-average** smoother is applied to the angle before thresholding.

### 4. Finite state machine

```
STANDING ─(angle drops below 155°)──► GOING DOWN
GOING DOWN ─(angle ≤ 100°)──────────► BOTTOM
BOTTOM ─(angle rises above 120°)────► GOING UP
GOING UP ─(angle ≥ 155°)────────────► STANDING  ← rep evaluated here
```

### 5. Form validation
A rep is only counted when **all** of the following pass:

| Check | Description |
|---|---|
| Squat depth | `min_angle ≤ SQUAT_DEPTH_ANGLE` |
| Hip below knee | Hip y ≥ knee y at the bottom |
| Full extension | Angle ≥ `FULL_EXT_ANGLE` at the top |
| Back straight | Torso lean < `TORSO_LEAN_THRESHOLD` degrees |

If any check fails, a coaching message is shown and the rep is **not counted**.

---

## 📊 Session Statistics (CSV example)

```
Metric,Value
Date,2026-07-03 10:30
Total Reps,30
Sets Completed,2
Duration (s),420.3
Avg Depth (°),88.4
Fastest Rep (s),1.82
Slowest Rep (s),3.11
Calories,9.6

Rep #,Duration (s),Min Angle (°)
1,2.10,91.2
2,1.82,85.6
...
```

---

## 🖼️ Screenshots

> _Place your screenshots here after running the application._

---

## 🔮 Future Improvements

- [ ] **PushUpCounter** — elbow angle FSM
- [ ] **LungeCounter** — front-knee-angle FSM
- [ ] **CurlCounter** — wrist/elbow angle FSM
- [ ] **PlankCounter** — body-alignment timer
- [ ] **JumpingJackCounter** — limb-spread FSM
- [ ] GUI settings panel (Tkinter)
- [ ] Live rep-speed graph
- [ ] Cloud sync / leaderboard
- [ ] Multi-person tracking
- [ ] Voice-commanded workout programs

All exercise counters inherit from `ExerciseBase` in `squat_counter.py`, making extension straightforward.

---

## 📝 License

MIT — feel free to use, modify, and distribute.
