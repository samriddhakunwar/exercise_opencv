"""
config.py
---------
Central configuration for the AI Fitness Tracker application.
Modify values here to change application behaviour without touching any other file.

Supported exercises: Squats, Push-ups, High Knees  (more via ExerciseBase subclasses)
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0          # Webcam device index (0 = default)
FRAME_WIDTH: int = 1280        # Capture width  (pixels)
FRAME_HEIGHT: int = 720        # Capture height (pixels)
TARGET_FPS: int = 30           # Desired capture frame rate

# ---------------------------------------------------------------------------
# Pose detection (MediaPipe)
# ---------------------------------------------------------------------------
MODEL_COMPLEXITY: int = 1           # 0 = lite, 1 = full, 2 = heavy
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.7
SMOOTH_LANDMARKS: bool = True

# ---------------------------------------------------------------------------
# Squat angle thresholds (degrees)
# ---------------------------------------------------------------------------
STANDING_ANGLE_MIN: float = 155.0   # Knee angle considered "standing"
SQUAT_DEPTH_ANGLE: float = 100.0    # Angle that qualifies as a proper squat
BAD_DEPTH_ANGLE: float = 120.0      # Angle below standing but not a good squat
FULL_EXT_ANGLE: float = 150.0       # Minimum angle to be counted as "standing"

# ---------------------------------------------------------------------------
# Form validation thresholds
# ---------------------------------------------------------------------------
HIP_KNEE_RATIO_THRESHOLD: float = 0.05   # Hip y should be >= knee y at bottom
TORSO_LEAN_THRESHOLD: float = 45.0       # Max forward lean angle (degrees)
KNEE_COLLAPSE_THRESHOLD: float = 0.08    # Normalised x-diff for knee cave

# ---------------------------------------------------------------------------
# Workout targets
# ---------------------------------------------------------------------------
TARGET_REPS: int = 15          # Reps per set
TARGET_SETS: int = 3           # Total sets
REST_BETWEEN_SETS: int = 30    # Seconds of rest between sets

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
ANGLE_SMOOTHING_WINDOW: int = 5   # Moving-average window for knee angle
LANDMARK_SMOOTH_ALPHA: float = 0.4  # EMA alpha for landmark smoothing (0-1)

# ---------------------------------------------------------------------------
# Countdown before start (seconds)
# ---------------------------------------------------------------------------
COUNTDOWN_SECONDS: int = 3

# ---------------------------------------------------------------------------
# Audio feedback
# ---------------------------------------------------------------------------
AUDIO_ENABLED: bool = True
AUDIO_COOLDOWN: float = 1.5       # Seconds between TTS announcements

# ---------------------------------------------------------------------------
# UI display toggles
# ---------------------------------------------------------------------------
DRAW_LANDMARKS: bool = True
SHOW_FPS: bool = True
SHOW_PROGRESS_BAR: bool = True
SHOW_ANGLE: bool = True
SHOW_SKELETON: bool = True

# ---------------------------------------------------------------------------
# Statistics & recording
# ---------------------------------------------------------------------------
SAVE_STATS_CSV: bool = True
STATS_DIR: str = "workouts"        # Folder created automatically
ENABLE_RECORDING: bool = False     # Set True to save session video
RECORDING_DIR: str = "recordings"
SAVE_HISTORY: bool = True          # Append each session to the history DB
HISTORY_DB: str = "data/history.db"
CHARTS_DIR: str = "data/charts"    # progress.py writes PNG charts here

# ---------------------------------------------------------------------------
# Push-up angle thresholds (degrees)  ← relaxed for easier detection
# ---------------------------------------------------------------------------
PUSHUP_TOP_ANGLE: float    = 155.0   # Elbow angle considered "arms locked out" at top  (was 165)
PUSHUP_BOTTOM_ANGLE: float = 100.0   # Elbow angle qualifying as proper depth at bottom  (was 85)
PUSHUP_LOWER_LIMIT: float  = 125.0   # Angle below top but not yet at proper bottom      (was 110)
PUSHUP_FULL_EXT_ANGLE: float = 140.0 # Minimum elbow angle to count as "arms extended"  (was 155)

# ---------------------------------------------------------------------------
# Push-up form validation thresholds  ← relaxed for easier detection
# ---------------------------------------------------------------------------
PUSHUP_HIP_SAG_THRESHOLD: float = 0.14   # Hip sag relative to shoulder-ankle line  (was 0.06)
PUSHUP_PIKE_THRESHOLD: float    = 0.16   # Excessive hip raise ("pike" position)     (was 0.08)
PUSHUP_SIDE_VIEW_THRESHOLD: float = 0.10  # Shoulder-vs-elbow depth for side-view    (was 0.12)

# ---------------------------------------------------------------------------
# Push-up workout targets
# ---------------------------------------------------------------------------
TARGET_PUSHUPS: int = 20            # Reps per set
TARGET_PUSHUP_SETS: int = 3         # Total sets

# ---------------------------------------------------------------------------
# Calories estimate
# ---------------------------------------------------------------------------
USER_WEIGHT_KG: float = 70.0       # Default user weight for calorie estimate
CALORIES_PER_SQUAT: float  = 0.32  # Approximate kcal per squat rep
CALORIES_PER_PUSHUP: float = 0.29  # Approximate kcal per push-up rep
CALORIES_PER_HIGH_KNEE: float = 0.10  # Approximate kcal per high-knee step (single lift)

# ---------------------------------------------------------------------------
# Colour palette  (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_BG_DARK    = (30,  30,  30)
COLOR_WHITE      = (255, 255, 255)
COLOR_GREEN      = (80,  200, 120)
COLOR_ORANGE     = (40,  170, 255)
COLOR_RED        = (70,  70,  220)
COLOR_YELLOW     = (60,  220, 220)
COLOR_ACCENT     = (200, 140, 60)
COLOR_SEMI_BG    = (20,  20,  20)   # Used with alpha blending

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
FONT_SCALE_LARGE: float  = 1.8
FONT_SCALE_MEDIUM: float = 1.0
FONT_SCALE_SMALL: float  = 0.65
FONT_THICKNESS: int = 2

# ---------------------------------------------------------------------------
# High Knee thresholds & targets
# ---------------------------------------------------------------------------

# Workout targets (Rep Mode)
HIGH_KNEE_TARGET_REPS: int = 20          # Left-right cycles per set
HIGH_KNEE_TARGET_SETS: int = 3           # Total sets

# Cardio mode durations (seconds) — user picks entry 0, 1, or 2
HIGH_KNEE_CARDIO_DURATIONS: list = [30, 45, 60]

# Detection thresholds
HIGH_KNEE_MIN_HEIGHT_RATIO: float = 0.72  # Knee y / hip y  (1.0 = at hip level)
HIGH_KNEE_SUPPORT_KNEE_ANGLE: float = 150.0  # Min knee angle on supporting leg
HIGH_KNEE_MIN_LIFT_DELTA: float = 0.05   # Min upward movement before "up" counts

# Pace limits (steps per minute — one knee-lift = one step)
HIGH_KNEE_MIN_SPEED: float = 30.0        # Below this → "Too Slow" (red)
HIGH_KNEE_GOOD_SPEED: float = 60.0       # Above this → "Good Pace" (green)
HIGH_KNEE_MAX_SPEED: float = 180.0       # Above this → "Slow Down!"

# Smoothing
HIGH_KNEE_SMOOTHING_WINDOW: int = 5      # Frames for height moving average

# UI toggles
SHOW_KNEE_HEIGHT_BAR: bool = True        # Show knee-height progress bar

# ---------------------------------------------------------------------------
# Plank (timed hold)
# ---------------------------------------------------------------------------
PLANK_TARGET_SECONDS: int = 30         # Hold length that completes a set
PLANK_TARGET_SETS: int = 3
PLANK_SAG_THRESHOLD: float = 0.10      # Hip below shoulder-ankle line (fraction of body length)
PLANK_PIKE_THRESHOLD: float = 0.12     # Hip above the line
PLANK_MAX_INCLINE: float = 35.0        # Max shoulder→ankle angle from horizontal (degrees)
PLANK_BREAK_GRACE: float = 1.0         # Seconds out of position before the hold ends
PLANK_MAX_FRAME_DT: float = 0.5        # Cap per-frame time step (pause / lag spikes)
CALORIES_PER_PLANK_SECOND: float = 0.07

# ---------------------------------------------------------------------------
# Jumping jacks (front view)
# ---------------------------------------------------------------------------
JJ_TARGET_REPS: int = 20
JJ_TARGET_SETS: int = 3
JJ_FEET_OPEN_RATIO: float = 1.2        # Ankle spread / shoulder width → "feet apart"
JJ_FEET_CLOSED_RATIO: float = 0.9      # Below this → "feet together" (gap = hysteresis)
JJ_ARMS_DOWN_MARGIN: float = 0.5       # Wrists this many shoulder-widths below shoulders
CALORIES_PER_JUMPING_JACK: float = 0.20

# ---------------------------------------------------------------------------
# Bicep curls (per arm)
# ---------------------------------------------------------------------------
CURL_TARGET_REPS: int = 12             # Reps per arm per set
CURL_TARGET_SETS: int = 3
CURL_EXTENDED_ANGLE: float = 150.0     # Elbow angle = arm fully extended
CURL_TOP_ANGLE: float = 55.0           # Elbow angle = fully curled
CURL_TOP_EXIT_ANGLE: float = 75.0      # Leaving the top (hysteresis)
CURL_REVERSAL_DELTA: float = 15.0      # Direction change that ends a partial rep
CURL_ELBOW_DRIFT_ANGLE: float = 35.0   # Max upper-arm angle from vertical
CURL_ELBOW_RISE_RATIO: float = 0.25    # Max elbow rise (fraction of upper-arm length)
CURL_MIN_VISIBILITY: float = 0.5       # Ignore an arm below this visibility
CALORIES_PER_CURL: float = 0.15

# ---------------------------------------------------------------------------
# Lunges (side view, per leg)
# ---------------------------------------------------------------------------
LUNGE_TARGET_REPS: int = 10            # Reps per leg per set
LUNGE_TARGET_SETS: int = 3
LUNGE_STANDING_ANGLE: float = 155.0    # Front-knee angle considered standing
LUNGE_DEPTH_ANGLE: float = 100.0       # Front-knee angle for a proper lunge
LUNGE_BOTTOM_EXIT_ANGLE: float = 120.0 # Leaving the bottom (hysteresis)
LUNGE_KNEE_TOE_TOLERANCE: float = 0.15 # Knee may pass the toe by this × shin length
LUNGE_FACING_MIN: float = 0.15         # |nose - shoulder mid| / torso length for side view
CALORIES_PER_LUNGE: float = 0.35

# ---------------------------------------------------------------------------
# Per-user calibration (C key / --calibrate)
# ---------------------------------------------------------------------------
CALIBRATION_FILE: str = "data/calibration.json"
CALIBRATION_STAND_SECONDS: float = 3.0   # Hold-still phase
CALIBRATION_REPS: int = 2                # Slow full-range reps to record
CALIBRATION_TIMEOUT: float = 30.0        # Max seconds for the reps phase

# ---------------------------------------------------------------------------
# Workout routines (--routine NAME)
# ---------------------------------------------------------------------------
ROUTINES_FILE: str = "routines.json"
