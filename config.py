"""
config.py
---------
Central configuration for the Squat Tracker application.
Modify values here to change application behaviour without touching any other file.
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

# ---------------------------------------------------------------------------
# Calories estimate
# ---------------------------------------------------------------------------
USER_WEIGHT_KG: float = 70.0       # Default user weight for calorie estimate
CALORIES_PER_SQUAT: float = 0.32   # Approximate kcal per squat rep

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
