"""
Smart Hand Mouse — control the Windows mouse with hand gestures via webcam.
Compatible with MediaPipe 0.10+ (Tasks API) and Python 3.12+.

HOW IT WORKS (high level)
──────────────────────────
1. The webcam captures frames in real time.
2. MediaPipe detects 21 landmarks (key points) on the hand skeleton.
3. We read the position of those landmarks to classify the hand shape
   into one of several gestures.
4. Each gesture maps to a mouse action sent through pyautogui.

GESTURE REFERENCE
─────────────────
  🤏 Pinch shape (relaxed) — thumb and index curved toward each other
                              but NOT touching (distance > PINCH_THRESHOLD)
                            → moves the cursor

  🤏 Full pinch            — thumb tip touches index tip
                              (distance < PINCH_THRESHOLD = 0.07 of frame width)
                            → left click (fires once per contact, no repeat)

  🖐️ Three fingers up       — index + middle + ring extended, thumb & pinky down
                            → right click

  ✊ Closed fist            — all fingers folded
                            → scroll  (hand high on screen = scroll up,
                                       hand low on screen = scroll down)

  ✌️ V / Peace sign         — index + middle up, others down
                            → toggle the system ON (green) / PAUSED (red)

KEYBOARD
────────
  Q — quit

TUNING CONSTANTS (top of file)
───────────────────────────────
  SMOOTHING        exponential filter strength for cursor movement (0 = raw, 1 = frozen)
  PINCH_THRESHOLD  how close thumb and index must be to count as a pinch (0–1, normalized)
  ACTIVE_ZONE      fraction of the camera frame used as the mouse surface (0.7 = central 70%)
  CLICK_COOLDOWN   minimum seconds between two consecutive clicks (anti-bounce)
"""
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui


# ──────────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# Changing these is the first thing to try when the app feels too sensitive,
# too slow, or too jumpy.
# ──────────────────────────────────────────────────────────────────────────────

# Exponential moving average weight for cursor smoothing.
# Formula: new_pos = SMOOTHING * old_pos + (1 - SMOOTHING) * raw_pos
# 0.0 = no filter (instant but shaky), 0.5 = heavy smoothing (laggy but stable).
SMOOTHING = 0.3

# Maximum normalized distance between thumb tip (landmark 4) and index tip
# (landmark 8) that counts as a "full pinch" → triggers a click.
# Coordinates are in [0, 1] relative to the frame dimensions.
# Increase if clicks are hard to trigger; decrease if they fire accidentally.
PINCH_THRESHOLD = 0.07

# Central fraction of the camera image that maps to the full screen.
# 0.7 means the outer 15% on each side is ignored: you don't need to move
# your hand to the very edge of the frame to reach a screen corner.
ACTIVE_ZONE = 0.7

# Minimum time (in seconds) between two click events.
# Prevents a single physical gesture from registering as dozens of clicks.
CLICK_COOLDOWN = 0.4

# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# MediaPipe 0.10+ uses an external .task file instead of a bundled model.
# The file is downloaded automatically on first run (~29 MB).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ──────────────────────────────────────────────────────────────────────────────
# SCREEN SIZE & SAFETY
# ──────────────────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()
# Disable pyautogui's built-in failsafe (moving the mouse to the top-left
# corner would otherwise raise an exception and crash the app).
pyautogui.FAILSAFE = False

# ──────────────────────────────────────────────────────────────────────────────
# HAND SKELETON CONNECTIONS
# MediaPipe labels 21 landmarks (0–20) on the hand.
# These pairs define the bones we draw as green lines on the video feed.
#
# Landmark map (simplified):
#   0 = wrist
#   1-4  = thumb  (1=base … 4=tip)
#   5-8  = index  (5=base … 8=tip)
#   9-12 = middle (9=base … 12=tip)
#   13-16 = ring  (13=base … 16=tip)
#   17-20 = pinky (17=base … 20=tip)
# ──────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm knuckles
]

# ──────────────────────────────────────────────────────────────────────────────
# RUNTIME STATE
# A single dict keeps mutable state out of global scope.
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "enabled":     True,             # False = paused, no mouse actions
    "last_click":  0.0,              # timestamp of the last click (for cooldown)
    "last_x":      SCREEN_W // 2,   # previous smoothed cursor X
    "last_y":      SCREEN_H // 2,   # previous smoothed cursor Y
    "is_pinching": False,            # True while pinch contact is held (prevents repeat clicks)
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
def ensure_model():
    """Download the MediaPipe hand landmarker model file if it is missing.

    The model is a binary TFLite bundle (~29 MB). It is saved next to this
    script and reused on every subsequent run.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL} …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Could not download the model: {e}\n"
                f"Download it manually from:\n  {MODEL_URL}\n"
                f"and save it as '{MODEL_PATH}' in the current directory."
            ) from e


# ──────────────────────────────────────────────────────────────────────────────
# LANDMARK HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def fingers_up(landmarks):
    """Return a 5-element list [thumb, index, middle, ring, pinky] where
    1 means the finger is extended and 0 means it is folded.

    HOW THE DETECTION WORKS
    ───────────────────────
    MediaPipe gives each landmark a (x, y) position normalized to [0, 1],
    where (0, 0) is the top-left of the frame.

    For fingers 2–5 (index to pinky):
      A finger is "up" when its TIP (highest knuckle) is ABOVE its PIP
      (middle knuckle) in the image, i.e. tip.y < pip.y  (y grows downward).

    For the thumb:
      The thumb opens sideways, not vertically, so the y-comparison breaks.
      Instead we compare x: on a mirrored (selfie-view) image, the right-hand
      thumb tip (landmark 4) is to the LEFT of its lower joint (landmark 3)
      when the thumb is extended.  tip.x < joint.x → thumb is open.
    """
    f = []
    # Thumb: horizontal comparison (works because the frame is already flipped)
    f.append(1 if landmarks[4].x < landmarks[3].x else 0)
    # Index, middle, ring, pinky: vertical comparison (tip vs. PIP knuckle)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        f.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return f


def pinch_distance(landmarks):
    """Return the Euclidean distance between the thumb tip (4) and the
    index tip (8) in normalized coordinates [0, 1].

    This is the core metric for both cursor control and click detection:
      distance > PINCH_THRESHOLD  →  relaxed pinch shape  →  cursor mode
      distance < PINCH_THRESHOLD  →  full pinch contact   →  left click
    """
    a, b = landmarks[4], landmarks[8]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def map_to_screen(landmark):
    """Convert a normalized landmark position (0–1) to a screen pixel coordinate,
    mapping only the central ACTIVE_ZONE of the frame to the full screen.

    WHY AN ACTIVE ZONE?
    ───────────────────
    Without it, reaching a screen corner would require moving your hand to the
    extreme edge of the camera's field of view, which is awkward and imprecise.
    By shrinking the usable camera area to the central 70%, small hand movements
    cover the entire screen comfortably.

    The math:
      margin = (1 - 0.7) / 2 = 0.15   (15% ignored on each side)
      nx = clamp((raw_x - margin) / ACTIVE_ZONE, 0, 1)
      screen_x = nx * SCREEN_W
    """
    margin = (1 - ACTIVE_ZONE) / 2
    nx = max(0.0, min(1.0, (landmark.x - margin) / ACTIVE_ZONE))
    ny = max(0.0, min(1.0, (landmark.y - margin) / ACTIVE_ZONE))
    return int(nx * SCREEN_W), int(ny * SCREEN_H)


# ──────────────────────────────────────────────────────────────────────────────
# MOUSE ACTIONS
# ──────────────────────────────────────────────────────────────────────────────
def can_click():
    """Return True if enough time has passed since the last click.

    This is a simple debounce: without it, a single pinch lasting several
    frames would fire dozens of clicks instead of one.
    """
    return time.time() - state["last_click"] > CLICK_COOLDOWN


def do_click(button="left"):
    """Send a mouse click if the cooldown has expired, then reset the timer."""
    if not can_click():
        return
    pyautogui.click(button=button)
    state["last_click"] = time.time()


def move_cursor(target_x, target_y):
    """Move the cursor to (target_x, target_y) with exponential smoothing.

    WHY SMOOTH?
    ───────────
    Hand tracking is noisy: even a perfectly still hand produces jittery
    landmark coordinates frame-to-frame. Without filtering, the cursor shakes
    constantly. The EMA (Exponential Moving Average) blends the previous
    position with the new target:

        smoothed = SMOOTHING * prev + (1 - SMOOTHING) * target

    SMOOTHING = 0.3 means 30% old position + 70% new → slight lag, no jitter.
    """
    sx = int(SMOOTHING * state["last_x"] + (1 - SMOOTHING) * target_x)
    sy = int(SMOOTHING * state["last_y"] + (1 - SMOOTHING) * target_y)
    pyautogui.moveTo(sx, sy)
    state["last_x"], state["last_y"] = sx, sy


# ──────────────────────────────────────────────────────────────────────────────
# GESTURE RECOGNITION
# ──────────────────────────────────────────────────────────────────────────────
def handle_gesture(landmarks):
    """Classify the current hand shape and execute the corresponding action.

    RECOGNITION ORDER MATTERS
    ─────────────────────────
    Gestures are checked from most specific to most general so that they
    cannot accidentally shadow each other.  In particular, the full-pinch
    check MUST come before the cursor check: when pinching, the index finger
    is slightly raised (f[1] == 1), so without the pinch-first rule the
    cursor branch would always fire and the click would never register.

    Returns a short string label used by the on-screen overlay.
    """
    f = fingers_up(landmarks)

    # ── ✌️ V / Peace sign ──────────────────────────────────────────────────────
    # Index and middle up, all others down.
    # Toggles the entire system on/off so you can move your hand freely
    # without accidentally triggering mouse events.
    if f == [0, 1, 1, 0, 0]:
        if can_click():
            state["enabled"] = not state["enabled"]
            state["last_click"] = time.time()
        return "ON/OFF"

    # Everything below this line is skipped when paused.
    if not state["enabled"]:
        return "(pause)"

    # ── ✊ Closed fist → scroll ────────────────────────────────────────────────
    # Checked FIRST: a tight fist brings thumb and index very close together,
    # which would otherwise trigger the pinch/click branch below.
    # Four fingers folded (thumb ignored — its horizontal check is unreliable
    # in a fist). Scroll direction is determined by the vertical position of
    # landmark 9 (middle-finger base / palm center):
    #   hand held HIGH (y < 0.45) → scroll UP
    #   hand held LOW  (y > 0.55) → scroll DOWN
    if f[1] == 0 and f[2] == 0 and f[3] == 0 and f[4] == 0:
        state["is_pinching"] = False  # prevent a ghost click when releasing the fist
        y = landmarks[9].y
        if y < 0.45:
            pyautogui.scroll(15)
        elif y > 0.55:
            pyautogui.scroll(-15)
        return "SCROLL"

    # ── 🤏 Full pinch → left click ─────────────────────────────────────────────
    # Thumb tip and index tip are within PINCH_THRESHOLD of each other.
    # The is_pinching flag converts the continuous proximity into a single
    # edge-triggered click (fires on contact, not on every frame).
    if pinch_distance(landmarks) < PINCH_THRESHOLD:
        if not state["is_pinching"]:
            do_click("left")
            state["is_pinching"] = True
        return "CLIC"
    else:
        # Reset so the next pinch contact fires a new click.
        state["is_pinching"] = False

    # ── 🤏 Relaxed pinch shape → move cursor ───────────────────────────────────
    # Index finger raised, middle/ring/pinky folded, thumb free (ignored).
    # This is the natural "pinch" grip before fully closing the fingers.
    # We track the index TIP (landmark 8) as the cursor control point.
    # Thumb state is intentionally ignored: holding the thumb still while
    # moving only the index is unnatural and unreliable.
    if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0:
        tx, ty = map_to_screen(landmarks[8])
        move_cursor(tx, ty)
        return "Curseur"

    # ── 🖐️ Three fingers → right click ────────────────────────────────────────
    # Index, middle, and ring extended; thumb and pinky folded.
    if f == [0, 1, 1, 1, 0]:
        do_click("right")
        return "CLIC DROIT"

    return "—"


# ──────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks):
    """Overlay the hand skeleton on the video frame.

    MediaPipe 0.10+ removed the built-in drawing_utils, so we draw manually:
      1. Convert normalized (0–1) landmark coordinates to pixel coordinates.
      2. Draw a green line for each bone defined in HAND_CONNECTIONS.
      3. Draw a white dot at each of the 21 joint positions.
    """
    h, w = frame.shape[:2]
    # Pre-compute all 21 pixel positions once (cheaper than converting inside the loops)
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def draw_overlay(frame, gesture, fps):
    """Print status text and the active-zone rectangle on the frame.

    The white rectangle shows the region of the camera that maps to the screen.
    Moving your hand outside it won't move the cursor further — it's a visual
    reminder of the ACTIVE_ZONE boundary.
    """
    status_color = (0, 255, 0) if state["enabled"] else (0, 0, 255)
    label = "ON" if state["enabled"] else "PAUSE"
    cv2.putText(frame, f"Status: {label}",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Draw the active zone rectangle
    h, w = frame.shape[:2]
    m = (1 - ACTIVE_ZONE) / 2
    cv2.rectangle(frame,
                  (int(w * m), int(h * m)),
                  (int(w * (1 - m)), int(h * (1 - m))),
                  (255, 255, 255), 1)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Initialize the detector and webcam, then run the capture/inference loop.

    MEDIAPIPE TASKS API OVERVIEW
    ────────────────────────────
    Unlike the old solutions API (mp.solutions.hands), the Tasks API requires:
      1. A separate .task model file on disk.
      2. A BaseOptions object pointing to that file.
      3. A task-specific Options object (HandLandmarkerOptions here).
      4. Calling create_from_options() to get the detector instance.

    Running mode VIDEO (vs. IMAGE or LIVE_STREAM):
      - detect_for_video() must receive a monotonically increasing timestamp
        (milliseconds). We use the wall-clock time for this.
      - VIDEO mode applies temporal smoothing between frames internally,
        which improves landmark stability compared to per-image detection.
    """
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,                        # track one hand only (faster)
        min_hand_detection_confidence=0.7,  # how certain before a hand is reported
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # On Windows, DirectShow avoids the multi-second MSMF startup delay.
    # On other platforms, fall back to the default backend.
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Cannot open the webcam.")

    print("Smart Hand Mouse started.")
    print("  Q          — quit")
    print("  V / Peace  — toggle pause")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Flip horizontally so the image acts as a mirror.
        # This makes left/right gestures feel natural and keeps landmark
        # coordinates consistent with the user's perspective.
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB; OpenCV reads BGR by default.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the numpy array in a MediaPipe Image object.
        # SRGB is the standard color space for webcam footage.
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        gesture = "—"
        if result.hand_landmarks:
            # result.hand_landmarks is a list of hands; we requested max 1.
            # Each element is a list of 21 NormalizedLandmark objects.
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)
            gesture = handle_gesture(landmarks)

        # FPS counter: measure wall-clock time between consecutive frames.
        now = time.time()
        fps = 1 / max(0.001, now - prev_time)
        prev_time = now

        draw_overlay(frame, gesture, fps)
        cv2.imshow("Smart Hand Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()  # frees the TFLite interpreter and GPU resources


if __name__ == "__main__":
    main()
