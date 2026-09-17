"""
AI-Based Virtual Assistant – Real-Time Squat Coach

Controls:
    q = Quit
    r = Reset repetitions and timer

The system detects squat posture using MediaPipe
and provides corrective feedback using voice prompts.
"""

import math
import queue
import threading
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import pyttsx3

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CAMERA_INDEX = 0

STANDING_KNEE_ANGLE = 160
SQUAT_KNEE_ANGLE = 100

# Feedback is not repeated continuously
FEEDBACK_DELAY_SECONDS = 3


# --------------------------------------------------
# MEDIAPIPE MODEL
# --------------------------------------------------

MODEL_FILE = Path(__file__).with_name("pose_landmarker_full.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


# MediaPipe Pose Landmark Indexes
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# --------------------------------------------------
# DOWNLOAD MODEL
# --------------------------------------------------

def download_pose_model_if_needed():

    if MODEL_FILE.exists():
        return True

    print("Downloading MediaPipe pose model...")

    try:
        urlretrieve(MODEL_URL, MODEL_FILE)

        print("Pose model downloaded successfully.")

        return True

    except OSError as error:

        print(f"Could not download pose model: {error}")

        return False


# --------------------------------------------------
# CALCULATE ANGLE
# --------------------------------------------------

def calculate_angle(point_a, point_b, point_c):

    ab_x = point_a.x - point_b.x
    ab_y = point_a.y - point_b.y

    cb_x = point_c.x - point_b.x
    cb_y = point_c.y - point_b.y

    dot_product = ab_x * cb_x + ab_y * cb_y

    magnitude_ab = math.sqrt(ab_x ** 2 + ab_y ** 2)
    magnitude_cb = math.sqrt(cb_x ** 2 + cb_y ** 2)

    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0

    cosine = dot_product / (magnitude_ab * magnitude_cb)

    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))


# --------------------------------------------------
# VOICE COACH
# --------------------------------------------------

class VoiceCoach:

    def __init__(self):

        self.messages = queue.Queue()

        self.last_message = ""

        self.last_spoken_time = 0

        threading.Thread(
            target=self._speak_messages,
            daemon=True
        ).start()


    def _speak_messages(self):

        engine = pyttsx3.init()

        engine.setProperty("rate", 165)

        while True:

            message = self.messages.get()

            engine.say(message)

            engine.runAndWait()


    def say(self, message):

        current_time = time.time()

        # Prevent the same message from being spoken repeatedly
        if (
            message != self.last_message
            or current_time - self.last_spoken_time
            >= FEEDBACK_DELAY_SECONDS
        ):

            self.messages.put(message)

            self.last_message = message

            self.last_spoken_time = current_time


# --------------------------------------------------
# LANDMARK VISIBILITY
# --------------------------------------------------

def landmark_visibility(landmark):

    return landmark.visibility or 0


# --------------------------------------------------
# CHOOSE BODY SIDE
# --------------------------------------------------

def choose_side(landmarks):

    left_hip = landmarks[LEFT_HIP]

    right_hip = landmarks[RIGHT_HIP]

    if landmark_visibility(left_hip) >= landmark_visibility(right_hip):

        return (
            landmarks[LEFT_SHOULDER],
            left_hip,
            landmarks[LEFT_KNEE],
            landmarks[LEFT_ANKLE],
            "left"
        )

    return (
        landmarks[RIGHT_SHOULDER],
        right_hip,
        landmarks[RIGHT_KNEE],
        landmarks[RIGHT_ANKLE],
        "right"
    )


# --------------------------------------------------
# SQUAT ASSESSMENT
# --------------------------------------------------

def assess_squat(shoulder, hip, knee, ankle):

    knee_angle = calculate_angle(
        hip,
        knee,
        ankle
    )

    hip_angle = calculate_angle(
        shoulder,
        hip,
        knee
    )


    # ----------------------------------------------
    # 1. KNEE POSITION
    # ----------------------------------------------

    if knee_angle < 60:

        message = "Keep your knees in line with your toes."


    # ----------------------------------------------
    # 2. CHEST POSITION
    # ----------------------------------------------

    elif hip_angle < 55:

        message = "Keep your chest more upright."


    # ----------------------------------------------
    # 3. CORRECT SQUAT DEPTH
    # ----------------------------------------------

    elif knee_angle <= SQUAT_KNEE_ANGLE:

        message = "Good depth. Push through your heels."


    # ----------------------------------------------
    # 4. NEED TO GO LOWER
    # ----------------------------------------------

    else:

        message = "Lower slowly into your squat."


    return knee_angle, hip_angle, message


# --------------------------------------------------
# DRAW TEXT
# --------------------------------------------------

def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.7
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


# --------------------------------------------------
# DRAW SQUAT LANDMARKS
# --------------------------------------------------

def draw_squat_landmarks(
    frame,
    shoulder,
    hip,
    knee,
    ankle
):

    height, width = frame.shape[:2]

    landmarks = [
        shoulder,
        hip,
        knee,
        ankle
    ]

    points = [
        (
            int(point.x * width),
            int(point.y * height)
        )
        for point in landmarks
    ]


    # Draw lines
    for first, second in zip(
        points,
        points[1:]
    ):

        cv2.line(
            frame,
            first,
            second,
            (0, 255, 0),
            3
        )


    # Draw points
    for point in points:

        cv2.circle(
            frame,
            point,
            6,
            (0, 0, 255),
            -1
        )


# --------------------------------------------------
# CREATE MEDIAPIPE POSE DETECTOR
# --------------------------------------------------

def create_pose_detector():

    options = vision.PoseLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(MODEL_FILE)
        ),

        running_mode=vision.RunningMode.VIDEO,

        num_poses=1,

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    return vision.PoseLandmarker.create_from_options(
        options
    )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    # Download model if required
    if not download_pose_model_if_needed():

        return


    # ----------------------------------------------
    # OPEN WEBCAM
    # ----------------------------------------------

    camera = cv2.VideoCapture(CAMERA_INDEX)


    if not camera.isOpened():

        print(
            "Could not open camera. "
            "Try changing CAMERA_INDEX to 1."
        )

        return


    # ----------------------------------------------
    # INITIALIZE VOICE ASSISTANT
    # ----------------------------------------------

    coach = VoiceCoach()


    # ----------------------------------------------
    # INITIAL VALUES
    # ----------------------------------------------

    repetitions = 0

    squat_stage = "up"

    workout_start_time = time.time()


    # ----------------------------------------------
    # START MEDIAPIPE
    # ----------------------------------------------

    with create_pose_detector() as pose_detector:

        while camera.isOpened():

            success, frame = camera.read()


            if not success:

                print("Could not read video frame.")

                break


            # Mirror camera
            frame = cv2.flip(frame, 1)


            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # Create MediaPipe image
            media_pipe_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb_frame
            )


            # Timestamp
            timestamp_ms = int(
                time.monotonic() * 1000
            )


            # Detect pose
            detection_result = pose_detector.detect_for_video(

                media_pipe_image,

                timestamp_ms
            )


            # ------------------------------------------
            # DEFAULT FEEDBACK
            # ------------------------------------------

            feedback = (
                "Step sideways so your full body is visible."
            )

            feedback_color = (0, 180, 255)


            # ------------------------------------------
            # IF PERSON IS DETECTED
            # ------------------------------------------

            if detection_result.pose_landmarks:

                landmarks = detection_result.pose_landmarks[0]


                # Select clearer body side
                (
                    shoulder,
                    hip,
                    knee,
                    ankle,
                    side
                ) = choose_side(landmarks)


                # Calculate squat angles
                (
                    knee_angle,
                    hip_angle,
                    feedback
                ) = assess_squat(
                    shoulder,
                    hip,
                    knee,
                    ankle
                )


                # --------------------------------------
                # REP COUNTING
                # --------------------------------------

                # Person reached squat position
                if knee_angle <= SQUAT_KNEE_ANGLE:

                    squat_stage = "down"


                # Person returned to standing
                elif (
                    knee_angle >= STANDING_KNEE_ANGLE
                    and squat_stage == "down"
                ):

                    squat_stage = "up"

                    repetitions += 1


                    # Voice feedback after successful rep
                    feedback = (
                        f"Great squat. "
                        f"Rep {repetitions} completed."
                    )

                    feedback_color = (0, 255, 0)


                    # Speak successful rep
                    coach.say(feedback)


                # --------------------------------------
                # NORMAL FEEDBACK
                # --------------------------------------

                else:

                    if "Good" in feedback:

                        feedback_color = (0, 255, 0)

                    elif "Great" in feedback:

                        feedback_color = (0, 255, 0)


                    # Voice corrective feedback
                    coach.say(feedback)


                # Draw skeleton
                draw_squat_landmarks(
                    frame,
                    shoulder,
                    hip,
                    knee,
                    ankle
                )


                # Display angles
                draw_text(
                    frame,
                    f"{side.title()} Knee: "
                    f"{int(knee_angle)} degrees",
                    (15, 105)
                )


                draw_text(
                    frame,
                    f"Hip: "
                    f"{int(hip_angle)} degrees",
                    (15, 135)
                )


            # ------------------------------------------
            # TIMER
            # ------------------------------------------

            elapsed_seconds = int(
                time.time() - workout_start_time
            )

            minutes, seconds = divmod(
                elapsed_seconds,
                60
            )


            # ------------------------------------------
            # DISPLAY INFORMATION
            # ------------------------------------------

            draw_text(
                frame,
                f"Reps: {repetitions}",
                (15, 35),
                (0, 255, 0),
                1
            )


            draw_text(
                frame,
                f"Time: {minutes:02}:{seconds:02}",
                (15, 70),
                (0, 255, 0),
                1
            )


            draw_text(
                frame,
                f"Stage: {squat_stage}",
                (15, 165)
            )


            draw_text(
                frame,
                feedback,
                (15, frame.shape[0] - 25),
                feedback_color,
                0.6
            )


            draw_text(
                frame,
                "q = quit | r = reset",
                (15, frame.shape[0] - 55),
                (200, 200, 200),
                0.55
            )


            # ------------------------------------------
            # SHOW CAMERA
            # ------------------------------------------

            cv2.imshow(
                "AI Squat Virtual Assistant",
                frame
            )


            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF


            # Quit
            if key == ord("q"):

                break


            # Reset
            if key == ord("r"):

                repetitions = 0

                squat_stage = "up"

                workout_start_time = time.time()

                coach.say(
                    "Counter and timer reset."
                )


    # ----------------------------------------------
    # CLOSE CAMERA
    # ----------------------------------------------

    camera.release()

    cv2.destroyAllWindows()


# --------------------------------------------------
# START PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    main()