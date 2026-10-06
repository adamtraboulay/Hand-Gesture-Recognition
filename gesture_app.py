from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = (
    Path(__file__).resolve().parent
    / "models"
    / "gesture_recognizer.task"
)
THRESHOLD = 0.60

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(p.x * width), int(p.y * height))
        for p in landmarks
    ]

    for start, end in EDGES:
        cv2.line(
            frame, points[start], points[end],
            (100, 220, 100), 2,
        )

    for point in points:
        cv2.circle(frame, point, 4, (0, 200, 255), -1)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f"Download the model first: {MODEL}")

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    camera = None

    try:
        with vision.GestureRecognizer.create_from_options(options) as model:
            camera = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

            if not camera.isOpened():
                raise RuntimeError("Could not open webcam.")

            print("Waiting for webcam frames...")
            last_good_frame = time.monotonic()
            previous_ms = -1
            previous_time = None

            while True:
                success, frame = camera.read()

                if not success or frame is None:
                    if time.monotonic() - last_good_frame > 5:
                        raise RuntimeError(
                            "No webcam frames for 5 seconds. "
                            "Close other camera apps and check camera access."
                        )
                    time.sleep(0.05)
                    continue

                last_good_frame = time.monotonic()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = max(
                    previous_ms + 1,
                    int(time.monotonic() * 1000),
                )
                previous_ms = timestamp_ms

                result = model.recognize_for_video(
                    image, timestamp_ms
                )

                label = "No hand"

                if result.hand_landmarks:
                    draw_hand(frame, result.hand_landmarks[0])
                    label = "Unknown"

                    if result.gestures and result.gestures[0]:
                        category = result.gestures[0][0]

                        if (
                            category.category_name != "None"
                            and category.score >= THRESHOLD
                        ):
                            name = category.category_name.replace("_", " ")
                            label = f"{name} ({category.score:.0%})"

                now = time.monotonic()
                fps = (
                    1.0 / max(now - previous_time, 0.000001)
                    if previous_time is not None
                    else 0.0
                )
                previous_time = now

                cv2.putText(
                    frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 150), 2,
                )
                cv2.putText(
                    frame, f"FPS: {fps:.1f} | q: quit", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2,
                )

                cv2.imshow("Hand Gesture Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()