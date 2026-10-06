import cv2

camera = cv2.VideoCapture(0)

try: 
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a webcam frame")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()