import cv2

camera = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION) #not sure to check if this second part is needed, come back to check but it works for now.

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