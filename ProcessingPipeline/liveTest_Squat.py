# Used for capstone enabling live interaction.
# This is not used in the pipeline but could be a good visual to understand. 

import cv2
import mediapipe as mp
import numpy as np
import time

# mediapipe 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# angle calculation for sqaut 
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# change somehow to do usb through ras camera 
# cap = cv2.VideoCapture(1) # For USB webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not detected")
    exit()


cv2.namedWindow("Squat Tracker", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Squat Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


counter = 0
stage = None
prev_time = 0


with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame error")
            break

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (960, 720))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        color = (0, 0, 255)

        # pose logic 
        if results.pose_landmarks:

            color = (0, 255, 0)

            landmarks = results.pose_landmarks.landmark

            # Right leg joints
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Angle
            angle = calculate_angle(hip, knee, ankle)

            # Visual angle text
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(knee, [960, 720]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Squat logic
            if angle > 160:
                stage = "up"

            if angle < 100 and stage == "up":
                stage = "down"
                counter += 1

            # Draw skeleton
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        else:
            cv2.putText(image, "Step into view!",
                        (300, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 3)


        # UI elements
        cv2.putText(image, "Squat Tracker", (30, 60),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (225, 255, 225), 3)

        # Squat counter box
        cv2.rectangle(image, (20, 100), (320, 220), (0, 0, 0), -1)

        cv2.putText(image, "SQUATS", (40, 140),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(image, str(counter), (40, 200),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (225, 255, 225), 4)

        cv2.putText(image, f"Stage: {stage if stage else '-'}", (180, 200),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(image, "Press Q to quit", (30, 700),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 2)

        # output 
        cv2.imshow("Squat Tracker", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
