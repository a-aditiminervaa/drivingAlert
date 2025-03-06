##to detect drowsiness and yawning and send an alert message
##specially for truck drivers for late night driving
##can be connected to external module to raise such alerts that driver gets attentive

import cv2
import dlib
import numpy as np
from scipy.spatial import distance


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar



EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
FRAME_COUNT = 20

counter = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)


        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [36, 37, 38, 39, 40, 41]])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [42, 43, 44, 45, 46, 47]])


        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])


        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        EAR = (left_EAR + right_EAR) / 2.0
        MAR = mouth_aspect_ratio(mouth)


        if EAR < EAR_THRESHOLD:
            counter += 1
            if counter >= FRAME_COUNT:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            counter = 0


        if MAR > MAR_THRESHOLD:
            cv2.putText(frame, "YAWNING ALERT!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
