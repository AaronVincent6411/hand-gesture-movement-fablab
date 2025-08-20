import cv2
import mediapipe as mp
import numpy as np

# -------------------------------
# 2D Kalman Filter
# -------------------------------
class Kalman2D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not self.initialized:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True

        pred = self.kalman.predict()
        correct = self.kalman.correct(measurement)

        return correct[0,0], correct[1,0], correct[2,0], correct[3,0]


# -------------------------------
# Mediapipe & Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

kf = Kalman2D()

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    DIRECTION = "no movement"

    if result.multi_hand_landmarks:

        if len(result.multi_hand_landmarks) == 1:

            hand = result.multi_hand_landmarks[0]

            # ---------------------------------------------------
            # COMPUTE CENTER OF HAND (finger wiggle proof)
            # ---------------------------------------------------
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            center_x = int(np.mean(xs) * w)
            center_y = int(np.mean(ys) * h)

            # Smooth center + estimate velocity
            x, y, vx, vy = kf.update(center_x, center_y)

            # Dynamic threshold based on image size
            THRESH = min(w, h) * 0.1   # 10% threshold

            # ---------------------------------------------------
            # REAL WAVE DETECTION (finger wiggle won't trigger)
            # ---------------------------------------------------
            if abs(vx) < THRESH and abs(vy) < THRESH:
                DIRECTION = "no movement"
            else:
                # horizontal wave
                if abs(vx) > abs(vy):
                    DIRECTION = "right" if vx > 0 else "left"
                # vertical wave
                else:
                    DIRECTION = "down" if vy > 0 else "up"

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, DIRECTION, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

    cv2.imshow("Anti-Finger-Wiggle Hand Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
