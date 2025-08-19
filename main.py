import cv2
import mediapipe as mp
import numpy as np
import math

class Kalman2D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
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

MAX_PHYSICAL_HEIGHT_MM = 2200  
HEIGHT_STABILITY_THRESHOLD = 40

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

kf = Kalman2D()

cap = cv2.VideoCapture(0)

stable_height_mm = 0
prev_raw_height = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    
    pitch_angle = 0  
    roll_angle = 0   
    fwd_bwd_state = "Neutral"
    
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            
            hand_label = result.multi_handedness[idx].classification[0].label
            
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]   
            middle_mcp = hand_landmarks.landmark[9]  
            pinky_mcp = hand_landmarks.landmark[17]  

            wx, wy = int(wrist.x * w), int(wrist.y * h)
            ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)
            mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
            px, py = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

            pixel_height = max(0, h - wy)
            
            raw_height_mm = int((pixel_height / h) * MAX_PHYSICAL_HEIGHT_MM)
            raw_height_mm = max(0, min(MAX_PHYSICAL_HEIGHT_MM, raw_height_mm))
            
            diff = abs(raw_height_mm - stable_height_mm)
            
            if diff > HEIGHT_STABILITY_THRESHOLD:
                stable_height_mm = raw_height_mm
            
            pitch_dy = wy - my
            pitch_dx = mx - wx
            pitch_angle = int(math.degrees(math.atan2(pitch_dy, abs(pitch_dx) + 1)))

            roll_dy = py - iy
            roll_dx = px - ix
            roll_angle = int(math.degrees(math.atan2(roll_dy, roll_dx)))

            is_palm_facing = False
            if hand_label == "Right":
                if ix < px: is_palm_facing = True
            else:
                if ix > px: is_palm_facing = True
            
            if is_palm_facing:
                fwd_bwd_state = "FORWARD"
                color_fb = (0, 255, 0)
            else:
                fwd_bwd_state = "BACKWARD"
                color_fb = (0, 0, 255)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.line(frame, (wx, wy), (mx, my), (0, 255, 255), 2)
            cv2.line(frame, (ix, iy), (px, py), (255, 0, 0), 2)
            
            cv2.line(frame, (wx, wy), (wx, h), (100, 100, 100), 1) 
            
            stable_y_px = h - int((stable_height_mm / MAX_PHYSICAL_HEIGHT_MM) * h)
            cv2.circle(frame, (wx, stable_y_px), 8, (0, 255, 255), -1)

    else:
        stable_height_mm=0

    cv2.rectangle(frame, (0, h-180), (350, h), (20, 20, 20), -1)
    
    cv2.putText(frame, f"Height: {stable_height_mm} mm", (15, h - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    cv2.putText(frame, f"Pitch: {pitch_angle} deg", (15, h - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Roll: {roll_angle} deg", (15, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

    cv2.imshow("Hexagonal Grid Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()