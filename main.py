import cv2
import mediapipe as mp
import numpy as np
import math

# -------------------------------
# 2D Kalman Filter (Kept for smooth motion detection)
# -------------------------------
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

# -------------------------------
# Mediapipe & Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

kf = Kalman2D()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    DIRECTION = "no movement"
    
    # -----------------------------------------------------------
    # OUTPUT VARIABLES FOR HEXAGONAL GRID
    # -----------------------------------------------------------
    ball_height_px = 0
    angle_lr = 0   # Left/Right (Roll)
    angle_fb = 0   # Forward/Backward (Pitch)

    if result.multi_hand_landmarks:
        if len(result.multi_hand_landmarks) == 1:
            hand = result.multi_hand_landmarks[0]

            # Landmarks: 0=Wrist, 9=Middle Finger MCP (Knuckle)
            wrist = hand.landmark[0]
            middle_mcp = hand.landmark[9]

            # Convert X, Y to Pixels
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
            
            # Convert Z to approximate Pixels (MediaPipe Z is relative scale similar to X)
            # We multiply by 'w' to make it comparable to X/Y for math
            wz = wrist.z * w
            mz = middle_mcp.z * w

            # ---------------------------------------------------
            # 1. HEIGHT (Floor to Wrist)
            # ---------------------------------------------------
            ball_height_px = max(0, h - wy)

            # ---------------------------------------------------
            # 2. LEFT / RIGHT ANGLE (Roll)
            # ---------------------------------------------------
            delta_x = mx - wx
            delta_y = wy - my # Invert Y so Up is Positive
            
            # Angle on the 2D Screen Plane
            angle_lr = int(math.degrees(math.atan2(delta_x, delta_y)))
            
            # Clamp for stability
            angle_lr = max(-90, min(90, angle_lr))

            # ---------------------------------------------------
            # 3. FORWARD / BACKWARD ANGLE (Pitch)
            # ---------------------------------------------------
            # Delta Z: How much closer/further is the knuckle compared to wrist?
            # Note: Negative Z in MediaPipe = Closer to camera.
            # If mz < wz: Knuckle closer than wrist -> Tilted Forward
            
            delta_z = wz - mz  # Positive value means Forward tilt
            
            # We compute angle against the Vertical Y axis
            # This estimates how much we are leaning into the Z-plane
            angle_fb = int(math.degrees(math.atan2(delta_z, delta_y)))
            
            # Clamp for stability
            angle_fb = max(-90, min(90, angle_fb))

            # ---------------------------------------------------
            # VISUALIZATION
            # ---------------------------------------------------
            # Draw Skeleton
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            # Draw Height Line
            cv2.line(frame, (wx, wy), (wx, h), (0, 255, 255), 2)
            
            # Draw Left/Right Tilt Line
            cv2.line(frame, (wx, wy), (mx, my), (0, 255, 0), 3)

            # ---------------------------------------------------
            # 4. MOTION DETECTION (Kalman)
            # ---------------------------------------------------
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            cx, cy = int(np.mean(xs) * w), int(np.mean(ys) * h)

            x, y, vx, vy = kf.update(cx, cy)
            THRESH = min(w, h) * 0.1

            if abs(vx) < THRESH and abs(vy) < THRESH:
                DIRECTION = "Stationary"
            else:
                if abs(vx) > abs(vy):
                    DIRECTION = "Right" if vx > 0 else "Left"
                else:
                    DIRECTION = "Down" if vy > 0 else "Up"

    # ---------------------------------------------------
    # UI DASHBOARD
    # ---------------------------------------------------
    # Box Background for UI
    cv2.rectangle(frame, (0, h-160), (300, h), (0, 0, 0), -1)
    
    # 1. Height
    cv2.putText(frame, f"Height: {ball_height_px}", (20, h - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # 2. L/R Angle
    col_lr = (0, 255, 0) if abs(angle_lr) < 15 else (0, 165, 255)
    cv2.putText(frame, f"L/R Angle: {angle_lr} deg", (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_lr, 2)

    # 3. F/B Angle
    col_fb = (0, 255, 0) if abs(angle_fb) < 15 else (255, 0, 255) # Purple if tilted
    fb_text = "Flat"
    if angle_fb > 15: fb_text = "Forward"
    if angle_fb < -15: fb_text = "Backward"
    
    cv2.putText(frame, f"F/B Angle: {angle_fb} deg ({fb_text})", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_fb, 2)

    # 4. Swipe Detection
    cv2.putText(frame, f"Swipe: {DIRECTION}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imshow("Hexagonal Grid Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()