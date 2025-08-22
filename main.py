import cv2
import mediapipe as mp
import numpy as np
import math

# --- Configuration Constants ---
MAX_Z_HEIGHT_MM = 2200      # Bottom of screen = 2200, Top = 0
MAX_RADIUS = 10000          # Flat plane
MIN_CURVE_RADIUS = 700      # Minimum radius before snapping to sphere
FIST_THRESHOLD_RATIO = 1.3  # Ratio of Wrist-Tip vs Wrist-MCP to detect fist

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

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def video_capture_coordinates():
    cap = cv2.VideoCapture(0)

    current_z_mm = 0
    current_radius = 10000

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
        mode_status = "FLAT / CURVE"
        color_status = (0, 255, 0)
        
        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                
                # Extract landmarks
                wrist = hand_landmarks.landmark[0]
                index_mcp = hand_landmarks.landmark[5]   
                middle_mcp = hand_landmarks.landmark[9]
                middle_tip = hand_landmarks.landmark[12] # Used for open/close detection
                pinky_mcp = hand_landmarks.landmark[17]
                
                # Convert to pixels
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)
                mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
                tx, ty = int(middle_tip.x * w), int(middle_tip.y * h)
                px, py = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

                # --- 1. Z-Height Calculation (0 to 2200) ---
                # Top of screen (y=0) -> 0mm
                # Bottom of screen (y=h) -> 2200mm
                current_z_mm = int((wy / h) * MAX_Z_HEIGHT_MM)
                current_z_mm = max(0, min(MAX_Z_HEIGHT_MM, current_z_mm))

                # --- 2. Calculate Hand Openness (Radius) ---
                # We compare distance of Wrist->Tip vs Wrist->MCP (Knuckle) to handle depth changes
                dist_wrist_tip = calculate_distance((wx, wy), (tx, ty))
                dist_wrist_mcp = calculate_distance((wx, wy), (mx, my))
                
                # Ratio: ~2.0 is Open Hand, ~1.0 is Fist
                ratio = dist_wrist_tip / (dist_wrist_mcp + 1e-6) 

                # Map Ratio to Radius (1.3 to 2.2 maps to 700 to 10000)
                if ratio < FIST_THRESHOLD_RATIO:
                    # --- FIST DETECTED (Sphere Mode) ---
                    current_radius = 0
                    pitch_angle = 0
                    roll_angle = 0
                    mode_status = "SPHERE (FIST)"
                    color_status = (0, 0, 255)
                else:
                    # --- OPEN HAND (Curve/Flat Mode) ---
                    # Normalize ratio to range 0.0 to 1.0 (between 1.3 and 2.2)
                    norm_openness = (ratio - FIST_THRESHOLD_RATIO) / (2.2 - FIST_THRESHOLD_RATIO)
                    norm_openness = max(0, min(1, norm_openness))
                    
                    # Lerp between 700 and 10000
                    current_radius = 10000

                    # --- Calculate Angles Only when Open ---
                    # Pitch
                    pitch_dy = wy - my
                    pitch_dx = mx - wx
                    pitch_angle = int(math.degrees(math.atan2(pitch_dy, abs(pitch_dx) + 1)))
                    pitch_angle = max(-60, min(60, pitch_angle))

                    # Roll
                    roll_dy = py - iy
                    roll_dx = px - ix
                    roll_angle = int(math.degrees(math.atan2(roll_dy, roll_dx)))
                    roll_angle = max(-60, min(60, roll_angle))

                # Visuals
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw Wrist Line for Height
                cv2.line(frame, (0, wy), (w, wy), (50, 50, 50), 1)
                cv2.circle(frame, (wx, wy), 8, color_status, -1)

        # --- Dashboard UI ---
        cv2.rectangle(frame, (0, h-220), (400, h), (20, 20, 20), -1)
        
        # Z Height
        cv2.putText(frame, f"Z Height: {current_z_mm}", (15, h - 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Radius
        cv2.putText(frame, f"Radius: {current_radius}", (15, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
        # Pitch
        color_p = (100, 100, 100) if current_radius == 0 else (0, 255, 255)
        cv2.putText(frame, f"Pitch: {pitch_angle}", (15, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_p, 2)

        # Roll
        color_r = (100, 100, 100) if current_radius == 0 else (255, 100, 100)
        cv2.putText(frame, f"Roll: {roll_angle}", (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_r, 2)
        
        # Mode
        cv2.putText(frame, f"Mode: {mode_status}", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 1)

        cv2.imshow("Hexagonal Grid Controller", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        return current_z_mm, current_radius, pitch_angle, roll_angle

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_capture_coordinates()