import cv2
import mediapipe as mp
import math
import numpy as np

# --- Configuration Constants ---
MAX_Z_HEIGHT_MM = 2200
MAX_RADIUS = 10000
FIST_THRESHOLD_RATIO = 1.3

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

class HandTracker:
    def __init__(self):
        """Initialize Camera and MediaPipe once."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # Default State
        self.z = 0
        self.radius = 10000
        self.pitch = 0
        self.roll = 0

    def get_coordinates(self):
        """
        Process ONE frame and return coordinates.
        Returns: (z, radius, pitch, roll) or None if quit.
        """
        if not self.cap.isOpened():
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        h, w, _ = frame.shape
        
        mode_status = "FLAT / CURVE"
        color_status = (0, 255, 0)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Landmarks
                wrist = hand_landmarks.landmark[0]
                index_mcp = hand_landmarks.landmark[5]   
                middle_mcp = hand_landmarks.landmark[9]
                middle_tip = hand_landmarks.landmark[12]
                pinky_mcp = hand_landmarks.landmark[17]
                
                # Pixels
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
                tx, ty = int(middle_tip.x * w), int(middle_tip.y * h)
                px, py = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
                ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)

                # 1. Z-Height
                self.z = int((wy / h) * MAX_Z_HEIGHT_MM)
                self.z = max(0, min(MAX_Z_HEIGHT_MM, self.z))

                # 2. Fist/Open Detection
                dist_wrist_tip = calculate_distance((wx, wy), (tx, ty))
                dist_wrist_mcp = calculate_distance((wx, wy), (mx, my))
                ratio = dist_wrist_tip / (dist_wrist_mcp + 1e-6) 

                if ratio < FIST_THRESHOLD_RATIO:
                    self.radius = 0
                    self.pitch = 0
                    self.roll = 0
                    mode_status = "SPHERE (FIST)"
                    color_status = (0, 0, 255)
                else:
                    self.radius = 10000
                    # Pitch
                    pitch_dy = wy - my
                    pitch_dx = mx - wx
                    self.pitch = int(math.degrees(math.atan2(pitch_dy, abs(pitch_dx) + 1)))
                    self.pitch = max(-60, min(60, self.pitch))

                    # Roll
                    roll_dy = py - iy
                    roll_dx = px - ix
                    self.roll = int(math.degrees(math.atan2(roll_dy, roll_dx)))
                    self.roll = max(-60, min(60, self.roll))

                # Draw
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (wx, wy), 8, color_status, -1)

        # Dashboard
        cv2.rectangle(frame, (0, h-60), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"Z:{self.z} R:{self.radius} P:{self.pitch} Y:{self.roll} | {mode_status}", 
                    (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            return None
            
        return self.z, self.radius, self.pitch, self.roll

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()