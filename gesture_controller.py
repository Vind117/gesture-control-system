import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable PyAutoGUI fail-safe exception
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands with high confidence thresholds
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Smoothing variables (Exponential Moving Average)
prev_x, prev_y = 0, 0
smoothening = 5  # Higher = smoother but more delay, Lower = faster but more jitter

print("Starting Optimized Hand Gesture Controller... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip horizontally for natural mirror interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Index Finger Tip (Landmark 8) & Thumb Tip (Landmark 4)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Map camera coordinates to screen resolution with frame padding (100px border)
            screen_x = np.interp(x_index, (100, w - 100), (0, screen_width))
            screen_y = np.interp(y_index, (100, h - 100), (0, screen_height))

            # Apply Smoothing (EMA)
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            # Move Mouse Cursor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Detect Click via pinch distance
            distance = np.hypot(x_index - x_thumb, y_index - y_thumb)
            if distance < 30:  # Pinch threshold
                pyautogui.click()
                cv2.putText(frame, "CLICK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display live feed
    cv2.imshow("Optimized AI Gesture Control", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()