import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import streamlit as st

st.set_page_config(
    page_title="HandTrack — AI Gesture Control Center",
    page_icon="🖐️",
    layout="wide"
)


st.markdown("""
    <style>
    .stApp {
        background-color: #0B1018;
        color: #F1F5F9;
    }
    [data-testid="stSidebar"] {
        background-color: #141E2B;
        border-right: 1px solid #1B2939;
    }
    .metric-card {
        background-color: #141E2B;
        border: 1px solid #1B2939;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    h1, h2, h3 {
        color: #F1F5F9 !important;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


pyautogui.FAILSAFE = True


st.sidebar.title("🎛️ Command Center")
app_status = st.sidebar.radio("System State", ["Stopped", "Running"])
enable_mouse = st.sidebar.checkbox("Enable System Mouse Control", value=False)
mirror_view = st.sidebar.checkbox("Mirror Camera Feed", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Gesture Tuning")
smoothening = st.sidebar.slider("Cursor Smoothing", 1, 10, 5)
pinch_threshold = st.sidebar.slider("Pinch Threshold (Distance)", 20, 50, 30)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Switch System State to **Running**.\n"
    "2. Check **Enable System Mouse Control** to activate cursor tracking.\n"
    "3. Use Index Finger to steer; Pinch Thumb & Index to click."
)


st.title("🖐️ HandTrack — AI-Powered Touchless Computer Control")
st.markdown("Real-time computer vision telemetry, landmark tracking, and system automation command center.")


col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    kpi_fps = st.metric(label="Frame Rate", value="0 FPS")
with col_kpi2:
    kpi_latency = st.metric(label="Processing Latency", value="0 ms")
with col_kpi3:
    kpi_hand = st.metric(label="Hand State", value="Inactive")
with col_kpi4:
    kpi_gesture = st.metric(label="Active Gesture", value="None")

st.markdown("---")


main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    st.subheader("📹 Live Camera Feed & Landmark Skeleton")
    video_placeholder = st.empty()

with main_col2:
    st.subheader("📋 Real-Time Event Log")
    event_log_container = st.container(height=380)
    event_logs = []

    def log_event(event_text):
        timestamp = time.strftime("%H:%M:%S")
        event_logs.insert(0, f"`[{timestamp}]` {event_text}")
        if len(event_logs) > 8:
            event_logs.pop()
        with event_log_container:
            st.markdown("\n\n".join(event_logs))


if app_status == "Running":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    mp_draw = mp.solutions.drawing_utils
    
    
    hand_connection_spec = mp_draw.DrawingSpec(color=(208, 230, 85), thickness=2) # BGR Cyan-ish
    hand_landmark_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)

    screen_width, screen_height = pyautogui.size()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    prev_x, prev_y = 0, 0
    prev_time = time.time()
    click_cooldown = 0

    log_event("System started. Camera stream initialized.")

    while cap.isOpened() and app_status == "Running":
        loop_start = time.time()
        success, frame = cap.read()
        if not success:
            st.error("⚠️ Failed to access webcam feed.")
            log_event("Camera error: Frame capture failed.")
            break

        if mirror_view:
            frame = cv2.flip(frame, 1)

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        current_gesture = "None"
        hand_state_text = "No Hand Detected ⚠️"

        if result.multi_hand_landmarks:
            hand_state_text = "Hand Detected 🟢"
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    hand_landmark_spec, hand_connection_spec
                )
                
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

                
                screen_x = np.interp(x_index, (80, w - 80), (0, screen_width))
                screen_y = np.interp(y_index, (80, h - 80), (0, screen_height))

                
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening

                if enable_mouse:
                    pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                
                distance = np.hypot(x_index - x_thumb, y_index - y_thumb)
                if distance < pinch_threshold:
                    current_gesture = "Pinch (Click) ⚡"
                    if enable_mouse and time.time() - click_cooldown > 0.5:
                        pyautogui.click()
                        click_cooldown = time.time()
                        log_event("⚡ Pinch detected: Click triggered.")
                    cv2.putText(frame, "CLICK!", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (85, 230, 208), 2)
                else:
                    current_gesture = "Hovering / Tracking"
        else:
            log_event("Warning: Hand lost from camera frame.")

       
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
        latency_ms = int((curr_time - loop_start) * 1000)
        prev_time = curr_time

       
        kpi_fps.metric(label="Frame Rate", value=f"{fps} FPS")
        kpi_latency.metric(label="Processing Latency", value=f"{latency_ms} ms")
        kpi_hand.metric(label="Hand State", value=hand_state_text)
        kpi_gesture.metric(label="Active Gesture", value=current_gesture)

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        
        if app_status == "Stopped":
            break

    cap.release()
    cv2.destroyAllWindows()
    log_event("System stopped. Camera resources released.")
    kpi_hand.metric(label="Hand State", value="Stopped 🔴")