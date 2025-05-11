import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import winsound
import time
import pyttsx3  # Import pyttsx3 for text-to-speech
import threading  # Import threading for asynchronous speech

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CONSEC_FRAMES = 15
BEEP_DURATION = 1000
BEEP_FREQ = 2000

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# Page styling
st.set_page_config(page_title="Student Focus Monitor", layout="wide")
st.title("📚 Student Focus Monitor")
st.markdown("This tool helps you stay alert and focused while studying. 🚀")

# Camera ON/OFF Toggle
camera_on = st.sidebar.checkbox("📷 Camera ON/OFF", value=False)
FRAME_WINDOW = st.image([])

# Initialize session state for camera
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Initialize pyttsx3 engine for TTS
engine = pyttsx3.init()

def text_to_speech(text):
    """Function to convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def calculate_ear(landmarks, eye_indices):
    points = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, mouth_indices):
    upper = np.array([landmarks[mouth_indices[0]].x, landmarks[mouth_indices[0]].y])
    lower = np.array([landmarks[mouth_indices[1]].x, landmarks[mouth_indices[1]].y])
    left = np.array([landmarks[mouth_indices[2]].x, landmarks[mouth_indices[2]].y])
    right = np.array([landmarks[mouth_indices[3]].x, landmarks[mouth_indices[3]].y])
    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal

# Real-time alerts tracking
alert_log = []

def async_text_to_speech(text):
    """Threaded function for non-blocking text-to-speech."""
    speech_thread = threading.Thread(target=text_to_speech, args=(text,))
    speech_thread.start()

if camera_on:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)

    cap = st.session_state.camera
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    eye_frame_counter = 0
    yawn_frame_counter = 0
    alert_active = False
    session_time_start = time.time()

    while camera_on:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(landmarks, MOUTH)

                alert_msg = ""
                alert_color = (0, 255, 0)  # Default to green for focused
                alert_icon = "✅"

                if avg_ear < EAR_THRESHOLD:
                    eye_frame_counter += 1
                    if eye_frame_counter >= CONSEC_FRAMES:
                        alert_msg = "😴 Wake Up! Stay Focused!"
                        alert_color = (0, 0, 255)  # Red alert for sleepy
                        alert_icon = "❌"
                        if not alert_active:
                            winsound.Beep(BEEP_FREQ, BEEP_DURATION)
                            async_text_to_speech(alert_msg) 
                            alert_active = True
                            alert_log.append((time.strftime("%H:%M:%S", time.gmtime(time.time() - session_time_start)), alert_msg))
                else:
                    eye_frame_counter = 0
                    alert_active = False

                if mar > MAR_THRESHOLD:
                    yawn_frame_counter += 1
                    if yawn_frame_counter >= CONSEC_FRAMES:
                        alert_msg = "🥱 Feeling sleepy? Take a short break!"
                        alert_color = (255, 165, 0)  # Orange alert for yawning
                        alert_icon = "⚠️"
                        if not alert_active:
                            winsound.Beep(BEEP_FREQ, BEEP_DURATION)
                            async_text_to_speech(alert_msg)  
                            alert_active = True
                            alert_log.append((time.strftime("%H:%M:%S", time.gmtime(time.time() - session_time_start)), alert_msg))
                else:
                    yawn_frame_counter = 0
                    alert_active = False

                # Overlay the alert on the camera frame
                cv2.putText(frame, alert_msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_color, 3)
                cv2.putText(frame, alert_icon, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_color, 3)

        # Display the frame in the Streamlit app
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Check if the camera toggle is off, break the loop if so
        if not camera_on:
            break

        # Add a small delay to avoid maxing out CPU
        time.sleep(0.1)

    cap.release()
    st.session_state.camera = None

else:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    st.info("✅ Toggle the camera ON using the switch in the sidebar.")
