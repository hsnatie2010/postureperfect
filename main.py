import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import csv
import matplotlib.pyplot as plt
from PIL import Image
import google.generativeai as genai
import streamlit.components.v1 as components
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Set up model directory (use /tmp for writable space in Streamlit Cloud)
MODEL_DIR = "/tmp/mediapipe_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Model URLs (choose one based on your needs)
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
}

@st.cache_resource
def load_pose_model(model_complexity=1):
    # Choose model type
    model_type = "lite" if model_complexity == 0 else "full"
    model_path = os.path.join(MODEL_DIR, f"pose_landmarker_{model_type}.task")
    
    # Download if not exists
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URLS[model_type], model_path)
    
    # Initialize with the downloaded model
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)

# Initialize the detector
pose_detector = load_pose_model(model_complexity=1)  # Use 0 for lite, 1 for full

def play_alert():
    components.html(
        f"""
        <audio autoplay>
            <source src="alert.mp3" type="audio/mpeg">
        </audio>
        """,
        height=0,
    )


ALERT_COOLDOWN = 3.0  # seconds between alerts

# Constants
API_KEY = "AIzaSyDZ7IH_4gxfzegWcQr7uqlsdU9k8Gvsrj0"  # Replace with your API key
LOG_FILE = "posture_data.csv"
FRAME_SKIP = 2 # Process every 3rd frame
TARGET_FPS = 20
RESOLUTION = (680,340)  # Balanced resolution

# Posture recommendations dictionary
posture_recommendations = {
    "Forward Head Posture": (
        "Recommendations:\n"
        "1. Perform chin tucks: Gently pull your chin back to align your head over your spine.\n"
        "2. Strengthen your neck muscles with isometric exercises.\n"
        "3. Adjust your screen to eye level to reduce strain.\n"
        "Exercises:\n"
        "- Neck stretches: Tilt your head to each side and hold for 10 seconds.\n"
        "- Shoulder blade squeezes: Squeeze your shoulder blades together and hold for 5 seconds."
    ),
    "Rounded Shoulders or Hunched Back": (
        "Recommendations:\n"
        "1. Sit up straight and pull your shoulders back.\n"
        "2. Use a lumbar support cushion to maintain the natural curve of your spine.\n"
        "3. Avoid slouching when sitting or standing.\n"
        "Exercises:\n"
        "- Chest stretches: Stretch your chest by clasping your hands behind your back and lifting them slightly.\n"
        "- Rows: Use resistance bands to strengthen your upper back muscles."
    ),
    "Sway Back or Misaligned Lower Body": (
        "Recommendations:\n"
        "1. Engage your core muscles to support your lower back.\n"
        "2. Avoid locking your knees when standing.\n"
        "3. Stretch your hip flexors regularly.\n"
        "Exercises:\n"
        "- Pelvic tilts: Lie on your back and tilt your pelvis upward to flatten your lower back.\n"
        "- Glute bridges: Lift your hips off the ground while lying on your back to strengthen your glutes."
    ),
    "Good Posture (Aligned)": (
        "Congratulations! Your posture is well-aligned.\n"
        "Recommendations:\n"
        "1. Maintain your posture by sitting and standing tall.\n"
        "2. Take regular breaks to stretch and move around.\n"
        "3. Continue practicing good ergonomics."
    ),
    "Too Close!": (
        "Recommendations:\n"
        "1. Move further away from your screen to reduce eye strain.\n"
        "2. Ensure your screen is at least an arm's length away.\n"
        "3. Adjust your chair and desk height for better ergonomics.\n"
        "Exercises:\n"
        "- Eye exercises: Look away from your screen every 20 minutes and focus on a distant object for 20 seconds."
    ),
    "Head Forward!": (
        "Recommendations:\n"
        "1. Align your head over your spine to reduce neck strain.\n"
        "2. Adjust your screen to eye level.\n"
        "3. Perform regular neck stretches.\n"
        "Exercises:\n"
        "- Chin tucks: Gently pull your chin back to align your head over your spine.\n"
        "- Neck stretches: Tilt your head to each side and hold for 10 seconds."
    )
}

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# — outside of any function —
mp_face_detector = mp_face.FaceDetection(
    min_detection_confidence=0.5,
    model_selection=0
)
mp_pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0      # simplest, fastest
)

# Initialize Gemini AI
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.last_update = time.time()
    
    def update(self, frame):
        with self.lock:
            if frame is not None and isinstance(frame, np.ndarray):
                self.frame = frame.copy()
                self.last_update = time.time()
        
    def get(self):
        with self.lock:
            if self.frame is None:
                print("❌ No frame in buffer")
                return None
            # Allow frames up to 10 seconds old (for testing)
            if time.time() - self.last_update > 10.0:
                print("⚠️ Frame expired")
                return self.frame.copy()  # ← still show itst anyway!
            print("✅ Returning frame")
            return self.frame.copy()

class CameraThread:
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer
        self.running = False
        self.cap = None
        self.thread = None
    
    def start_capture(self):
        if self.running:
            print("🚫 Camera already running")
            return False

        print("🔄 Attempting to start camera...")

        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            print(f"🎥 Trying backend: {backend}")
            self.cap = cv2.VideoCapture(0, backend)
            if self.cap.isOpened():
                print(f"✅ Camera opened using backend: {backend}")
                break

        if not self.cap or not self.cap.isOpened():
            print("❌ Failed to open camera with all backends")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        # Start the thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("🚀 Camera thread started")
        return True

    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert to 3-channel BGR if needed
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                self.frame_buffer.update(frame)
            else:
                time.sleep(0.01)
                print("🌀 Camera thread running...")

            ret, frame = self.cap.read()
            print(f"📸 Frame read: {ret}, shape: {frame.shape if ret else 'N/A'}")
            if ret:
                self.frame_buffer.update(frame)
            else:
                time.sleep(0.01)
    
    def stop_capture(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.thread = None

def analyze_posture(frame, face_model, pose_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # run detectors once
    face_results = face_model.process(rgb)
    pose_results = pose_model.process(rgb)

    status = "Good Posture"

    # — Too close?
    if face_results.detections:
        # only check first face
        if face_results.detections[0].location_data.relative_bounding_box.width > 0.4:
            status = "Too Close!"

    # — Head forward?
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_line = (l_sh.y + r_sh.y) / 2 - 0.2
        if nose.y > shoulder_line:
            status = "Head Forward!"

    # — Overlay only the status text
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return status, frame




def plot_graph():
    try:
        data = []
        with open(LOG_FILE, "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                data.append(row)

        if not data:
            st.warning("No data available. Start detection first.")
            return

        time_stamps = []
        posture_statuses = []
        good_posture_time = 0
        bad_posture_time = 0

        for row in data:
            time_stamp = float(row[0])
            posture_status = row[1]
            time_stamps.append(time_stamp / 60)  # Convert to minutes
            posture_statuses.append(posture_status)

            if posture_status == "Good Posture":
                good_posture_time += 1
            else:
                bad_posture_time += 1

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(time_stamps) - 1):
            color = "green" if posture_statuses[i] == "Good Posture" else "red"
            ax.plot(time_stamps[i:i+2], [i, i+1], color=color, linewidth=2)

        ax.set_xlabel("Time (Minutes)")
        ax.set_ylabel("Posture Status")
        ax.set_title("Posture Over Time")
        ax.grid(True)
        ax.legend(["Good Posture", "Needs Improvement"], loc="upper left")

        st.pyplot(fig)

        # Delete the log file after plotting
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")


def process_image_for_posture(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error loading image"
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        results = pose.process(rgb_image)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            
            # Use side landmarks (e.g. left ear and left shoulder)
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            
            # Check forward head posture (ear too far in front of shoulder in x-axis)
            if left_ear.visibility > 0.5 and left_shoulder.visibility > 0.5:
                ear_shoulder_x_diff = left_shoulder.x - left_ear.x  # +ve if ear ahead (bad)
                if ear_shoulder_x_diff < -0.05:  # Threshold for forward head posture
                    posture_status = "Head Forward!"
                else:
                    posture_status = "Good Posture"
            else:
                posture_status = "Landmarks not clear"
        else:
            posture_status = "No pose detected"

        cv2.putText(image, posture_status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image, posture_status


def get_ai_recommendations(posture_status):
    try:
        prompt = f"Provide posture improvement recommendations for: {posture_status}. Include both exercises and ergonomic adjustments.In english and arabic."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not get AI recommendations: {str(e)}"

def process_health_inquiry(symptoms):
    try:
        prompt = f"Analyze these posture-related symptoms: {symptoms}. Provide recommendations in english and arabic."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not process health inquiry: {str(e)}"

def main():
    st.set_page_config(page_title="Posture Perfect", layout="wide")
    
    # Initialize session state
    if 'frame_buffer' not in st.session_state:
        st.session_state.frame_buffer = FrameBuffer()
    
    if 'camera_thread' not in st.session_state:
        st.session_state.camera_thread = CameraThread(st.session_state.frame_buffer)
    
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
        st.session_state.start_time = 0
        st.session_state.posture_status = "Good Posture"
        st.session_state.ai_response = ""
        st.session_state.frame_counter = 0
    # Show project logo in sidebar
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width
=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
        ["Live Detection", "AI Recommendations", "Analytics", "Image Analysis", "Health"])

    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = 0.0


    if page == "Live Detection":
        st.title(" Posture Monitoring ")

        start = st.button("Start Detection")
        stop  = st.button("Stop Detection")

        # Start
        if start and not st.session_state.detection_active:
            st.session_state.camera_thread.start_capture()
            st.session_state.detection_active   = True
            st.session_state.start_time         = time.time()
            st.session_state.frame_counter      = 0
            # init alert timer if using pygame
            if 'last_alert_time' not in st.session_state:
                st.session_state.last_alert_time = 0.0
            # ensure CSV header
            if not os.path.exists(LOG_FILE):
                with open(LOG_FILE, "w", newline="") as f:
                    csv.writer(f).writerow(["timestamp","posture"])
            st.success("🎥 Detection started")

        # Stop
        if stop and st.session_state.detection_active:
            st.session_state.camera_thread.stop_capture()
            st.session_state.detection_active = False
            st.success("🛑 Detection stopped")

        # placeholder for frames (always defined)
        frame_pl = st.empty()

        # only enter loop when active
        while st.session_state.detection_active:
            frame = st.session_state.frame_buffer.get()
            if frame is None:
                time.sleep(0.1)
                continue

            # skip frames for speed
            st.session_state.frame_counter += 1
            if st.session_state.frame_counter % FRAME_SKIP != 0:
                continue

            # analyze posture
            status, annotated = analyze_posture(
                frame,
                mp_face_detector,
                mp_pose_detector
            )

            # display with container width
            frame_pl.image(annotated, channels="BGR", use_container_width=True)

            # play alert (pygame example)
            current_time = time.time()
            if status != "Good Posture" and (current_time - st.session_state.last_alert_time > ALERT_COOLDOWN):
              play_alert()
              st.session_state.last_alert_time = current_time

            # log timestamp + status
            ts = time.time() - st.session_state.start_time
            with open(LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow([ts, status])

            # save for AI Recommendations tab
            st.session_state.posture_status = status
            st.session_state.ai_response    = posture_recommendations.get(status, "")

            # throttle
            time.sleep(1.0 / TARGET_FPS)


    elif page == "AI Recommendations":
        st.title("AI Posture Advisor")

        status = st.session_state.get("posture_status", None)

        if status:
            st.subheader(f"Last Detected Posture: **{status}**")

            # show existing recommendation if any
            if st.session_state.get("ai_response", ""):
                st.text_area("Recommendations", st.session_state.ai_response, height=300)
            else:
                st.info("Click below to generate tailored advice for this posture.")

            # button to (re)generate advice for this exact status
            if st.button(f"Get Advice for “{status}”"):
                with st.spinner("Generating advice…"):
                    advice = get_ai_recommendations(status)
                    st.session_state.ai_response = advice
                    st.text_area("Recommendations", advice, height=300)
        else:
            st.warning("No posture detected yet. Go to **Live Detection** and start monitoring first!")

        st.markdown("---")
        if st.button("Get General Posture Advice"):
            with st.spinner("Generating general advice…"):
                advice = get_ai_recommendations("general posture improvement")
                st.text_area("General Advice", advice, height=300)

    elif page == "Analytics":
        st.title("Posture Analytics")

        if st.button("Show *Last* Posture Trends"):
            plot_graph()

        # Offer CSV download if data exists
        try:
            with open(LOG_FILE, "r") as f:
                # if file has at least one data row
                if f.readline():
                    st.download_button(
                        "Download Posture Data",
                        data=open(LOG_FILE).read(),
                        file_name="posture_data.csv"
                    )
                else:
                    st.warning("No posture data available yet.")
        except FileNotFoundError:
            st.warning("No posture data available yet.")

    elif page == "Image Analysis":
        st.title("Posture Image Analysis")

        st.markdown(
            "**⚠️ Please upload a side‑view (profile) photo, not a face‑forward shot.**"
        )
        # example side‑view image
        if os.path.exists("example.png"):
            st.image("example.png",
                    caption="✅ Good: side profile for posture analysis",
                    use_container_width=True)

        uploaded = st.file_uploader("Upload posture image",
                                    type=["jpg", "jpeg", "png"])
        
        if uploaded:
            with open("temp.jpg", "wb") as f:
                f.write(uploaded.getbuffer())
            
            image, status = process_image_for_posture("temp.jpg")
            
            if image is not None:
                st.image(image, use_container_width=True)
                st.subheader("Analysis Result")
                st.write(status)
                
                if status != "Good Posture":
                    st.text_area("Recommendations",
                                posture_recommendations.get(status, ""),
                                height=200)
            
            os.remove("temp.jpg")

    elif page == "Health":
        st.title("Posture Health Advisor")
        
        symptoms = st.text_area("Describe your symptoms (backpain,neckpain .....)", height=100)
        
        if st.button("Analyze Symptoms"):
            if symptoms.strip():
                with st.spinner("Analyzing..."):
                    response = process_health_inquiry(symptoms)
                    st.text_area("Recommendations", response, height=300)
            else:
                st.warning("Please describe your symptoms")


if __name__ == "__main__":
    main()
