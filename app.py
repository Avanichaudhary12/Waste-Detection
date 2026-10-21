import cv2
import pandas as pd
from datetime import datetime
import streamlit as st
from ultralytics import YOLO
import os

# Page configuration
st.set_page_config(
    page_title="Smart Garbage Detector",
    page_icon="♻️",
    layout="wide"
)

# Custom CSS for modern UI cards
st.markdown("""
    <style>
    .detection-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-top: 15px;
    }
    .detected-tag {
        display: inline-block;
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 15px;
        margin: 4px;
    }
    .empty-tag {
        color: #94A3B8;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Load YOLO Model (cached so it only loads once)
@st.cache_resource
def load_model():
    return YOLO("best_1.pt")

model = load_model()

# CSV Log file path
LOG_FILE = "detections.csv"

def log_detection(item_count, detected_items_str):
    """Appends timestamp, count, and item labels to a local CSV log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, item_count, detected_items_str]], columns=["Timestamp", "Items_Detected", "Labels"])
    
    if os.path.exists(LOG_FILE):
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(LOG_FILE, mode='w', header=True, index=False)

# App UI Header
st.title("♻️ AI-Powered Garbage & Recyclables Detection System")
st.markdown("Detect cans and bottles in real-time using your custom YOLO model, OpenCV, and Streamlit.")

# Sidebar Navigation
app_mode = st.sidebar.selectbox(
    "Choose Application Mode",
    ["Image Detection", "Real-Time Webcam Feed", "Detection History & Logs"]
)

# --- MODE 1: IMAGE DETECTION ---
if app_mode == "Image Detection":
    st.header("Upload an Image for Detection")
    uploaded_file = st.file_uploader("Choose an image (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Save temporary image
        temp_path = "temp_img.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Run YOLO inference
        results = model(temp_path, conf=0.4)
        annotated_frame = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display Processed Image
        st.subheader("Processed Output")
        st.image(annotated_rgb, use_container_width=True)
        
        # Extract detected class labels
        boxes = results[0].boxes
        num_items = len(boxes)
        detected_names = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detected_names.append(class_name)
            
        # Format tags UI
        if num_items > 0:
            tags_html = "".join([f'<span class="detected-tag">🏷️ {name}</span>' for name in set(detected_names)])
            labels_str = ", ".join(detected_names)
            log_detection(num_items, labels_str)
        else:
            tags_html = '<span class="empty-tag">No waste items detected in this image.</span>'

        # Render Designed Output Card
        st.markdown(f"""
            <div class="detection-card">
                <h3 style="margin-top:0; color:#38BDF8;">📊 Analysis Summary</h3>
                <p style="font-size:18px; margin-bottom:10px;"><b>Total Items Detected:</b> <span style="color:#10B981; font-size:22px; font-weight:bold;">{num_items}</span></p>
                <div><b>Detected Categories:</b></div>
                <div style="margin-top:8px;">{tags_html}</div>
            </div>
        """, unsafe_allow_html=True)

# --- MODE 2: REAL-TIME WEBCAM FEED ---
elif app_mode == "Real-Time Webcam Feed":
    st.header("Live Camera Detection Feed")
    st.markdown("Click **Start Webcam** below to launch live tracking. Press **Stop** in the app to exit.")
    
    run_webcam = st.checkbox("Start Webcam")
    
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open web camera.")
        else:
            while run_webcam:
                success, frame = cap.read()
                if not success:
                    st.warning("Failed to grab camera frame.")
                    break
                
                results = model(frame, conf=0.4)
                annotated_frame = results[0].plot()
                boxes = results[0].boxes
                num_items = len(boxes)
                
                detected_names = [model.names[int(box.cls[0])] for box in boxes]
                labels_str = ", ".join(detected_names) if detected_names else "None"
                
                cv2.putText(
                    annotated_frame, 
                    f"Items: {num_items}", 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA
                )
                
                if num_items > 0:
                    log_detection(num_items, labels_str)
                
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                status_placeholder.info(f"Live Status: Tracking active | Items in frame: **{num_items}** ({labels_str})")
                
                if not st.session_state.get('Start Webcam', True):
                    break
            
            cap.release()
    else:
        st.info("Webcam is currently stopped.")

# --- MODE 3: DETECTION HISTORY & LOGS ---
elif app_mode == "Detection History & Logs":
    st.header("Detection Activity Logs")
    st.markdown("Review historical records of detected garbage events logged by the system.")
    
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df, use_container_width=True)
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Log as CSV",
            data=csv_data,
            file_name="garbage_detection_logs.csv",
            mime="text/csv",
        )
        
        if st.button("Clear Log History"):
            os.remove(LOG_FILE)
            st.rerun()
    else:
        st.info("No detection logs found yet. Run an image or webcam scan to generate logs.")