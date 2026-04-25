"""
Deepfake Detection Web App - With Live rPPG Tracking for Uploaded Videos
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, find_peaks
from pathlib import Path
import tempfile
import time
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Deepfake Detection System - Live rPPG Tracking",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .real-badge {
        background-color: #10b981;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .fake-badge {
        background-color: #ef4444;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    model_path = "deepfake_model_persistent.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            return data['model'], data['scaler'], data['feature_names']
    return None, None, None

# RPPG Extractor with Live Visualization
class RPPGExtractor:
    def __init__(self, fps=30):
        self.fps = fps
        self.lowcut = 0.75
        self.highcut = 3.5
        self.order = 4
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.signal_buffer = []
        self.roi_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': [], 'chin': []}
        self._init_filter()
        
        # ROI colors for visualization
        self.roi_colors = {
            'forehead': (255, 0, 0),      # Blue
            'left_cheek': (0, 255, 255),   # Yellow
            'right_cheek': (255, 255, 0),  # Cyan
            'chin': (255, 0, 255)          # Magenta
        }
    
    def _init_filter(self):
        nyquist = 0.5 * self.fps
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        if 0 < low < 1 and 0 < high < 1 and low < high:
            self.b, self.a = butter(self.order, [low, high], btype='band')
        else:
            self.b, self.a = [1.0], [1.0]
    
    def extract_rois(self, frame, face_bbox):
        x, y, w, h = face_bbox
        rois = {}
        
        # Forehead (Blue)
        fy1 = max(0, y + int(h * 0.1))
        fy2 = min(frame.shape[0], y + int(h * 0.35))
        fx1 = max(0, x + int(w * 0.2))
        fx2 = min(frame.shape[1], x + int(w * 0.8))
        if fy1 < fy2 and fx1 < fx2:
            rois['forehead'] = frame[fy1:fy2, fx1:fx2]
        
        # Left Cheek (Yellow)
        lx1 = max(0, x + int(w * 0.05))
        lx2 = max(0, x + int(w * 0.25))
        ly1 = max(0, y + int(h * 0.5))
        ly2 = min(frame.shape[0], y + int(h * 0.75))
        if ly1 < ly2 and lx1 < lx2:
            rois['left_cheek'] = frame[ly1:ly2, lx1:lx2]
        
        # Right Cheek (Cyan)
        rx1 = min(frame.shape[1], x + int(w * 0.75))
        rx2 = min(frame.shape[1], x + int(w * 0.95))
        ry1 = max(0, y + int(h * 0.5))
        ry2 = min(frame.shape[0], y + int(h * 0.75))
        if ry1 < ry2 and rx1 < rx2:
            rois['right_cheek'] = frame[ry1:ry2, rx1:rx2]
        
        # Chin (Magenta)
        cy1 = min(frame.shape[0], y + int(h * 0.75))
        cy2 = min(frame.shape[0], y + int(h * 0.95))
        cx1 = max(0, x + int(w * 0.3))
        cx2 = min(frame.shape[1], x + int(w * 0.7))
        if cy1 < cy2 and cx1 < cx2:
            rois['chin'] = frame[cy1:cy2, cx1:cx2]
        
        return rois
    
    def draw_rois(self, frame, rois, face_bbox):
        x, y, w, h = face_bbox
        result = frame.copy()
        
        # Draw face rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw each ROI
        # Forehead
        if 'forehead' in rois:
            fy1 = y + int(h * 0.1)
            fy2 = y + int(h * 0.35)
            fx1 = x + int(w * 0.2)
            fx2 = x + int(w * 0.8)
            cv2.rectangle(result, (fx1, fy1), (fx2, fy2), self.roi_colors['forehead'], 2)
            cv2.putText(result, "Forehead", (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['forehead'], 2)
        
        # Left Cheek
        if 'left_cheek' in rois:
            lx1 = x + int(w * 0.05)
            lx2 = x + int(w * 0.25)
            ly1 = y + int(h * 0.5)
            ly2 = y + int(h * 0.75)
            cv2.rectangle(result, (lx1, ly1), (lx2, ly2), self.roi_colors['left_cheek'], 2)
            cv2.putText(result, "L-Cheek", (lx1, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['left_cheek'], 2)
        
        # Right Cheek
        if 'right_cheek' in rois:
            rx1 = x + int(w * 0.75)
            rx2 = x + int(w * 0.95)
            ry1 = y + int(h * 0.5)
            ry2 = y + int(h * 0.75)
            cv2.rectangle(result, (rx1, ry1), (rx2, ry2), self.roi_colors['right_cheek'], 2)
            cv2.putText(result, "R-Cheek", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['right_cheek'], 2)
        
        # Chin
        if 'chin' in rois:
            cy1 = y + int(h * 0.75)
            cy2 = y + int(h * 0.95)
            cx1 = x + int(w * 0.3)
            cx2 = x + int(w * 0.7)
            cv2.rectangle(result, (cx1, cy1), (cx2, cy2), self.roi_colors['chin'], 2)
            cv2.putText(result, "Chin", (cx1, cy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['chin'], 2)
        
        return result
    
    def extract_chrom_signal(self, roi):
        if roi is None or roi.size == 0:
            return None
        roi_float = roi.astype(np.float32)
        r = np.mean(roi_float[:, :, 2]) / 255.0
        g = np.mean(roi_float[:, :, 1]) / 255.0
        b = np.mean(roi_float[:, :, 0]) / 255.0
        Xs = 3 * r - 2 * g
        Ys = 1.5 * r + g - 1.5 * b
        norm = np.sqrt(Xs**2 + Ys**2)
        if norm > 0:
            Xs = Xs / norm
            Ys = Ys / norm
        return Xs + Ys
    
    def compute_heart_rate(self, signals):
        if len(signals) < 45:
            return None
        signal_data = np.array(signals[-150:])
        signal_data = detrend(signal_data)
        if np.std(signal_data) > 0:
            signal_data = signal_data / np.std(signal_data)
        try:
            filtered = filtfilt(self.b, self.a, signal_data)
        except:
            filtered = signal_data
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, 1.0/self.fps)
        fft_vals = np.abs(np.fft.rfft(filtered))
        mask = (freqs >= 0.75) & (freqs <= 3.5)
        if not np.any(mask):
            return None
        freq_range = freqs[mask]
        power_range = fft_vals[mask]
        if len(power_range) == 0:
            return None
        hr_freq = freq_range[np.argmax(power_range)]
        return np.clip(hr_freq * 60, 45, 180)
    
    def compute_rppg_score(self):
        if len(self.signal_buffer) < 45:
            return 0.5
        
        hr = self.compute_heart_rate(self.signal_buffer)
        signal_std = np.std(self.signal_buffer[-100:]) if len(self.signal_buffer) >= 100 else np.std(self.signal_buffer)
        
        if hr is None:
            hr_score = 0.3
        elif 60 <= hr <= 100:
            hr_score = 1.0
        else:
            hr_score = 0.5
        
        quality = min(1.0, signal_std * 2)
        
        peaks, _ = find_peaks(self.signal_buffer, height=0.2*np.std(self.signal_buffer), distance=10)
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            regularity = 1.0 - min(0.9, np.std(intervals) / (np.mean(intervals) + 1e-6))
        else:
            regularity = 0.3
        
        return 0.5 * hr_score + 0.3 * quality + 0.2 * regularity
    
    def process_video_with_visualization(self, video_path, max_frames=300):
        """Process video and return frames with ROI visualization"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps
        self._init_filter()
        
        # Reset buffers
        self.signal_buffer = []
        self.roi_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': [], 'chin': []}
        
        processed_frames = []
        frame_count = 0
        process_every = max(1, int(fps / 15))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every == 0:
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
                
                if len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Extract ROIs
                    rois = self.extract_rois(frame, best_face)
                    
                    # Draw ROIs on frame
                    display_frame = self.draw_rois(frame, rois, best_face)
                    
                    # Extract signals from each ROI
                    for roi_name, roi in rois.items():
                        signal_val = self.extract_chrom_signal(roi)
                        if signal_val is not None:
                            self.roi_signals[roi_name].append(signal_val)
                            if roi_name == 'forehead':
                                self.signal_buffer.append(signal_val)
                                if len(self.signal_buffer) > 300:
                                    self.signal_buffer.pop(0)
                    
                    # Add metrics on frame
                    rppg_score = self.compute_rppg_score()
                    heart_rate = self.compute_heart_rate(self.signal_buffer)
                    
                    # Display scores on frame
                    cv2.putText(display_frame, f"rPPG Score: {rppg_score:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if heart_rate:
                        cv2.putText(display_frame, f"Heart Rate: {heart_rate:.0f} BPM", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    processed_frames.append(display_frame)
                else:
                    # No face detected
                    cv2.putText(frame, "No Face Detected", (frame.shape[1]//2-100, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    processed_frames.append(frame)
                
                # Update progress
                progress = min(1.0, frame_count / max_frames)
                progress_bar.progress(progress)
                status_text.text(f"Processing frames: {frame_count}/{max_frames}")
            
            frame_count += 1
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        # Calculate final features
        features = {}
        for roi_name, signals in self.roi_signals.items():
            if len(signals) >= 30:
                hr = self.compute_heart_rate(signals)
                signal_std = np.std(signals)
                
                if hr is None:
                    hr_score = 0.3
                elif 60 <= hr <= 100:
                    hr_score = 1.0
                else:
                    hr_score = 0.5
                
                quality = min(1.0, signal_std * 2)
                
                peaks, _ = find_peaks(signals, height=0.2*np.std(signals), distance=10)
                if len(peaks) > 1:
                    intervals = np.diff(peaks)
                    regularity = 1.0 - min(0.9, np.std(intervals) / (np.mean(intervals) + 1e-6))
                else:
                    regularity = 0.3
                
                features[f'{roi_name}_hr'] = hr_score
                features[f'{roi_name}_quality'] = quality
                features[f'{roi_name}_regularity'] = regularity
            else:
                features[f'{roi_name}_hr'] = 0.5
                features[f'{roi_name}_quality'] = 0.5
                features[f'{roi_name}_regularity'] = 0.5
        
        # Combined score
        hr_values = [features.get(f'{roi}_hr', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        quality_values = [features.get(f'{roi}_quality', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        features['combined_score'] = np.mean(hr_values) * 0.6 + np.mean(quality_values) * 0.4
        
        return processed_frames, features, self.roi_signals


def create_signal_plot(roi_signals):
    """Create interactive signal plot for all ROIs"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Forehead (Blue)', 'Left Cheek (Yellow)', 'Right Cheek (Cyan)', 'Chin (Magenta)'))
    
    colors = {'forehead': 'blue', 'left_cheek': 'yellow', 'right_cheek': 'cyan', 'chin': 'magenta'}
    positions = {'forehead': (1, 1), 'left_cheek': (1, 2), 'right_cheek': (2, 1), 'chin': (2, 2)}
    
    for roi_name, signals in roi_signals.items():
        if len(signals) > 0:
            row, col = positions.get(roi_name, (1, 1))
            fig.add_trace(
                go.Scatter(y=signals, mode='lines', name=roi_name, line=dict(color=colors.get(roi_name, 'white'), width=2)),
                row=row, col=col
            )
            fig.update_xaxes(title_text="Frame", row=row, col=col)
            fig.update_yaxes(title_text="Amplitude", row=row, col=col)
    
    fig.update_layout(height=600, showlegend=True, title_text="rPPG Signals from Different ROIs")
    return fig


def main():
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Deepfake Detection System</h1>
        <p>Live rPPG Tracking | Multi-ROI Analysis | Real-time Visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.warning("⚠️ Pre-trained model not found. Using rule-based detection.")
    else:
        st.success("✅ Pre-trained model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 About")
        st.markdown("""
        This system detects deepfake videos using **rPPG (remote Photoplethysmography)** technology.
        
        **Live Tracking Features:**
        - 🔵 **Forehead** - Primary rPPG signal source
        - 🟡 **Left Cheek** - Secondary signal verification
        - 🩵 **Right Cheek** - Secondary signal verification  
        - 🟪 **Chin** - Additional signal source
        
        **Detection Method:**
        - Tracks color changes from blood flow
        - Calculates heart rate patterns
        - Uses ML model for classification
        """)
        
        st.markdown("---")
        st.markdown("### 🎨 ROI Color Legend")
        st.markdown("🔵 **Blue** - Forehead")
        st.markdown("🟡 **Yellow** - Left Cheek")
        st.markdown("🩵 **Cyan** - Right Cheek")
        st.markdown("🟪 **Magenta** - Chin")
        st.markdown("🟢 **Green** - Face Boundary")
    
    # Main content
    tab1, tab2 = st.tabs(["📹 Live Webcam", "🎥 Upload Video with Live Tracking"])
    
    with tab1:
        st.markdown("### Live Webcam Detection")
        st.info("Click 'Start Camera' to see real-time rPPG tracking with ROI visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            start_cam = st.button("▶️ Start Camera", use_container_width=True, type="primary")
        with col2:
            stop_cam = st.button("⏹️ Stop Camera", use_container_width=True)
        
        if start_cam:
            extractor = RPPGExtractor(fps=30)
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot open camera!")
            else:
                video_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = extractor.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(100, 100))
                    
                    display_frame = frame.copy()
                    
                    if len(faces) > 0:
                        best_face = max(faces, key=lambda f: f[2] * f[3])
                        rois = extractor.extract_rois(frame, best_face)
                        display_frame = extractor.draw_rois(display_frame, rois, best_face)
                        
                        # Extract signal from forehead
                        if 'forehead' in rois:
                            signal_val = extractor.extract_chrom_signal(rois['forehead'])
                            if signal_val is not None:
                                extractor.signal_buffer.append(signal_val)
                                if len(extractor.signal_buffer) > 300:
                                    extractor.signal_buffer.pop(0)
                    
                    rppg_score = extractor.compute_rppg_score()
                    heart_rate = extractor.compute_heart_rate(extractor.signal_buffer)
                    
                    # Add metrics on frame
                    cv2.putText(display_frame, f"rPPG: {rppg_score:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if heart_rate:
                        cv2.putText(display_frame, f"HR: {heart_rate:.0f} BPM", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Prediction
                    if rppg_score > 0.6:
                        label = "REAL"
                        color = (0, 255, 0)
                    elif rppg_score > 0.45:
                        label = "UNCERTAIN"
                        color = (0, 165, 255)
                    else:
                        label = "FAKE"
                        color = (0, 0, 255)
                    
                    cv2.putText(display_frame, f"Prediction: {label}", (display_frame.shape[1]-250, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    metrics_placeholder.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #1f2937; border-radius: 10px;">
                        <h3 style="color: {'#10b981' if label == 'REAL' else '#ef4444' if label == 'FAKE' else '#f59e0b'}">
                            {label}
                        </h3>
                        <p>rPPG Score: {rppg_score:.3f}</p>
                        <p>Heart Rate: {heart_rate:.0f} BPM</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if stop_cam:
                        break
                    
                    time.sleep(0.03)
                
                cap.release()
                cv2.destroyAllWindows()
    
    with tab2:
        st.markdown("### Upload Video with Live rPPG Tracking")
        st.info("Watch the video with real-time ROI tracking and rPPG signal visualization")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to see live rPPG tracking with ROI visualization"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Display video info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(uploaded_file)
            with col2:
                st.info(f"""
                **Video Information:**
                - Filename: {uploaded_file.name}
                - Size: {uploaded_file.size / 1024 / 1024:.2f} MB
                """)
            
            if st.button("🔍 Analyze with Live Tracking", use_container_width=True, type="primary"):
                with st.spinner("Processing video with live rPPG tracking..."):
                    # Process video with visualization
                    extractor = RPPGExtractor()
                    processed_frames, features, roi_signals = extractor.process_video_with_visualization(video_path)
                    
                    if processed_frames and len(processed_frames) > 0:
                        # Display processed video with ROI tracking
                        st.markdown("### 🎥 Live rPPG Tracking Result")
                        st.markdown("*Video with ROI visualization (Forehead: Blue, Cheeks: Yellow/Cyan, Chin: Magenta)*")
                        
                        # Display video frames
                        frame_placeholder = st.empty()
                        progress_slider = st.slider("Frame Navigation", 0, len(processed_frames)-1, 0)
                        
                        # Auto-play option
                        auto_play = st.checkbox("Auto-play video")
                        
                        if auto_play:
                            for i, frame in enumerate(processed_frames):
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                                time.sleep(0.05)
                        else:
                            frame_rgb = cv2.cvtColor(processed_frames[progress_slider], cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Calculate final prediction
                        combined_score = features.get('combined_score', 0.5)
                        
                        if combined_score > 0.6:
                            label = "REAL"
                            confidence = combined_score
                        elif combined_score > 0.45:
                            label = "UNCERTAIN"
                            confidence = 0.5
                        else:
                            label = "FAKE"
                            confidence = 1 - combined_score
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if label == "REAL":
                                st.markdown('<div class="real-badge">✅ REAL</div>', unsafe_allow_html=True)
                            elif label == "FAKE":
                                st.markdown('<div class="fake-badge">❌ FAKE</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="metric-card">⚠️ UNCERTAIN</div>', unsafe_allow_html=True)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col2:
                            st.metric("Combined Score", f"{combined_score:.3f}")
                            st.metric("Forehead Quality", f"{features.get('forehead_quality', 0.5):.3f}")
                        
                        with col3:
                            st.metric("Left Cheek", f"{features.get('left_cheek_quality', 0.5):.3f}")
                            st.metric("Right Cheek", f"{features.get('right_cheek_quality', 0.5):.3f}")
                        
                        # Signal plots
                        st.markdown("### 📈 rPPG Signals by ROI")
                        signal_plot = create_signal_plot(roi_signals)
                        st.plotly_chart(signal_plot, use_container_width=True)
                        
                        # Feature explanation
                        with st.expander("📖 Understanding the Results"):
                            st.markdown("""
                            **What the colors mean:**
                            - 🔵 **Blue Box (Forehead)**: Primary rPPG signal source - most reliable for heart rate detection
                            - 🟡 **Yellow Box (Left Cheek)**: Secondary signal for verification
                            - 🩵 **Cyan Box (Right Cheek)**: Secondary signal for verification
                            - 🟪 **Magenta Box (Chin)**: Additional signal source
                            - 🟢 **Green Box**: Detected face boundary
                            
                            **Signal Quality Indicators:**
                            - **High score (>0.6)**: Strong, consistent rPPG signal - characteristic of REAL videos
                            - **Medium score (0.45-0.6)**: Moderate signal quality - could be UNCERTAIN
                            - **Low score (<0.45)**: Weak or inconsistent signal - common in DEEPFAKES
                            
                            **Heart Rate:**
                            - Normal human heart rate: 60-100 BPM
                            - Unusual heart rates may indicate manipulation
                            """)
                    else:
                        st.error("Could not process video. Make sure a face is clearly visible.")
            
            # Cleanup
            Path(video_path).unlink()


if __name__ == "__main__":
    main()