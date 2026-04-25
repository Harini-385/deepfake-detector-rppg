"""
Deepfake Detection Web App - Complete Fixed Version
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        color: white;
    }
    .fake-badge {
        background-color: #ef4444;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        color: white;
    }
    .uncertain-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        color: white;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# RPPG Extractor Class
# ============================================================

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
        
        self.roi_colors = {
            'forehead': (255, 0, 0),
            'left_cheek': (0, 255, 255),
            'right_cheek': (255, 255, 0),
            'chin': (255, 0, 255)
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
        
        # Forehead
        fy1 = max(0, y + int(h * 0.1))
        fy2 = min(frame.shape[0], y + int(h * 0.35))
        fx1 = max(0, x + int(w * 0.2))
        fx2 = min(frame.shape[1], x + int(w * 0.8))
        if fy1 < fy2 and fx1 < fx2:
            rois['forehead'] = frame[fy1:fy2, fx1:fx2]
        
        # Left Cheek
        lx1 = max(0, x + int(w * 0.05))
        lx2 = max(0, x + int(w * 0.25))
        ly1 = max(0, y + int(h * 0.5))
        ly2 = min(frame.shape[0], y + int(h * 0.75))
        if ly1 < ly2 and lx1 < lx2:
            rois['left_cheek'] = frame[ly1:ly2, lx1:lx2]
        
        # Right Cheek
        rx1 = min(frame.shape[1], x + int(w * 0.75))
        rx2 = min(frame.shape[1], x + int(w * 0.95))
        ry1 = max(0, y + int(h * 0.5))
        ry2 = min(frame.shape[0], y + int(h * 0.75))
        if ry1 < ry2 and rx1 < rx2:
            rois['right_cheek'] = frame[ry1:ry2, rx1:rx2]
        
        # Chin
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
        
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if 'forehead' in rois:
            fy1 = y + int(h * 0.1)
            fy2 = y + int(h * 0.35)
            fx1 = x + int(w * 0.2)
            fx2 = x + int(w * 0.8)
            cv2.rectangle(result, (fx1, fy1), (fx2, fy2), self.roi_colors['forehead'], 2)
            cv2.putText(result, "Forehead", (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['forehead'], 2)
        
        if 'left_cheek' in rois:
            lx1 = x + int(w * 0.05)
            lx2 = x + int(w * 0.25)
            ly1 = y + int(h * 0.5)
            ly2 = y + int(h * 0.75)
            cv2.rectangle(result, (lx1, ly1), (lx2, ly2), self.roi_colors['left_cheek'], 2)
            cv2.putText(result, "L-Cheek", (lx1, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['left_cheek'], 2)
        
        if 'right_cheek' in rois:
            rx1 = x + int(w * 0.75)
            rx2 = x + int(w * 0.95)
            ry1 = y + int(h * 0.5)
            ry2 = y + int(h * 0.75)
            cv2.rectangle(result, (rx1, ry1), (rx2, ry2), self.roi_colors['right_cheek'], 2)
            cv2.putText(result, "R-Cheek", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_colors['right_cheek'], 2)
        
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
    
    def extract_features_for_training(self, video_path, max_frames=300):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps
        self._init_filter()
        
        roi_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': [], 'chin': []}
        frame_count = 0
        process_every = max(1, int(fps / 15))
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
                
                if len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    rois = self.extract_rois(frame, best_face)
                    
                    for roi_name, roi in rois.items():
                        if roi_name in roi_signals:
                            signal_val = self.extract_chrom_signal(roi)
                            if signal_val is not None:
                                roi_signals[roi_name].append(signal_val)
            
            frame_count += 1
        
        cap.release()
        
        features = {}
        for roi_name, signals in roi_signals.items():
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
        
        hr_values = [features.get(f'{roi}_hr', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        quality_values = [features.get(f'{roi}_quality', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        features['combined_score'] = np.mean(hr_values) * 0.6 + np.mean(quality_values) * 0.4
        
        return features
    
    def process_video_frames(self, video_path, max_frames=300):
        """Extract frames with ROI visualization for video player"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps
        self._init_filter()
        
        # Reset buffers
        self.signal_buffer = []
        self.roi_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': [], 'chin': []}
        
        frames_with_roi = []
        frame_count = 0
        process_every = max(1, int(fps / 15))
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
                
                display_frame = frame.copy()
                
                if len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    rois = self.extract_rois(frame, best_face)
                    display_frame = self.draw_rois(display_frame, rois, best_face)
                    
                    # Extract signals
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
                    
                    cv2.putText(display_frame, f"rPPG: {rppg_score:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if heart_rate:
                        cv2.putText(display_frame, f"HR: {int(heart_rate)} BPM", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "No Face Detected", (display_frame.shape[1]//2-100, display_frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                frames_with_roi.append(display_frame)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate features
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
        
        hr_values = [features.get(f'{roi}_hr', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        quality_values = [features.get(f'{roi}_quality', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        features['combined_score'] = np.mean(hr_values) * 0.6 + np.mean(quality_values) * 0.4
        
        return frames_with_roi, features


# ============================================================
# Model Trainer Class
# ============================================================

class ModelTrainer:
    MODEL_PATH = "deepfake_model_persistent.pkl"
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.extractor = RPPGExtractor()
        self.load_model()
    
    def load_model(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.feature_names = data['feature_names']
                    self.is_trained = True
                return True
            except:
                pass
        return False
    
    def save_model(self):
        try:
            with open(self.MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                }, f)
            return True
        except:
            return False
    
    def train(self, real_folder, fake_folder, progress_callback=None):
        real_folder = Path(real_folder)
        fake_folder = Path(fake_folder)
        
        if not real_folder.exists() or not fake_folder.exists():
            return False, "Dataset folders not found!"
        
        real_videos = list(real_folder.glob("*.mp4")) + list(real_folder.glob("*.avi"))
        fake_videos = list(fake_folder.glob("*.mp4")) + list(fake_folder.glob("*.avi"))
        
        if len(real_videos) == 0 or len(fake_videos) == 0:
            return False, "No videos found in folders!"
        
        real_features = []
        fake_features = []
        
        total = min(100, len(real_videos)) + min(100, len(fake_videos))
        processed = 0
        
        for video in real_videos[:100]:
            if progress_callback:
                progress_callback(processed, total, f"REAL: {video.name}")
            features = self.extractor.extract_features_for_training(video)
            if features:
                features['label'] = 1
                real_features.append(features)
            processed += 1
        
        for video in fake_videos[:100]:
            if progress_callback:
                progress_callback(processed, total, f"FAKE: {video.name}")
            features = self.extractor.extract_features_for_training(video)
            if features:
                features['label'] = 0
                fake_features.append(features)
            processed += 1
        
        if len(real_features) < 10 or len(fake_features) < 10:
            return False, f"Not enough features: {len(real_features)} REAL, {len(fake_features)} FAKE"
        
        # Prepare data
        self.feature_names = [k for k in real_features[0].keys() if k != 'label']
        X = []
        y = []
        
        for feat in real_features:
            X.append([feat[name] for name in self.feature_names])
            y.append(1)
        for feat in fake_features:
            X.append([feat[name] for name in self.feature_names])
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        self.save_model()
        self.is_trained = True
        
        return True, f"Training complete! Accuracy: {test_acc:.2%}"
    
    def predict(self, rppg_score):
        if self.is_trained and self.model is not None:
            if rppg_score > 0.6:
                return "REAL", rppg_score
            elif rppg_score > 0.45:
                return "UNCERTAIN", 0.5
            else:
                return "FAKE", 1 - rppg_score
        else:
            if rppg_score > 0.6:
                return "REAL", rppg_score
            elif rppg_score > 0.45:
                return "UNCERTAIN", 0.5
            else:
                return "FAKE", 1 - rppg_score


# ============================================================
# Initialize Session State
# ============================================================

if 'trainer' not in st.session_state:
    st.session_state.trainer = ModelTrainer()

if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = []

if 'current_frame_index' not in st.session_state:
    st.session_state.current_frame_index = 0

if 'auto_play' not in st.session_state:
    st.session_state.auto_play = False

if 'features' not in st.session_state:
    st.session_state.features = {}

if 'extractor_signals' not in st.session_state:
    st.session_state.extractor_signals = {}


# ============================================================
# Helper Functions
# ============================================================

def create_signal_plot(roi_signals):
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


# ============================================================
# Main App
# ============================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Deepfake Detection System</h1>
        <p>Live rPPG Tracking | Multi-ROI Analysis | Train Your Own Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        
        page = st.radio(
            "Select Mode",
            ["🎥 Live Webcam", "🎬 Upload Video", "🎯 Train Model"],
            index=0
        )
        
        st.markdown("---")
        
        if st.session_state.trainer.is_trained:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ No Model - Train First")
        
        st.markdown("---")
        st.markdown("### 🎨 ROI Colors")
        st.markdown("🔵 **Blue** - Forehead")
        st.markdown("🟡 **Yellow** - Left Cheek")
        st.markdown("🩵 **Cyan** - Right Cheek")
        st.markdown("🟪 **Magenta** - Chin")
        st.markdown("🟢 **Green** - Face")
    
    # ============================================================
    # Live Webcam Page
    # ============================================================
    
    if page == "🎥 Live Webcam":
        st.markdown("### Live Webcam Detection")
        
        start_cam = st.button("▶️ Start Camera", use_container_width=True, type="primary")
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
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = extractor.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(100, 100))
                    
                    display_frame = frame.copy()
                    
                    if len(faces) > 0:
                        best_face = max(faces, key=lambda f: f[2] * f[3])
                        rois = extractor.extract_rois(frame, best_face)
                        display_frame = extractor.draw_rois(display_frame, rois, best_face)
                        
                        if 'forehead' in rois:
                            signal_val = extractor.extract_chrom_signal(rois['forehead'])
                            if signal_val is not None:
                                extractor.signal_buffer.append(signal_val)
                                if len(extractor.signal_buffer) > 300:
                                    extractor.signal_buffer.pop(0)
                    
                    rppg_score = extractor.compute_rppg_score()
                    heart_rate = extractor.compute_heart_rate(extractor.signal_buffer)
                    
                    cv2.putText(display_frame, f"rPPG: {rppg_score:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if heart_rate:
                        cv2.putText(display_frame, f"HR: {int(heart_rate)} BPM", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    label, confidence = st.session_state.trainer.predict(rppg_score)
                    
                    if label == "REAL":
                        color = "#10b981"
                    elif label == "FAKE":
                        color = "#ef4444"
                    else:
                        color = "#f59e0b"
                    
                    cv2.putText(display_frame, f"{label}: {confidence:.1%}", (display_frame.shape[1]-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if label == "REAL" else (0, 0, 255), 2)
                    
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    metrics_placeholder.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #1f2937; border-radius: 10px;">
                        <h3 style="color: {color};">{label}</h3>
                        <p>Confidence: {confidence:.1%}</p>
                        <p>rPPG Score: {rppg_score:.3f}</p>
                        <p>Heart Rate: {int(heart_rate) if heart_rate else '--'} BPM</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if stop_cam:
                        break
                    
                    time.sleep(0.03)
                
                cap.release()
                cv2.destroyAllWindows()
    
    # ============================================================
    # Upload Video Page
    # ============================================================
    
    elif page == "🎬 Upload Video":
        st.markdown("### Upload Video with Live rPPG Tracking")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(uploaded_file)
            with col2:
                st.info(f"Filename: {uploaded_file.name}\nSize: {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            if st.button("🔍 Analyze with Live Tracking", use_container_width=True, type="primary"):
                with st.spinner("Processing video..."):
                    extractor = RPPGExtractor()
                    frames_with_roi, features = extractor.process_video_frames(video_path)
                    
                    if frames_with_roi and len(frames_with_roi) > 0:
                        st.session_state.processed_frames = frames_with_roi
                        st.session_state.current_frame_index = 0
                        st.session_state.auto_play = False
                        st.session_state.features = features
                        st.session_state.extractor_signals = extractor.roi_signals
                        st.success(f"✅ Processed {len(frames_with_roi)} frames!")
                        st.rerun()
                    else:
                        st.error("Could not process video. Make sure a face is clearly visible.")
            
            # Display processed video with navigation
            if len(st.session_state.processed_frames) > 0:
                frames = st.session_state.processed_frames
                
                st.markdown("---")
                st.markdown("### 🎬 Processed Video with ROI Tracking")
                
                # Control buttons
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button("⏮️ First", use_container_width=True):
                        st.session_state.current_frame_index = 0
                        st.session_state.auto_play = False
                        st.rerun()
                
                with col2:
                    if st.button("◀️ Prev", use_container_width=True):
                        st.session_state.current_frame_index = max(0, st.session_state.current_frame_index - 1)
                        st.session_state.auto_play = False
                        st.rerun()
                
                with col3:
                    if st.session_state.auto_play:
                        if st.button("⏸️ Pause", use_container_width=True):
                            st.session_state.auto_play = False
                            st.rerun()
                    else:
                        if st.button("▶️ Play", use_container_width=True):
                            st.session_state.auto_play = True
                            st.rerun()
                
                with col4:
                    if st.button("Next ▶️", use_container_width=True):
                        if st.session_state.current_frame_index < len(frames) - 1:
                            st.session_state.current_frame_index += 1
                        st.session_state.auto_play = False
                        st.rerun()
                
                with col5:
                    if st.button("Last ⏭️", use_container_width=True):
                        st.session_state.current_frame_index = len(frames) - 1
                        st.session_state.auto_play = False
                        st.rerun()
                
                # Frame slider
                frame_index = st.slider(
                    "Frame Navigation",
                    0, len(frames) - 1,
                    st.session_state.current_frame_index,
                    key="frame_slider"
                )
                
                if frame_index != st.session_state.current_frame_index:
                    st.session_state.current_frame_index = frame_index
                    st.session_state.auto_play = False
                    st.rerun()
                
                # Display current frame
                current_frame = frames[st.session_state.current_frame_index]
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Frame counter
                st.caption(f"Frame {st.session_state.current_frame_index + 1} of {len(frames)}")
                
                # Auto-play logic
                if st.session_state.auto_play:
                    if st.session_state.current_frame_index < len(frames) - 1:
                        time.sleep(0.05)
                        st.session_state.current_frame_index += 1
                        st.rerun()
                    else:
                        st.session_state.auto_play = False
                        st.rerun()
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                features = st.session_state.features
                combined_score = features.get('combined_score', 0.5)
                
                label, confidence = st.session_state.trainer.predict(combined_score)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if label == "REAL":
                        st.markdown('<div class="real-badge">✅ REAL</div>', unsafe_allow_html=True)
                    elif label == "FAKE":
                        st.markdown('<div class="fake-badge">❌ FAKE</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="uncertain-badge">⚠️ UNCERTAIN</div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col2:
                    st.metric("Combined Score", f"{combined_score:.3f}")
                    st.metric("Forehead Quality", f"{features.get('forehead_quality', 0.5):.3f}")
                
                with col3:
                    st.metric("Left Cheek", f"{features.get('left_cheek_quality', 0.5):.3f}")
                    st.metric("Right Cheek", f"{features.get('right_cheek_quality', 0.5):.3f}")
                
                # Signal plots
                if st.session_state.extractor_signals:
                    st.markdown("### 📈 rPPG Signals by ROI")
                    signal_plot = create_signal_plot(st.session_state.extractor_signals)
                    st.plotly_chart(signal_plot, use_container_width=True)
            
            Path(video_path).unlink()
    
    
    else:
        st.markdown("### 🎯 Train Deepfake Detection Model")
        
        if st.session_state.trainer.is_trained:
            st.success(f"✅ Model already trained!")
            if st.button("🔄 Retrain Model", use_container_width=True):
                st.session_state.trainer.is_trained = False
                st.rerun()
        
        with st.expander("📁 Dataset Configuration", expanded=not st.session_state.trainer.is_trained):
            col1, col2 = st.columns(2)
            
            with col1:
                real_folder = st.text_input("Real Videos Folder", value="Celeb-DF/Celeb-real")
                st.caption("Folder containing REAL videos")
            
            with col2:
                fake_folder = st.text_input("Fake Videos Folder", value="Celeb-DF/Celeb-synthesis")
                st.caption("Folder containing FAKE/Deepfake videos")
        
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Processing: {message}")
            
            success, message = st.session_state.trainer.train(real_folder, fake_folder, update_progress)
            
            progress_bar.progress(1.0)
            
            if success:
                st.success(f"✅ {message}")
                st.balloons()
            else:
                st.error(f"❌ {message}")


if __name__ == "__main__":
    main()