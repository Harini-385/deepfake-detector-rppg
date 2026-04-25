"""
train_model.py
Train deepfake detection model on Celeb-DF dataset
Run this once before using the GUI
"""

import cv2
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, find_peaks
from scipy.fft import fft2, fftshift
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ROIRPPGExtractor:
    def __init__(self, fps=30):
        self.fps = fps
        self.lowcut = 0.75
        self.highcut = 3.5
        self.order = 4
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._init_filter()
    
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
    
    def compute_heart_rate(self, signal_buffer):
        if len(signal_buffer) < 45:
            return None
        signal_data = np.array(signal_buffer)
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
    
    def extract_features_from_video(self, video_path, max_frames=300):
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
        face_detected_frames = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
                
                if len(faces) > 0:
                    face_detected_frames += 1
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    rois = self.extract_rois(frame, best_face)
                    
                    for roi_name, roi in rois.items():
                        if roi_name in roi_signals:
                            signal_val = self.extract_chrom_signal(roi)
                            if signal_val is not None:
                                roi_signals[roi_name].append(signal_val)
            
            frame_count += 1
        
        cap.release()
        
        # Check if we have enough data
        valid_rois = {k: v for k, v in roi_signals.items() if len(v) >= 30}
        if len(valid_rois) < 2:
            return None
        
        features = {}
        
        for roi_name, signals in valid_rois.items():
            hr = self.compute_heart_rate(signals)
            signal_std = np.std(signals)
            
            if hr is None:
                hr_score = 0.3
            elif 60 <= hr <= 100:
                hr_score = 1.0
            elif 50 <= hr <= 110:
                hr_score = 0.6
            else:
                hr_score = 0.2
            
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
        
        features['face_detection_rate'] = face_detected_frames / max(frame_count // process_every, 1)
        
        # Calculate combined score
        hr_values = [features.get(f'{roi}_hr', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        quality_values = [features.get(f'{roi}_quality', 0.5) for roi in ['forehead', 'left_cheek', 'right_cheek', 'chin']]
        
        features['combined_score'] = np.mean(hr_values) * 0.6 + np.mean(quality_values) * 0.4
        
        return features

def main():
    print("="*60)
    print("DEEPFAKE DETECTION MODEL TRAINING")
    print("="*60)
    
    real_folder = Path("Celeb-DF/Celeb-real")
    fake_folder = Path("Celeb-DF/Celeb-synthesis")
    
    if not real_folder.exists() or not fake_folder.exists():
        print("❌ Dataset folders not found!")
        print("Expected: Celeb-DF/Celeb-real and Celeb-DF/Celeb-synthesis")
        return
    
    real_videos = list(real_folder.glob("*.mp4")) + list(real_folder.glob("*.avi"))
    fake_videos = list(fake_folder.glob("*.mp4")) + list(fake_folder.glob("*.avi"))
    
    print(f"\n📹 Found: {len(real_videos)} REAL, {len(fake_videos)} FAKE videos")
    
    extractor = ROIRPPGExtractor()
    
    print("\n📊 Extracting features from REAL videos...")
    real_features = []
    for video in tqdm(real_videos[:150], desc="REAL"):
        features = extractor.extract_features_from_video(video)
        if features:
            features['label'] = 1
            real_features.append(features)
    
    print("\n📊 Extracting features from FAKE videos...")
    fake_features = []
    for video in tqdm(fake_videos[:150], desc="FAKE"):
        features = extractor.extract_features_from_video(video)
        if features:
            features['label'] = 0
            fake_features.append(features)
    
    print(f"\n✅ Extracted: {len(real_features)} REAL, {len(fake_features)} FAKE")
    
    # Prepare data
    feature_names = [k for k in real_features[0].keys() if k != 'label']
    X = []
    y = []
    
    for feat in real_features:
        X.append([feat[name] for name in feature_names])
        y.append(1)
    for feat in fake_features:
        X.append([feat[name] for name in feature_names])
        y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\n📊 Results:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Save model
    with open('deepfake_model_trained.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names}, f)
    
    print("\n💾 Model saved as: deepfake_model_trained.pkl")
    print("\n✅ Training complete! You can now run deepfake_gui.py")

if __name__ == "__main__":
    main()