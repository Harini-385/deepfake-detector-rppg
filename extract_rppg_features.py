"""
extract_rppg_features_improved.py
Enhanced rPPG feature extraction with better discrimination
"""

import cv2
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, find_peaks
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ImprovedRPPGExtractor:
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
    
    def extract_forehead_roi(self, frame, face_bbox):
        x, y, w, h = face_bbox
        # Forehead region
        fy1 = max(0, y + int(h * 0.1))
        fy2 = min(frame.shape[0], y + int(h * 0.35))
        fx1 = max(0, x + int(w * 0.2))
        fx2 = min(frame.shape[1], x + int(w * 0.8))
        
        if fy1 >= fy2 or fx1 >= fx2:
            return None
        return frame[fy1:fy2, fx1:fx2]
    
    def extract_multiple_rois(self, frame, face_bbox):
        """Extract multiple ROIs for better signal"""
        x, y, w, h = face_bbox
        rois = []
        
        # Forehead
        forehead = self.extract_forehead_roi(frame, face_bbox)
        if forehead is not None:
            rois.append(forehead)
        
        # Left cheek
        lx1 = max(0, x + int(w * 0.05))
        lx2 = max(0, x + int(w * 0.25))
        ly1 = max(0, y + int(h * 0.5))
        ly2 = min(frame.shape[0], y + int(h * 0.8))
        if ly1 < ly2 and lx1 < lx2:
            left_cheek = frame[ly1:ly2, lx1:lx2]
            if left_cheek.size > 0:
                rois.append(left_cheek)
        
        # Right cheek
        rx1 = min(frame.shape[1], x + int(w * 0.75))
        rx2 = min(frame.shape[1], x + int(w * 0.95))
        ry1 = max(0, y + int(h * 0.5))
        ry2 = min(frame.shape[0], y + int(h * 0.8))
        if ry1 < ry2 and rx1 < rx2:
            right_cheek = frame[ry1:ry2, rx1:rx2]
            if right_cheek.size > 0:
                rois.append(right_cheek)
        
        return rois
    
    def extract_chrom_signal_from_roi(self, roi):
        """Extract CHROM signal from single ROI"""
        if roi is None or roi.size == 0:
            return None
        
        roi_float = roi.astype(np.float32)
        r = np.mean(roi_float[:, :, 2]) / 255.0
        g = np.mean(roi_float[:, :, 1]) / 255.0
        b = np.mean(roi_float[:, :, 0]) / 255.0
        
        # CHROM method
        Xs = 3 * r - 2 * g
        Ys = 1.5 * r + g - 1.5 * b
        
        norm = np.sqrt(Xs**2 + Ys**2)
        if norm > 0:
            Xs = Xs / norm
            Ys = Ys / norm
        
        return Xs + Ys
    
    def compute_advanced_features(self, chrom_signals):
        """Compute advanced features for better discrimination"""
        if len(chrom_signals) < 30:
            return None
        
        signals = np.array(chrom_signals)
        
        # Detrend
        signals = detrend(signals)
        
        # Normalize
        if np.std(signals) > 0:
            signals = signals / np.std(signals)
        
        # Apply bandpass
        try:
            filtered = filtfilt(self.b, self.a, signals)
        except:
            filtered = signals
        
        # FFT analysis
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, 1.0/self.fps)
        fft_vals = np.abs(np.fft.rfft(filtered))
        
        # Heart rate detection
        mask = (freqs >= self.lowcut) & (freqs <= self.highcut)
        if np.any(mask):
            freq_range = freqs[mask]
            power_range = fft_vals[mask]
            hr_freq = freq_range[np.argmax(power_range)]
            heart_rate = hr_freq * 60
        else:
            heart_rate = 0
        
        # Signal quality metrics
        # 1. Peak-to-peak regularity
        peaks, _ = find_peaks(filtered, height=0.2*np.std(filtered), distance=10)
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            regularity = 1.0 - min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-6))
        else:
            regularity = 0.3
        
        # 2. Signal-to-noise ratio
        total_power = np.sum(fft_vals ** 2)
        hr_power = np.sum(fft_vals[mask] ** 2) if np.any(mask) else 0
        snr = hr_power / (total_power - hr_power + 1e-6)
        snr_score = min(1.0, snr / 8.0)
        
        # 3. Spectral entropy (measure of signal complexity)
        psd = fft_vals / (np.sum(fft_vals) + 1e-6)
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-6))
        spectral_entropy = min(1.0, spectral_entropy / 5.0)
        
        # 4. Heart rate plausibility (60-100 BPM ideal)
        if 60 <= heart_rate <= 100:
            hr_score = 1.0
        elif 50 <= heart_rate <= 110:
            hr_score = 0.7
        elif heart_rate > 0:
            hr_score = 0.3
        else:
            hr_score = 0.0
        
        # 5. Signal variation (higher variation often indicates real)
        variation = np.std(filtered)
        variation_score = min(1.0, variation / 1.5)
        
        # 6. Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(filtered)))
        zcr = zero_crossings / len(filtered)
        zcr_score = min(1.0, zcr * 5)
        
        features = {
            'heart_rate': float(heart_rate),
            'hr_score': float(hr_score),
            'regularity': float(regularity),
            'snr_score': float(snr_score),
            'spectral_entropy': float(spectral_entropy),
            'variation_score': float(variation_score),
            'zcr_score': float(zcr_score),
            'overall_quality': float(0.4*hr_score + 0.3*regularity + 0.3*snr_score)
        }
        
        return features
    
    def process_video(self, video_path, max_frames=450):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30
        
        self.fps = fps
        self._init_filter()
        
        process_every = max(1, int(fps / 15))
        all_chrom_signals = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
                
                if len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    rois = self.extract_multiple_rois(frame, best_face)
                    
                    # Average signals from multiple ROIs
                    roi_signals = []
                    for roi in rois:
                        chrom = self.extract_chrom_signal_from_roi(roi)
                        if chrom is not None:
                            roi_signals.append(chrom)
                    
                    if roi_signals:
                        avg_signal = np.mean(roi_signals)
                        all_chrom_signals.append(avg_signal)
                
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        if len(all_chrom_signals) < 45:
            return None
        
        features = self.compute_advanced_features(all_chrom_signals)
        if features:
            features['signal_length'] = len(all_chrom_signals)
            features['face_detection_rate'] = processed_frames / max(frame_count, 1)
        
        return features

def process_video_file(video_path, is_real):
    try:
        extractor = ImprovedRPPGExtractor()
        features = extractor.process_video(str(video_path))
        
        if features and features['signal_length'] >= 45:
            features['is_real'] = is_real
            features['video_path'] = str(video_path)
            features['video_name'] = video_path.name
            return features
    except Exception as e:
        pass
    return None

def scan_celeb_df_dataset(base_path="Celeb-DF"):
    base_path = Path(base_path)
    real_videos = []
    fake_videos = []
    
    if not base_path.exists():
        return real_videos, fake_videos
    
    for real_dir in ['Celeb-real', 'YouTube-real']:
        dir_path = base_path / real_dir
        if dir_path.exists():
            real_videos.extend(list(dir_path.glob("*.mp4")))
            real_videos.extend(list(dir_path.glob("*.avi")))
    
    fake_dir = base_path / "Celeb-synthesis"
    if fake_dir.exists():
        fake_videos.extend(list(fake_dir.glob("*.mp4")))
        fake_videos.extend(list(fake_dir.glob("*.avi")))
    
    return real_videos, fake_videos

def main():
    print("="*70)
    print("IMPROVED RPPG FEATURE EXTRACTOR")
    print("="*70)
    
    features_file = "improved_rppg_features.json"
    
    if Path(features_file).exists():
        print(f"\n✅ Features already extracted!")
        response = input("Re-extract? (y/N): ")
        if response.lower() != 'y':
            with open(features_file, 'r') as f:
                features = json.load(f)
            print(f"Loaded {len(features)} features")
            return features
    
    real_videos, fake_videos = scan_celeb_df_dataset()
    print(f"\nFound: {len(real_videos)} real, {len(fake_videos)} fake videos")
    
    # Process more videos for better training
    max_videos = int(input("How many videos to process? (default=200): ") or "200")
    
    all_videos = [(v, True) for v in real_videos[:max_videos//2]] + [(v, False) for v in fake_videos[:max_videos//2]]
    
    features_list = []
    for video, is_real in tqdm(all_videos, desc="Processing"):
        result = process_video_file(video, is_real)
        if result:
            features_list.append(result)
    
    with open(features_file, 'w') as f:
        json.dump(features_list, f, indent=2)
    
    print(f"\n✅ Extracted {len(features_list)} features")
    
    # Show statistics
    real_feats = [f for f in features_list if f['is_real']]
    fake_feats = [f for f in features_list if not f['is_real']]
    
    if real_feats and fake_feats:
        print(f"\n📊 Feature Comparison:")
        print(f"   Overall Quality - Real: {np.mean([f['overall_quality'] for f in real_feats]):.3f}, Fake: {np.mean([f['overall_quality'] for f in fake_feats]):.3f}")
        print(f"   HR Score - Real: {np.mean([f['hr_score'] for f in real_feats]):.3f}, Fake: {np.mean([f['hr_score'] for f in fake_feats]):.3f}")
        print(f"   Regularity - Real: {np.mean([f['regularity'] for f in real_feats]):.3f}, Fake: {np.mean([f['regularity'] for f in fake_feats]):.3f}")
        print(f"   SNR Score - Real: {np.mean([f['snr_score'] for f in real_feats]):.3f}, Fake: {np.mean([f['snr_score'] for f in fake_feats]):.3f}")

if __name__ == "__main__":
    main()