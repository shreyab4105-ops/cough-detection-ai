import os
import numpy as np
import librosa
import pywt
import joblib
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, 'Source')

# Automatically detect all categories in Source/
categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
print("Detected categories for training:", categories)

X = []
y = []

def extract_features(y, sr=22050, n_mfcc=13):
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=n_mfcc)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        coeffs, _ = pywt.cwt(y, np.arange(1, 32), 'morl', sampling_period=1/sr)
        wavelet_mean = np.mean(np.abs(coeffs), axis=1)
        wavelet_std = np.std(np.abs(coeffs), axis=1)
        energy = np.sum(librosa.feature.rms(y=y))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(contrast, axis=1),
            wavelet_mean,
            wavelet_std,
            [energy, flatness]
        ])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Load all audio files
for idx, cat in enumerate(categories):
    folder = os.path.join(SOURCE_DIR, cat)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            path = os.path.join(folder, file)
            try:
                y, sr = librosa.load(path, sr=22050)
                target_len = 6 * sr
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)))
                else:
                    y = y[:target_len]

                features = extract_features(y, sr)
                if features is not None:
                    X.append(features)
                    y.append(cat)
            except Exception as e:
                print(f"Skipping {path}: {e}")

print(f"Total samples found: {len(X)}")

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
joblib.dump(clf, 'cough_model.pkl')
print("Model trained and saved as cough_model.pkl")
