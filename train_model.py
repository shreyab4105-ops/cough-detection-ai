import os
import glob
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import librosa

# --- Directory ---
DATA_DIR = "RESIZED"

# --- Categories (MUST match app.py) ---
categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']
print(f"Detected categories for training: {categories}")

X = []
y = []

# --- Feature Extraction (IMPROVED) ---
def extract_features(file_path):
    try:
        y_audio, sr = librosa.load(file_path, sr=22050)

        # Normalize
        if np.max(np.abs(y_audio)) != 0:
            y_audio = y_audio / np.max(np.abs(y_audio))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)

        # Delta features
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Extra features (🔥 important)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_audio))
        chroma = np.mean(librosa.feature.chroma_stft(y=y_audio, sr=sr))
        spectral = np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr))

        # Combine → 42 features total
        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1),
            zcr,
            chroma,
            spectral
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Load Data ---
total_files = 0

for idx, cat in enumerate(categories):
    category_path = os.path.join(DATA_DIR, cat)

    files = glob.glob(os.path.join(category_path, "*.wav")) + \
            glob.glob(os.path.join(category_path, "*.WAV"))

    # Shuffle + balance
    random.shuffle(files)
    files = files[:40]   # 🔥 increased from 24 → better learning

    if not files:
        print(f"⚠️ Warning: No files for {cat}")
        continue

    for f in files:
        features = extract_features(f)
        if features is not None:
            X.append(features)
            y.append(idx)
            total_files += 1

print(f"Total samples used: {total_files}")

if total_files == 0:
    raise ValueError("No audio files found!")

# Convert
X = np.array(X)
y = np.array(y)

# Shuffle dataset
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Model ---
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)

# Train
clf.fit(X_train, y_train)

# Test
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("🔥 Accuracy:", acc)

# --- Save Model ---
joblib.dump(clf, "cough_model.pkl")
print("✅ Model saved as 'cough_model.pkl'")
