import os
import glob
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa

# Directory with organized audio files
DATA_DIR = "RESIZED"

# ✅ FIXED CATEGORY ORDER (VERY IMPORTANT)
categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']
print(f"Detected categories for training: {categories}")

X = []
y = []

# ✅ Feature Extraction (IMPROVED)
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)

        # Normalize audio
        if np.max(np.abs(audio)) != 0:
            audio = audio / np.max(np.abs(audio))

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Delta features
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Combine features
        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ✅ Iterate over categories
total_files = 0

for idx, cat in enumerate(categories):
    category_path = os.path.join(DATA_DIR, cat)

    # Get all wav files
    files = glob.glob(os.path.join(category_path, "*.wav")) + \
            glob.glob(os.path.join(category_path, "*.WAV"))

    # 🔥 Shuffle and balance dataset
    random.shuffle(files)
    files = files[:24]   # take equal samples from each class

    if not files:
        print(f"⚠️ Warning: No audio files found for category '{cat}'!")
        continue

    for f in files:
        features = extract_features(f)
        if features is not None:
            X.append(features)
            y.append(idx)
            total_files += 1

print(f"Total samples used: {total_files}")

# Safety check
if total_files == 0:
    raise ValueError("No audio files found! Check RESIZED folder.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# ✅ Improved model
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

clf.fit(X, y)

# Save model
joblib.dump(clf, "cough_model.pkl")

print("✅ Model trained and saved as 'cough_model.pkl'")
