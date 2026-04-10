import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa

# Directory with organized audio files
DATA_DIR = "RESIZED"

# Categories are subfolders
categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']
print(f"Detected categories for training: {categories}")

X = []
y = []

# Function to extract features from audio
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        audio = audio / np.max(np.abs(audio))
       delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Iterate over categories
total_files = 0
for idx, cat in enumerate(categories):
    category_path = os.path.join(DATA_DIR, cat)
    # Grab all .wav files, any case
    files = glob.glob(os.path.join(category_path, "*.wav")) + \
            glob.glob(os.path.join(category_path, "*.WAV"))
import random

random.shuffle(files)
files = files[:24]
    if not files:
        print(f"⚠️  Warning: No audio files found for category '{cat}'!")
        continue

    for f in files:
        features = extract_features(f)
        if features is not None:
            X.append(features)
            y.append(idx)
            total_files += 1

print(f"Total samples found: {total_files}")

if total_files == 0:
    raise ValueError("No audio files found! Check RESIZED folder structure and file extensions.")

X = np.array(X)
y = np.array(y)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save model
joblib.dump(clf, "cough_model.pkl")
print("✅ Model trained and saved as 'cough_model.pkl'")
