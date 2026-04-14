import os
import glob
import random
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from utils import extract_features

DATA_DIR = "RESIZED"

categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']

X = []
y = []

print("Training categories:", categories)

for idx, cat in enumerate(categories):
    folder = os.path.join(DATA_DIR, cat)

    files = glob.glob(os.path.join(folder, "*.wav")) + \
            glob.glob(os.path.join(folder, "*.WAV"))

    print(cat, "files:", len(files))

    random.shuffle(files)

    for f in files:
        try:
            y_audio, sr = librosa.load(f, sr=22050)

            # Pad/Trim to 6 sec
            target_len = 6 * sr
            if len(y_audio) < target_len:
                y_audio = np.pad(y_audio, (0, target_len - len(y_audio)))
            else:
                y_audio = y_audio[:target_len]

            features = extract_features(y_audio, sr)

            if features is not None:
                X.append(features)
                y.append(idx)

        except Exception as e:
            print("Error:", f, e)

print("Total samples:", len(X))

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🔥 Accuracy:", acc)

# Save
joblib.dump(model, "cough_model.pkl")
joblib.dump(categories, "labels.pkl")

print("✅ Model & labels saved")
