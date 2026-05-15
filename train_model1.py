import os
import glob
import random
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from utils import extract_features

# -----------------------------
# Dataset
# -----------------------------
DATA_DIR = "RESIZED"

categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']

X = []
y = []

print("Training Started...")

# -----------------------------
# Load Data
# -----------------------------
for cat in categories:

    folder = os.path.join(DATA_DIR, cat)

    files = glob.glob(os.path.join(folder, "*.wav")) + \
            glob.glob(os.path.join(folder, "*.WAV"))

    print(cat, "files:", len(files))

    random.shuffle(files)

    for f in files:

        try:
            y_audio, sr = librosa.load(f, sr=22050, mono=True)

            features = extract_features(y_audio, sr)

            if features is not None:
                X.append(features)
                y.append(cat)

        except Exception as e:
            print("Error:", f, e)

# -----------------------------
# Convert
# -----------------------------
X = np.array(X)
y = np.array(y)

print("\nSamples:", len(X))
print("Feature shape:", X.shape)

X, y = shuffle(X, y, random_state=42)

# -----------------------------
# Encode labels
# -----------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder.classes_, "labels.pkl")

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# One-hot
# -----------------------------
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -----------------------------
# ANN Model
# -----------------------------
model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(len(categories), activation='softmax'))

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\n🔥 Accuracy:", acc)

# -----------------------------
# Save model
# -----------------------------
model.save("ann_cough_model.h5")

print("✅ Model Saved")
