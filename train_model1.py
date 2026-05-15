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

# ---------------------------------------------------
# Dataset Path
# ---------------------------------------------------
DATA_DIR = "RESIZED"

categories = [
    'Asthama',
    'CROUP',
    'LTRI',
    'NORMAL',
    'PNEUMONIA',
    'URTI'
]

# ---------------------------------------------------
# Data storage
# ---------------------------------------------------
X = []
y = []

print("Training Started...")

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------
for cat in categories:

    folder = os.path.join(DATA_DIR, cat)

    files = glob.glob(os.path.join(folder, "*.wav")) + \
            glob.glob(os.path.join(folder, "*.WAV"))

    print(f"{cat} files: {len(files)}")

    random.shuffle(files)

    for f in files:

        try:

            audio, sr = librosa.load(f, sr=22050)

            # pad/trim to 6 sec
            target_len = 6 * sr

            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            # normalize
            if np.max(np.abs(audio)) != 0:
                audio = audio / np.max(np.abs(audio))

            # extract features
            features = extract_features(audio, sr)

            if features is not None:
                X.append(features)
                y.append(cat)

        except Exception as e:
            print("Error:", f, e)

# ---------------------------------------------------
# Convert to numpy
# ---------------------------------------------------
X = np.array(X)
y = np.array(y)

print("\nTotal Samples:", len(X))
print("Feature Shape:", X.shape)

# shuffle
X, y = shuffle(X, y, random_state=42)

# ---------------------------------------------------
# Encode labels
# ---------------------------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder.classes_, "labels.pkl")

# one-hot encoding
y_cat = to_categorical(y_encoded)

# ---------------------------------------------------
# Scale features
# ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ---------------------------------------------------
# Train-test split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------------------------------
# ANN Model
# ---------------------------------------------------
model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(len(categories), activation='softmax'))

# ---------------------------------------------------
# Compile
# ---------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------
# Train
# ---------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\n🔥 Accuracy:", acc)

# ---------------------------------------------------
# Save model
# ---------------------------------------------------
model.save("ann_cough_model.h5")

print("✅ ANN Model Saved Successfully")
