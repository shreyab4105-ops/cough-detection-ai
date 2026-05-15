import os
import glob
import random
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from utils import extract_features

# ---------------------------------------------------
# Dataset Path
# ---------------------------------------------------
DATA_DIR = "RESIZED"

# ---------------------------------------------------
# Categories
# ---------------------------------------------------
categories = [
    'Asthama',
    'CROUP',
    'LTRI',
    'NORMAL',
    'PNEUMONIA',
    'URTI'
]

# ---------------------------------------------------
# Feature Lists
# ---------------------------------------------------
X = []

y = []

print("Training Categories:", categories)

# ---------------------------------------------------
# Read Dataset
# ---------------------------------------------------
for cat in categories:

    folder = os.path.join(DATA_DIR, cat)

    files = glob.glob(
        os.path.join(folder, "*.wav")
    ) + glob.glob(
        os.path.join(folder, "*.WAV")
    )

    print(cat, "files:", len(files))

    random.shuffle(files)

    for f in files:

        try:

            # Load Audio
            y_audio, sr = librosa.load(
                f,
                sr=22050
            )

            # Pad / Trim to 6 sec
            target_len = 6 * sr

            if len(y_audio) < target_len:

                y_audio = np.pad(
                    y_audio,
                    (0, target_len - len(y_audio))
                )

            else:

                y_audio = y_audio[:target_len]

            # Extract Features
            features = extract_features(
                y_audio,
                sr
            )

            if features is not None:

                X.append(features)

                y.append(cat)

        except Exception as e:

            print("Error:", f)

            print(e)

# ---------------------------------------------------
# Convert to NumPy Arrays
# ---------------------------------------------------
X = np.array(X)

y = np.array(y)

print("\nTotal Samples:", len(X))

print("Feature Shape:", X.shape)

# ---------------------------------------------------
# Encode Labels
# ---------------------------------------------------
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# Save Labels
joblib.dump(
    encoder.classes_,
    "labels.pkl"
)

# ---------------------------------------------------
# One Hot Encoding
# ---------------------------------------------------
y_categorical = to_categorical(y_encoded)

# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save Scaler
joblib.dump(
    scaler,
    "scaler.pkl"
)

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y_categorical,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded
)

# ---------------------------------------------------
# Build ANN Model
# ---------------------------------------------------
model = Sequential()

# Input Layer
model.add(
    Dense(
        256,
        activation='relu',
        input_shape=(X.shape[1],)
    )
)

model.add(
    Dropout(0.3)
)

# Hidden Layer
model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.3)
)

# Hidden Layer
model.add(
    Dense(
        64,
        activation='relu'
    )
)

# Output Layer
model.add(
    Dense(
        len(categories),
        activation='softmax'
    )
)

# ---------------------------------------------------
# Compile Model
# ---------------------------------------------------
model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# ---------------------------------------------------
# Train ANN
# ---------------------------------------------------
history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=16,

    validation_data=(X_test, y_test)
)

# ---------------------------------------------------
# Evaluate Model
# ---------------------------------------------------
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\n🔥 ANN Accuracy:", accuracy)

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------
model.save(
    "ann_cough_model.h5"
)

print("\n✅ ANN Model Saved Successfully")
