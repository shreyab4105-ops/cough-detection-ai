import os
import librosa
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# Dataset Path
# ---------------------------------------------------
DATASET_PATH = "RESIZED"

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
# Feature Extraction Function
# ---------------------------------------------------
def extract_features(file_path):

    try:

        # Load audio
        y, sr = librosa.load(
            file_path,
            sr=22050,
            mono=True
        )

        # Pad / Trim to 6 seconds
        target_len = 6 * sr

        if len(y) < target_len:

            y = np.pad(
                y,
                (0, target_len - len(y))
            )

        else:

            y = y[:target_len]

        # MFCC Features
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        # Mean features
        mfccs_scaled = np.mean(
            mfccs.T,
            axis=0
        )

        return mfccs_scaled

    except Exception as e:

        print("Error:", file_path)

        return None

# ---------------------------------------------------
# Prepare Dataset
# ---------------------------------------------------
X = []
y = []

for category in categories:

    folder_path = os.path.join(
        DATASET_PATH,
        category
    )

    print(f"\nProcessing {category}...")

    for file in os.listdir(folder_path):

        if file.endswith(".wav") or file.endswith(".WAV"):

            file_path = os.path.join(
                folder_path,
                file
            )

            features = extract_features(file_path)

            if features is not None:

                X.append(features)

                y.append(category)

print("\n✅ Feature Extraction Completed")

# ---------------------------------------------------
# Convert to NumPy Arrays
# ---------------------------------------------------
X = np.array(X)

y = np.array(y)

print("Features Shape:", X.shape)

print("Labels Shape:", y.shape)

# ---------------------------------------------------
# Encode Labels
# ---------------------------------------------------
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# Save labels
joblib.dump(
    encoder.classes_,
    "labels.pkl"
)

print("\nEncoded Labels:")
print(encoder.classes_)

# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(
    scaler,
    "scaler.pkl"
)

# ---------------------------------------------------
# Save Features and Labels
# ---------------------------------------------------
np.save("features.npy", X_scaled)

np.save("labels.npy", y_encoded)

print("\n✅ Features Saved")

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y_encoded,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded
)

# ---------------------------------------------------
# Save Split Data
# ---------------------------------------------------
np.save("X_train.npy", X_train)

np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)

np.save("y_test.npy", y_test)

print("\n🎉 ANN Dataset Preparation Completed")
