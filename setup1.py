import os
import librosa
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# Dataset Path
# ---------------------------------------------------
DATASET_PATH = "RESIZED"

categories = [
    'Asthama',
    'CROUP',
    'LTRI',
    'NORMAL',
    'PNEUMONIA',
    'URTI'
]

# ---------------------------------------------------
# Feature Extraction (ANN compatible)
# ---------------------------------------------------
def extract_features(file_path):

    try:

        y, sr = librosa.load(file_path, sr=22050, mono=True)

        target_len = 6 * sr

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # normalize
        if np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))

        # -----------------------------
        # MFCC (13)
        # -----------------------------
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Delta
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)

        # Delta2
        delta2 = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2, axis=1)

        # Extra features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        features = np.hstack([
            mfcc_mean,
            delta_mean,
            delta2_mean,
            zcr,
            chroma,
            spectral_centroid,
            spectral_rolloff
        ])

        return features

    except Exception as e:
        print("Error:", file_path, e)
        return None

# ---------------------------------------------------
# Build Dataset
# ---------------------------------------------------
X = []
y = []

for category in categories:

    folder = os.path.join(DATASET_PATH, category)

    print(f"\nProcessing {category}...")

    for file in os.listdir(folder):

        if file.endswith(".wav") or file.endswith(".WAV"):

            file_path = os.path.join(folder, file)

            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(category)

# ---------------------------------------------------
# Convert
# ---------------------------------------------------
X = np.array(X)
y = np.array(y)

print("\nTotal samples:", len(X))
print("Feature shape:", X.shape)

# ---------------------------------------------------
# Encode labels
# ---------------------------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder.classes_, "labels.pkl")

# ---------------------------------------------------
# Scale features
# ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ---------------------------------------------------
# Split dataset
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------------------------------
# Save data
# ---------------------------------------------------
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("features.npy", X_scaled)
np.save("labels.npy", y_encoded)

print("\n🎉 Dataset Ready for ANN Training")
