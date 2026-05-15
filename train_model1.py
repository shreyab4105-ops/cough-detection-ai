import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from utils1 import extract_features

DATA_DIR = "RESIZED"

categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']

X = []
y = []

for cat in categories:

    folder = os.path.join(DATA_DIR, cat)

    for file in os.listdir(folder):

        if file.endswith(".wav") or file.endswith(".WAV"):

            path = os.path.join(folder, file)

            audio, sr = librosa.load(path, sr=22050)

            features = extract_features(audio, sr)

            if features is not None:
                X.append(features)
                y.append(cat)

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder.classes_, "labels.pkl")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(len(categories), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

model.save("ann_cough_model.h5")
