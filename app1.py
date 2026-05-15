import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from utils1 import extract_features

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "ann_cough_model.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
labels = joblib.load(os.path.join(BASE_DIR, "labels.pkl"))

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["audio"]
    path = "temp.wav"
    file.save(path)

    try:
        y, sr = librosa.load(path, sr=22050)

        features = extract_features(y, sr)

        features = scaler.transform([features])

        pred = model.predict(features)

        label = labels[np.argmax(pred)]

        return jsonify({"result": label})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    app.run(debug=True)
