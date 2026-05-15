import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from utils1 import extract_features

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model + scaler + labels
model = load_model(os.path.join(BASE_DIR, "ann_cough_model.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
labels = joblib.load(os.path.join(BASE_DIR, "labels.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["audio"]
    path = "temp.wav"
    file.save(path)

    try:
        y, sr = librosa.load(path, sr=22050)

        features = extract_features(y, sr)
        features = scaler.transform([features])

        pred = model.predict(features)[0]
        idx = np.argmax(pred)

        label = labels[idx]

        # probabilities for chart
        prob_dict = {
            labels[i]: float(pred[i])
            for i in range(len(labels))
        }

        return jsonify({
            "result": label,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    app.run(debug=True)
