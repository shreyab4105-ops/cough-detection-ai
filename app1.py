import os
import numpy as np
import librosa
import joblib

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from pydub import AudioSegment

from utils import extract_features

# ---------------------------------------------------
# Base Directory
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ---------------------------------------------------
# FFmpeg Path
# ---------------------------------------------------
AudioSegment.converter = r"C:\Users\ACER\Downloads\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

# ---------------------------------------------------
# Load ANN Model
# ---------------------------------------------------
model = load_model(
    os.path.join(BASE_DIR, "ann_cough_model.h5")
)

# ---------------------------------------------------
# Load Labels
# ---------------------------------------------------
categories = joblib.load(
    os.path.join(BASE_DIR, "labels.pkl")
)

# ---------------------------------------------------
# Load Scaler
# ---------------------------------------------------
scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)

print("✅ ANN Model Loaded Successfully")
print("Categories:", categories)

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    # Check file
    if 'audio' not in request.files:
        return jsonify({
            'error': 'No file uploaded'
        }), 400

    file = request.files['audio']

    # Empty filename
    if file.filename == '':
        return jsonify({
            'error': 'No selected file'
        }), 400

    # Temporary file path
    temp_path = os.path.join(BASE_DIR, "temp.wav")

    # Save uploaded file
    file.save(temp_path)

    try:

        # ---------------------------------------------------
        # Load Audio
        # ---------------------------------------------------
        try:

            y, sr = librosa.load(
                temp_path,
                sr=22050,
                mono=True
            )

        except Exception:

            audio = AudioSegment.from_file(temp_path)

            audio = audio.set_frame_rate(22050)

            audio = audio.set_channels(1)

            y = np.array(
                audio.get_array_of_samples(),
                dtype=np.float32
            )

            y = y / np.max(np.abs(y))

            sr = 22050

        # ---------------------------------------------------
        # Pad or Trim Audio to 6 Seconds
        # ---------------------------------------------------
        target_len = 6 * sr

        if len(y) < target_len:

            y = np.pad(
                y,
                (0, target_len - len(y))
            )

        else:

            y = y[:target_len]

        # ---------------------------------------------------
        # Extract Features
        # ---------------------------------------------------
        features = extract_features(y, sr)

        if features is None:

            return jsonify({
                'error': 'Feature extraction failed'
            }), 500

        # ---------------------------------------------------
        # Reshape Features
        # ---------------------------------------------------
        features = np.array(features).reshape(1, -1)

        # ---------------------------------------------------
        # Scale Features
        # ---------------------------------------------------
        features = scaler.transform(features)

        # ---------------------------------------------------
        # ANN Prediction
        # ---------------------------------------------------
        prediction = model.predict(features)

        pred_index = np.argmax(prediction)

        label = categories[pred_index]

        # ---------------------------------------------------
        # Probability Scores
        # ---------------------------------------------------
        prob_dict = {

            categories[i]: round(
                float(prediction[0][i]) * 100,
                2
            )

            for i in range(len(categories))
        }

        # ---------------------------------------------------
        # Return Result
        # ---------------------------------------------------
        return jsonify({

            'prediction': label,

            'probabilities': prob_dict

        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500

    finally:

        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------------------------------------------
# Run Flask App
# ---------------------------------------------------
if __name__ == "__main__":

    app.run(
        debug=True
    )
