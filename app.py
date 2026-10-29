import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify, render_template
from utils import extract_features
from pydub import AudioSegment

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- FFmpeg path ---
AudioSegment.converter = r"C:\Users\ACER\Downloads\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

# --- Load Model & Labels ---
model = joblib.load(os.path.join(BASE_DIR, "cough_model.pkl"))
categories = joblib.load(os.path.join(BASE_DIR, "labels.pkl"))

print("✅ Model loaded")
print("Categories:", categories)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio']
    temp_path = os.path.join(BASE_DIR, "temp.wav")
    file.save(temp_path)

    try:
        # Load audio
        try:
            y, sr = librosa.load(temp_path, sr=22050, mono=True)
        except:
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_frame_rate(22050).set_channels(1)
            y = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
            sr = 22050

        # Pad/Trim (6 sec)
        target_len = 6 * sr
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Extract features
        features = extract_features(y, sr)

        if features is None:
            return jsonify({'error': 'Feature failed'}), 500

        # Predict
        pred = model.predict([features])[0]
        label = categories[pred]

        # Probabilities
        probs = model.predict_proba([features])[0]
        prob_dict = {
            categories[i]: float(probs[i])
            for i in range(len(categories))
        }

        return jsonify({
            'label': label,
            'probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
