import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify, render_template

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- FFmpeg Fix ---
from pydub import AudioSegment
AudioSegment.converter = r"C:\Users\ACER\Downloads\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

# --- Categories (same as training) ---
categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']
print("Detected categories:", categories)

# --- Feature Extraction (42 FEATURES) ---
def extract_features(y, sr=22050):
    try:
        # Normalize
        if np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Delta
        delta = librosa.feature.delta(mfcc)

        # Delta2
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Extra features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        spectral = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Combine → 42 features
        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1),
            zcr,
            chroma,
            spectral
        ])

        print("Feature length:", len(features))  # 🔥 Debug check

        return features

    except Exception as e:
        print("Feature error:", e)
        return None

# --- Load Model ---
model = None

def load_model():
    global model
    try:
        model = joblib.load(os.path.join(BASE_DIR, 'cough_model.pkl'))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print("❌ Model load error:", e)
        model = None

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = os.path.join(BASE_DIR, 'temp_audio.wav')
    file.save(temp_path)

    import time
    start_time = time.time()

    try:
        # --- Load audio ---
        try:
            y, sr = librosa.load(temp_path, sr=22050, mono=True)
        except:
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_frame_rate(22050).set_channels(1)
            y = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
            sr = 22050

        # --- Pad / Trim (6 sec) ---
        target_len = 6 * sr
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # --- Extract features ---
        features = extract_features(y, sr)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        # --- Prediction ---
        if model:
            pred_index = int(model.predict([features])[0])
            prediction = categories[pred_index]

            try:
                probs = model.predict_proba([features])[0]
                prob_dict = {
                    str(cat): float(p)
                    for cat, p in zip(categories, probs)
                }
            except:
                prob_dict = None
        else:
            prediction = "NORMAL"
            prob_dict = {cat: 0.16 for cat in categories}
            prob_dict["NORMAL"] = 0.2

        print(f"Prediction done in {time.time() - start_time:.2f}s")

        return jsonify({
            'label': prediction,
            'probabilities': prob_dict
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Run ---
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
