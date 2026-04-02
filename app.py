import os
import numpy as np
import librosa
import pywt
import joblib
from flask import Flask, request, jsonify, render_template

# --- Absolute template folder fix ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- Feature Extraction ---
def extract_features(y, sr=22050, n_mfcc=13):
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=n_mfcc)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        coeffs, _ = pywt.cwt(y, np.arange(1, 32), 'morl', sampling_period=1/sr)
        wavelet_mean = np.mean(np.abs(coeffs), axis=1)
        wavelet_std = np.std(np.abs(coeffs), axis=1)
        energy = np.sum(librosa.feature.rms(y=y))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(contrast, axis=1),
            wavelet_mean,
            wavelet_std,
            [energy, flatness]
        ])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- Model loading ---
model = None
SOURCE_DIR = os.path.join(BASE_DIR, 'Source')
categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
print("Detected categories:", categories)

def load_model():
    global model
    model_path = os.path.join(BASE_DIR, 'cough_model.pkl')
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except:
        print("Model file not found. Using mock predictions.")
        model = None

# --- Flask Routes ---
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
        # Load audio
        try:
            y, sr = librosa.load(temp_path, sr=22050, mono=True, res_type='kaiser_fast')
        except Exception as e:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_frame_rate(22050).set_channels(1)
            y = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
            sr = 22050

        # Pad/trim to 6 seconds
        target_len = 6 * sr
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Feature extraction
        features = extract_features(y, sr)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        # Predict
        if model:
            prediction = model.predict([features])[0]
            try:
                probs = model.predict_proba([features])[0]
                prob_dict = {cat: float(p) for cat, p in zip(categories, probs)}
            except:
                prob_dict = None
        else:
            prediction = "NORMAL (Mock)"
            prob_dict = {cat: 0.16 for cat in categories}
            prob_dict['NORMAL'] = 0.2

        print(f"DEBUG: Prediction done in {time.time() - start_time:.2f}s")
        return jsonify({'label': prediction, 'probabilities': prob_dict})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    load_model()
    # Always accessible from network
    app.run(host='0.0.0.0', port=5000, debug=True)
