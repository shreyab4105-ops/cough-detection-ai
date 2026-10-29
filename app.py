import os
import tempfile
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils import extract_features
from pydub import AudioSegment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "cough_model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit

# Optional: set FFMPEG_EXE environment variable if FFmpeg is not on PATH.
ffmpeg_exe = os.environ.get("FFMPEG_EXE")
if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
    AudioSegment.converter = ffmpeg_exe

model = joblib.load(MODEL_PATH)
categories = joblib.load(LABELS_PATH) if os.path.isfile(LABELS_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.isfile(SCALER_PATH) else None

print("Cough model loaded:", type(model).__name__)
print("Model classes:", getattr(model, "classes_", "not exposed"))
print("Feature scaler loaded:", scaler is not None)
if categories is None:
    print("WARNING: labels.pkl not found. Add it beside app.py for readable class names.")

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"}

def class_label(class_value):
    """Map a model class to its human-readable label without assuming classes are 0..N-1."""
    if categories is None:
        return str(class_value)
    try:
        # Labels are commonly a list/array indexed by integer model class.
        idx = int(class_value)
        if 0 <= idx < len(categories):
            return str(categories[idx])
    except (ValueError, TypeError, IndexError):
        pass
    if isinstance(categories, dict):
        return str(categories.get(class_value, categories.get(str(class_value), class_value)))
    return str(class_value)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "labels_loaded": categories is not None,
        "scaler_loaded": scaler is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    patient_name = (request.form.get("patient_name") or "Not provided").strip()[:100]
    patient_age = (request.form.get("patient_age") or "").strip()[:3]
    patient_id = (request.form.get("patient_id") or "").strip()[:50]
    audio_file = request.files.get("audio")

    if not audio_file or not audio_file.filename:
        return jsonify({"error": "Please select or record a cough audio file."}), 400

    safe_name = secure_filename(audio_file.filename)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported audio format. Use WAV, MP3, M4A, OGG, FLAC, AAC, or WEBM."}), 400

    temp_path = None
    try:
        suffix = "." + ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=BASE_DIR) as temp:
            temp_path = temp.name
            audio_file.save(temp_path)

        # Decode audio, resample to the rate expected by the model, and use mono audio.
        try:
            y, sr = librosa.load(temp_path, sr=22050, mono=True)
        except Exception:
            audio = AudioSegment.from_file(temp_path).set_frame_rate(22050).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            scale = float(1 << (8 * audio.sample_width - 1))
            y = samples / scale
            sr = 22050

        if y is None or len(y) == 0 or not np.all(np.isfinite(y)):
            return jsonify({"error": "The audio could not be read. Try a clear WAV recording."}), 400

        # Keep the same 6-second input length as the original application.
        target_len = 6 * sr
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        features = extract_features(y, sr)
        if features is None:
            return jsonify({"error": "Feature extraction failed. Check that utils.py matches the trained model."}), 500

        features = np.asarray(features, dtype=np.float64).reshape(1, -1)
        # Apply the saved training scaler when provided with the project.
        model_features = scaler.transform(features) if scaler is not None else features
        pred = model.predict(model_features)[0]
        label = class_label(pred)

        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(model_features)[0]
            model_classes = getattr(model, "classes_", range(len(probs)))
            probabilities = {
                class_label(cls): round(float(prob), 6)
                for cls, prob in zip(model_classes, probs)
            }
            probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

        return jsonify({
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_id": patient_id,
            "label": label,
            "probabilities": probabilities,
            "filename": safe_name,
            "sample_rate": sr,
            "duration_seconds_used": min(6, round(len(y) / sr, 2)),
            "report_note": "This is an AI audio-classification result, not a medical diagnosis."
        })
    except Exception as exc:
        app.logger.exception("Cough prediction failed")
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "File is too large. Maximum upload size is 25 MB."}), 413

if __name__ == "__main__":
    # Debug mode is off by default. Use debug=True only during local development.
    app.run(host="127.0.0.1", port=5000, debug=False)
