import os
import numpy as np
import librosa
import pywt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Feature Extraction (Same as app.py) ---
def extract_features(y, sr=22050, n_mfcc=13):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=n_mfcc)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    coeffs, _ = pywt.cwt(y, np.arange(1, 32), 'morl', sampling_period=1/sr)
    wavelet_mean = np.mean(np.abs(coeffs), axis=1)
    wavelet_std = np.std(np.abs(coeffs), axis=1)
    energy = np.sum(librosa.feature.rms(y=y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(contrast, axis=1),
        wavelet_mean,
        wavelet_std,
        [energy, flatness]
    ])

def train_and_save(data_dir='RESIZED', model_name='cough_model.pkl'):
    X, y = [], []
    categories = ['Asthama', 'CROUP', 'LTRI', 'NORMAL', 'PNEUMONIA', 'URTI']
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Please provide a path to processed audio.")
        return

    print("Loading data and extracting features...")
    for label in categories:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print(f"Skipping category {label} - folder not found.")
            continue
            
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                file_path = os.path.join(folder, file)
                try:
                    signal, sr = librosa.load(file_path, sr=22050, mono=True)
                    # Use central 6 seconds or pad
                    target_len = 6 * 22050
                    if len(signal) < target_len:
                        signal = np.pad(signal, (0, target_len - len(signal)))
                    else:
                        signal = signal[:target_len]
                        
                    features = extract_features(signal, sr)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if len(X) == 0:
        print("No audio data found to train.")
        return

    print(f"Found {len(X)} samples. Training RandomForest model...")
    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_name)
    print(f"Model saved to {model_name}.")

if __name__ == "__main__":
    # If RESIZED doesn't exist, we might want to check RENAMED or ORIGINAL.
    # The user should ensure data is in a supported structure before running this.
    train_and_save()
