import numpy as np
import librosa

def extract_features(y, sr=22050):

    try:
        # -----------------------------
        # Ensure fixed length input
        # -----------------------------
        target_len = 6 * sr

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # -----------------------------
        # Normalize (ONLY HERE)
        # -----------------------------
        if np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))

        # -----------------------------
        # MFCC features
        # -----------------------------
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Delta
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)

        # Delta2
        delta2 = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2, axis=1)

        # -----------------------------
        # Extra features
        # -----------------------------
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # -----------------------------
        # Final feature vector (43)
        # -----------------------------
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
        print("Feature error:", e)
        return None
