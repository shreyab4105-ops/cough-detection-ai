
import numpy as np
import librosa

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

        return features

    except Exception as e:
        print("Feature error:", e)
        return None
