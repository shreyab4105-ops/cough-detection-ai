# 🫁 Cough Detection AI - Smart Diagnostic Prototype

An advanced machine learning prototype for detecting respiratory disorders (Asthma, Croup, Pneumonia, etc.) from cough sound signatures. This project combines a high-performance Python backend with a premium, glassmorphic medical-tech web interface.

## 🚀 Quick Setup (The "Get It Running" Guide)

Follow these steps to set up the project on a fresh machine (Ubuntu/Linux):

### 1. Environment Initialization
Create and activate a Python virtual environment to keep your system clean:
```bash
# Create the environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries (Librosa, Scikit-Learn, Flask, etc.):
```bash
pip install -r requirements.txt
```

### 3. Prepare the Model (Optional)
If you have your own dataset, you can train the model:
1.  Ensure your processed audio is in a directory named `RESIZED`.
2.  Run the training script:
    ```bash
    python train_model.py
    ```
    This creates `cough_model.pkl`. If this file isn't present, the app will run in **Mock Mode** for demonstration.

### 4. Launch the Web Application
Start the Flask server:
```bash
python app.py
```
Visit **`http://127.0.0.1:5000`** in your browser to begin analysis.

---

## � Project Features

### 📡 Real-time Acoustic Analysis
*   **Dual Input**: Support for local file uploads (`.wav`, `.mp3`) and live microphone recording.
*   **Browser-Based WAV Encoding**: Our custom JavaScript encoder converts raw mic data into standard PCM WAV—eliminating the need for system-level `ffmpeg` or `ffprobe`.

### 🧠 Modern AI Architecture
*   **Complex Feature Extraction**: Combines MFCCs, Log-Mel Spectrograms, Spectral Contrast, and **Continuous Wavelet Transforms (CWT)** for high-fidelity signal profiling.
*   **Optimized Pipeline**: Backend processing is optimized for responsiveness, delivering results in <2 seconds.

### 🎨 Premium UI/UX
*   **Glassmorphic Design**: Modern dark mode with translucent "frost" effects and radial neon gradients.
*   **Scanning Simulation**: Live "Cough Pattern Analysis" overlay with animated scanning bars for a professional medical-tech feel.
*   **Interactive Visualizations**: Real-time confidence tracking using high-contrast neon bar charts (Chart.js).

---

## 📂 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `app.py` | Flask backend with feature extraction and prediction endpoints. |
| `train_model.py` | Script to train the RandomForest classifier from processed audio. |
| `DEMO.ipynb` | Research notebook showing data exploration and feature testing. |
| `static/` | UI assets: `style.css` (Glassmorphism) and `script.js` (Recording/Encoding). |
| `templates/` | `index.html` structure with medical dashboard layout. |
| `requirements.txt` | Core ML and Web dependencies. |

## 🛠️ Requirements
*   **Python 3.10+**
*   **Modern Browser** (Chrome or Firefox recommended for high-quality audio capture)
*   **Microphone** (for live diagnostic simulation)

---
*Developed for advanced respiratory diagnostic research and educational demonstration.*