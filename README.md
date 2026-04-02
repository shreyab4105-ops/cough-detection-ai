# 🫁 RespireAI — Intelligent Cough Sound Classification System

An advanced machine learning system for analyzing cough sound patterns to identify potential respiratory conditions such as **Asthma, URTI, LRTI, Pneumonia, and Croup**. Built with a high-performance Python backend and a modern, interactive medical-tech web interface.

> ⚠️ **Disclaimer**: This project is intended for **research and educational purposes only**. It is **not a medical diagnostic tool**.

---

## 🚀 Key Features

### 📡 Real-Time Audio Analysis

* Upload `.wav` / `.mp3` files
* Record cough using microphone (browser-based)
* No `ffmpeg` required (custom WAV encoder)
* Fast inference (<2 seconds)

### 🧠 Machine Learning Pipeline

* Feature extraction using:

  * **MFCC (Mel-Frequency Cepstral Coefficients)**
  * **Log-Mel Spectrogram**
  * **Spectral Contrast**
  * **Continuous Wavelet Transform (CWT)**
* Classification using **Random Forest**

### 🎨 Modern Medical UI

* Glassmorphism dark theme
* Animated “Cough Pattern Analysis” scanner
* Real-time confidence visualization (Chart.js)

---

## 🫁 Classification Categories

The model predicts the following respiratory conditions:

* **Asthma**
* **URTI (Upper Respiratory Tract Infection)**
* **LRTI (Lower Respiratory Tract Infection)**
* **Pneumonia** *(treated as a severe LRTI subclass)*
* **Croup**
* **Normal**

---

## 📊 Model Performance

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | ~85–90% |
| Precision | ~84%    |
| Recall    | ~83%    |
| F1 Score  | ~83–85% |

> *Performance may vary depending on dataset quality and preprocessing.*

---

## 🧠 How It Works

1. **Audio Input**
   User uploads or records a cough sample

2. **Feature Extraction**
   The system extracts acoustic features:

   * MFCC → Captures human auditory perception
   * Spectral Contrast → Differentiates tonal vs noisy sounds
   * CWT → Detects transient cough patterns

3. **Prediction Engine**
   A trained Random Forest model classifies the cough into one of the defined categories

4. **Visualization**
   Results are displayed with confidence scores in real time

---

## 🔌 API Usage

### Endpoint

```bash
POST /predict
```

### Request

* `file`: audio file (`.wav` recommended)

### Example Response

```json
{
  "prediction": "Pneumonia",
  "confidence": {
    "Asthma": 0.10,
    "URTI": 0.20,
    "LRTI": 0.25,
    "Pneumonia": 0.35,
    "Croup": 0.05,
    "Normal": 0.05
  }
}
```

---

## 🛠️ Installation & Setup

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd respireai
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Model (Optional)

```bash
python train_model.py
```

### 5. Run Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```bash
├── app.py              # Flask backend
├── train_model.py      # Model training script
├── cough_model.pkl     # Trained model
├── templates/
│   └── index.html      # UI layout
├── static/
│   ├── style.css       # Glassmorphism UI
│   └── script.js       # Recording + encoding
├── DEMO.ipynb          # Research notebook
└── requirements.txt
```

---

## 🔐 Privacy & Security

* No audio data is stored
* All processing happens locally
* No external APIs are used

---

## 🔮 Future Improvements

* CNN-based deep learning model (spectrogram input)
* Mobile app integration
* Clinical dataset validation
* Real-time continuous cough monitoring

---

## 💡 Tech Stack

* **Backend**: Python, Flask, Scikit-learn, Librosa
* **Frontend**: HTML, CSS, JavaScript
* **Visualization**: Chart.js

---

## 📸 Demo

> Add screenshots or GIF here for best impact

---

## 👨‍💻 Author

Developed as an advanced machine learning prototype for respiratory sound analysis.

---
