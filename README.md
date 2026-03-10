# Depression Indicator API

API untuk mendeteksi **indikator depresi** berdasarkan **ekspresi wajah** dan **tingkat kelelahan (fatigue)** menggunakan pendekatan **Deep Learning** dan **Late Fusion**.

Sistem ini memanfaatkan analisis wajah dari gambar untuk menghasilkan **skor indikasi depresi** dalam rentang **0–100**.

⚠️ **Catatan:** Sistem ini **bukan alat diagnosis medis** dan hanya digunakan untuk **indikator awal berbasis analisis visual**.

---

# Overview

Penelitian menunjukkan bahwa **ekspresi wajah** dan **tanda kelelahan** sering muncul pada individu dengan gangguan depresi. Sistem ini menggabungkan dua model deep learning:

* **Emotion Recognition Model**
* **Fatigue Detection Model**

Hasil dari kedua model kemudian digabungkan menggunakan metode **Late Fusion (Weighted Average)** untuk menghasilkan skor indikasi depresi.

---

# System Architecture

Pipeline sistem:

```
Input Image
     ↓
Face Detection (MTCNN)
     ↓
Face Cropping
     ↓
 ┌─────────────────────┐
 │ Expression Model     │
 │ ResNet-34 (FER2013)  │
 └─────────────────────┘
          ↓
 ┌─────────────────────┐
 │ Fatigue Model        │
 │ ResNet-18            │
 └─────────────────────┘
          ↓
        Late Fusion
          ↓
    Depression Score
          ↓
        API Output
```

---

# Model Architecture

## Expression Model

* Backbone: **ResNet-34**
* Dataset: **FER2013**
* Classes: 7 emosi

| Emotion  |
| -------- |
| Angry    |
| Disgust  |
| Fear     |
| Happy    |
| Sad      |
| Surprise |
| Neutral  |

---

## Fatigue Model

* Backbone: **ResNet-18**
* Classes:

| Class      |
| ---------- |
| Fatigue    |
| NonFatigue |

---

# Late Fusion Strategy

Sistem menggunakan **Weighted Average Late Fusion** berdasarkan relevansi gejala terhadap **DSM-5 Depression Criteria**.

### Fusion Formula

```
Depression Score =
(0.6 × Expression Score) +
(0.4 × Fatigue Score)
```

---

# Emotion → Depression Mapping

| Emotion  | Weight | Interpretation             |
| -------- | ------ | -------------------------- |
| Sad      | +1.0   | Core symptom of depression |
| Neutral  | +0.5   | Blunted / flat affect      |
| Angry    | +0.3   | Irritability               |
| Fear     | +0.3   | Anxiety comorbidity        |
| Disgust  | +0.1   | Low relevance              |
| Happy    | -0.8   | Counter-indicator          |
| Surprise | -0.2   | Weak counter-indicator     |

---

# Depression Score Interpretation

| Score Range | Level    |
| ----------- | -------- |
| 0 – 33      | Low      |
| 34 – 66     | Moderate |
| 67 – 100    | High     |

---

# Project Structure

```
depression-api/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
├── models/
│   ├── __init__.py
│   ├── expression.py
│   └── fatigue.py
│
├── services/
│   ├── __init__.py
│   ├── face_detector.py
│   └── fusion.py
│
└── weights/
    ├── best_model_ekspresi.pth
    └── best_model_fatigue.pth
```

---

# Requirements

* Python **3.10+**
* PyTorch
* Docker **(optional)**
* RAM minimal **4 GB**

---

# Installation (Local Python)

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd depression-api
```

---

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download Model Weights

Download model berikut dan letakkan pada folder `weights/`:

```
weights/
├── best_model_ekspresi.pth
└── best_model_fatigue.pth
```

---

### 5️⃣ Run API

```bash
python app.py
```

Server akan berjalan di:

```
http://localhost:8080
```

---

# Running with Docker (Recommended)

Build dan jalankan container:

```bash
docker compose up --build
```

API akan tersedia di:

```
http://localhost:8080
```

---

# API Endpoints

## Health Check

### GET `/health`

Digunakan untuk memastikan server berjalan dengan baik.

### Response

```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": true
}
```

---

# Prediction Endpoint

### POST `/predict`

Endpoint utama untuk memproses gambar wajah.

### Request

* Method: `POST`
* Content-Type: `multipart/form-data`
* Field: `image`

Supported formats:

```
jpg
jpeg
png
bmp
webp
```

Max size:

```
10 MB
```

---

### Example Request

```bash
curl -X POST http://localhost:8080/predict \
-F "image=@face.jpg"
```

---

# Example Response

```json
{
  "success": true,
  "face_detected": true,
  "face_confidence": 0.9987,
  "score": 34.91,
  "level": "Moderate",
  "expression": {
    "score": 0.1619,
    "probabilities": {
      "Angry": 0.1299,
      "Disgust": 0.0901,
      "Fear": 0.0477,
      "Happy": 0.2068,
      "Sad": 0.0653,
      "Surprise": 0.0433,
      "Neutral": 0.4168
    }
  },
  "fatigue": {
    "score": 0.6298,
    "probabilities": {
      "Fatigue": 0.6298,
      "NonFatigue": 0.3702
    }
  }
}
```

---

# Error Response

Jika wajah tidak terdeteksi:

```json
{
  "success": false,
  "face_detected": false,
  "error": "Wajah tidak terdeteksi dalam gambar."
}
```

---

# Disclaimer

Sistem ini **tidak dimaksudkan sebagai alat diagnosis medis**.

Output yang dihasilkan hanya merupakan **indikator awal berbasis analisis visual** menggunakan model machine learning.

Untuk diagnosis klinis, silakan berkonsultasi dengan **tenaga kesehatan profesional**.

---

# Future Improvements

Beberapa pengembangan yang dapat dilakukan:

* Video-based depression detection
* Temporal facial analysis
* Multimodal fusion (audio + text)
* Clinical dataset integration

---

# Author

Developed for research purposes in **Depression Indicator Detection using Deep Learning and Facial Analysis**.

---

# License

For research and educational purposes.
