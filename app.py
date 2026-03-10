import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

from models import load_expression_model, load_fatigue_model, EMOTIONS, FATIGUE_CLASSES
from services import (
    FaceDetector,
    compute_expression_depression_score,
    compute_fatigue_score,
    late_fusion,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

EXPR_WEIGHT = os.path.join(WEIGHTS_DIR, "best_model_ekspresi.pth")
FATIGUE_WEIGHT = os.path.join(WEIGHTS_DIR, "best_model_fatigue.pth")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

print(f"[INFO] Device: {DEVICE}")
print("[INFO] Loading expression model...")
expr_model = load_expression_model(EXPR_WEIGHT, DEVICE)
print("[OK] Expression model loaded")

print("[INFO] Loading fatigue model...")
fatigue_model = load_fatigue_model(FATIGUE_WEIGHT, DEVICE)
print("[OK] Fatigue model loaded")

print("[INFO] Initializing face detector...")
face_detector = FaceDetector(DEVICE)
print("[OK] Face detector ready")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict_expression(face_image):
    img_tensor = preprocess(face_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = expr_model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return {emo: round(float(p), 4) for emo, p in zip(EMOTIONS, probs)}


def predict_fatigue(face_image):
    img_tensor = preprocess(face_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = fatigue_model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return {cls: round(float(p), 4) for cls, p in zip(FATIGUE_CLASSES, probs)}


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "Field 'image' tidak ditemukan. Kirim gambar dengan key 'image'."
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "Nama file kosong."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Format file tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal membaca gambar: {str(e)}"
        }), 400

    face_image, bbox, face_conf = face_detector.detect_and_crop(image)

    if face_image is None:
        return jsonify({
            "success": False,
            "error": "Wajah tidak terdeteksi dalam gambar. Pastikan wajah terlihat jelas.",
            "face_detected": False,
        }), 200

    emotion_probs = predict_expression(face_image)
    fatigue_probs = predict_fatigue(face_image)

    expr_score = compute_expression_depression_score(emotion_probs)
    fat_score = compute_fatigue_score(fatigue_probs)
    result = late_fusion(expr_score, fat_score)

    return jsonify({
        "success": True,
        "face_detected": True,
        "face_confidence": round(face_conf, 4),
        "score": result["score"],
        "level": result["level"],
        "disclaimer": result["disclaimer"],
        "expression": {
            "score": result["expr_depression_score"],
            "probabilities": emotion_probs,
        },
        "fatigue": {
            "score": result["fatigue_score"],
            "probabilities": fatigue_probs,
        },
    }), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to Depression Indicator API",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": True,
    }), 200


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Depression Indicator API")
    print("  Endpoint: POST /predict")
    print("  Health:   GET  /health")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=8080, debug=False)