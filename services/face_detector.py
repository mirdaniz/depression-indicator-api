from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:
    def __init__(self, device):
        self.mtcnn = MTCNN(
            image_size=224,
            margin=40,
            keep_all=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=device
        )

    def detect_and_crop(self, image: Image.Image):
        boxes, probs = self.mtcnn.detect(image)

        if boxes is not None and len(boxes) > 0:
            best_idx = probs.argmax()
            box = boxes[best_idx]
            confidence = float(probs[best_idx])

            x1, y1, x2, y2 = [int(b) for b in box]
            w, h = image.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped = image.crop((x1, y1, x2, y2))
            return cropped, [x1, y1, x2, y2], confidence

        return None, None, 0.0