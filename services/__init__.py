from .face_detector import FaceDetector
from .fusion import (
    compute_expression_depression_score,
    compute_fatigue_score,
    late_fusion,
    EMOTION_DEPRESSION_WEIGHTS,
    W_EXPRESSION,
    W_FATIGUE,
)