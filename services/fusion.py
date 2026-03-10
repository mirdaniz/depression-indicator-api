EMOTION_DEPRESSION_WEIGHTS = {
    'Sad':      1.0,
    'Neutral':  0.5,
    'Angry':    0.3,
    'Fear':     0.3,
    'Disgust':  0.1,
    'Happy':   -0.8,
    'Surprise':-0.2,
}

W_EXPRESSION = 0.6
W_FATIGUE    = 0.4


def compute_expression_depression_score(emotion_probs: dict) -> float:

    score = sum(
        prob * EMOTION_DEPRESSION_WEIGHTS.get(emo, 0.0)
        for emo, prob in emotion_probs.items()
    )
    return max(0.0, min(1.0, score))


def compute_fatigue_score(fatigue_probs: dict) -> float:
    return fatigue_probs.get('Fatigue', 0.0)


def late_fusion(expr_score: float, fatigue_score: float) -> dict:

    raw = (W_EXPRESSION * expr_score) + (W_FATIGUE * fatigue_score)
    final = max(0.0, min(100.0, raw * 100))

    if final <= 33:
        level = "Indikator depresi tidak signifikan"
    elif final <= 66:
        level = "Terdapat beberapa indikator depresi"
    else:
        level = "Indikator depresi cukup signifikan"

    return {
        "score": round(final, 2),
        "level": level,
        "expr_depression_score": round(expr_score, 4),
        "fatigue_score": round(fatigue_score, 4),
        "disclaimer": "Hasil ini bukan diagnosis medis. Merupakan indikator awal berdasarkan analisis ekspresi wajah dan kelelahan. Konsultasikan dengan profesional kesehatan mental untuk evaluasi lebih lanjut."
    }