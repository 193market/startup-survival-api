import os
import json
import numpy as np
import joblib
from config import MODEL_DIR

# 서버 시작 시 1회 로드
_model = joblib.load(os.path.join(MODEL_DIR, 'survival_model.pkl'))
_slug_encoder = joblib.load(os.path.join(MODEL_DIR, 'slug_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_importance.json'), 'r', encoding='utf-8') as f:
    _importance_info = json.load(f)

FEATURE_NAMES = _importance_info['feature_names']
FEATURE_NAMES_KR = _importance_info['feature_names_kr']


def predict_survival(input_data: dict) -> dict:
    features = _extract_features(input_data)
    prob = _model.predict_proba([features])[0]
    survival_prob = round(float(prob[1]), 3)

    top_factor = _importance_info['top_risk_factor']

    return {
        'survival_prob': survival_prob,
        'survival_class': '생존' if survival_prob >= 0.5 else '위험',
        'confidence': round(float(max(prob)), 3),
        'top_risk_factor': top_factor,
        'top_risk_factor_kr': FEATURE_NAMES_KR.get(top_factor, top_factor),
        'model_accuracy': _importance_info['test_accuracy'],
        'feature_importance': {
            FEATURE_NAMES_KR.get(k, k): round(v, 4)
            for k, v in list(_importance_info['feature_importance'].items())[:5]
        },
    }


def _extract_features(data: dict) -> list:
    """입력 데이터에서 12개 feature 벡터 추출. 없는 값은 0으로 대체."""
    slug = data.get('slug', '')
    try:
        slug_code = int(_slug_encoder.transform([slug])[0])
    except (ValueError, KeyError):
        slug_code = 0

    def _v(val, default=0):
        return default if val is None else float(val)

    return [
        _v(data.get('national_survival_rate', data.get('survival_rate')), 50),
        _v(data.get('national_n5'), 0),
        _v(data.get('sido_survival_rate', data.get('survival_rate')), 50),
        _v(data.get('sido_n1'), 0),
        _v(data.get('sido_total', data.get('total')), 0),
        _v(data.get('comp_count', data.get('competitors')), 0),
        _v(data.get('population'), 0),
        _v(data.get('density', data.get('saturation')), 0),
        _v(data.get('avg_income'), 0),
        _v(data.get('young_ratio'), 0),
        _v(data.get('old_ratio'), 0),
        float(slug_code),
    ]
