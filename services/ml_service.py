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
    """입력 데이터에서 feature 벡터 추출. feature_importance.json의 feature_names 순서에 맞춤."""
    slug = data.get('slug', '')
    try:
        slug_code = int(_slug_encoder.transform([slug])[0])
    except (ValueError, KeyError):
        slug_code = 0

    def _v(val, default=0):
        return default if val is None else float(val)

    # FEATURE_NAMES 순서에 맞춰 동적 추출
    feature_map = {
        'national_survival_rate': _v(data.get('national_survival_rate', data.get('survival_rate')), 50),
        'national_n5': _v(data.get('national_n5'), 0),
        'sido_survival_rate': _v(data.get('sido_survival_rate', data.get('survival_rate')), 50),
        'sido_n1': _v(data.get('sido_n1'), 0),
        'sido_total': _v(data.get('sido_total', data.get('total')), 0),
        'comp_count': _v(data.get('comp_count', data.get('competitors')), 0),
        'population': _v(data.get('population'), 0),
        'density': _v(data.get('density'), 0),
        'avg_income': _v(data.get('avg_income'), 0),
        'young_ratio': _v(data.get('young_ratio'), 0),
        'old_ratio': _v(data.get('old_ratio'), 0),
        'saturation': _v(data.get('saturation'), 0),
        'rent_level': _v(data.get('rent_level', data.get('rent')), 0),
        'building_density': _v(data.get('building_density'), 0),
        'slug_code': float(slug_code),
    }
    return [feature_map.get(f, 0) for f in FEATURE_NAMES]
