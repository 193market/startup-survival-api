"""
창업 생존율 ML 모델 학습 스크립트
- 입력: L1, L2, competition_summary, kosis_population JSON
- 출력: survival_model.pkl, feature_importance.json
"""
import json
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Big-data', '001', '060_startup-survival', 'src', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_json(name):
    with open(os.path.join(DATA_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def build_training_data():
    l1 = load_json('survival_L1_summary.json')
    l2 = load_json('survival_L2_all.json')
    comp = load_json('competition_summary.json')
    kosis = load_json('kosis_population.json')

    # slug -> L1 전국 데이터
    l1_map = {item['slug']: item for item in l1['업종별']}

    # 업종 라벨 인코더
    all_slugs = sorted(l1_map.keys())
    slug_encoder = LabelEncoder()
    slug_encoder.fit(all_slugs)

    rows = []
    for slug, l2_data in l2.items():
        l1_item = l1_map.get(slug)
        if not l1_item:
            continue

        for sido_item in l2_data['시도별']:
            sido = sido_item['sido']
            pop_info = kosis.get(sido)
            if not pop_info:
                continue

            # 경쟁 업체 수 (시도 전체 합산)
            comp_count = 0
            if sido in comp:
                for sg_data in comp[sido].values():
                    if slug in sg_data:
                        comp_count += sg_data[slug]['count']

            # 목표: 5년 생존율 >= 50% -> 1(생존), < 50% -> 0(위험)
            n5 = sido_item.get('n5')
            if n5 is None:
                # n5 없으면 survival_rate 사용
                n5_proxy = sido_item['survival_rate']
            else:
                n5_proxy = n5

            label = 1 if n5_proxy >= 50 else 0

            age_dist = pop_info.get('age_distribution', {})
            young_ratio = age_dist.get('20대', 0) + age_dist.get('30대', 0)
            old_ratio = age_dist.get('50대', 0) + age_dist.get('60대이상', 0)

            row = {
                'national_survival_rate': l1_item['survival_rate'],
                'national_n5': l1_item.get('n5') or 0,
                'sido_survival_rate': sido_item['survival_rate'],
                'sido_n1': sido_item.get('n1') or 0,
                'sido_total': sido_item['total'],
                'comp_count': comp_count,
                'population': pop_info.get('population', 0),
                'density': pop_info.get('density', 0),
                'avg_income': pop_info.get('avg_income', 0),
                'young_ratio': young_ratio,
                'old_ratio': old_ratio,
                'slug_code': slug_encoder.transform([slug])[0],
                'label': label,
            }
            rows.append(row)

    feature_names = [
        'national_survival_rate', 'national_n5', 'sido_survival_rate',
        'sido_n1', 'sido_total', 'comp_count', 'population', 'density',
        'avg_income', 'young_ratio', 'old_ratio', 'slug_code',
    ]
    feature_names_kr = {
        'national_survival_rate': '전국 생존율',
        'national_n5': '전국 5년 생존율',
        'sido_survival_rate': '지역 생존율',
        'sido_n1': '지역 1년 생존율',
        'sido_total': '지역 등록 업체수',
        'comp_count': '경쟁업체 수',
        'population': '인구 수',
        'density': '인구밀도',
        'avg_income': '가구소득',
        'young_ratio': '20~30대 비율',
        'old_ratio': '50대 이상 비율',
        'slug_code': '업종 코드',
    }

    X = np.array([[r[f] for f in feature_names] for r in rows])
    y = np.array([r['label'] for r in rows])

    print(f"총 {len(rows)}행, feature {len(feature_names)}개")
    print(f"label 분포: 생존={np.sum(y==1)}, 위험={np.sum(y==0)}")

    return X, y, feature_names, feature_names_kr, slug_encoder


def train_models(X_train, X_test, y_train, y_test):
    models = {
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=7),
        'decision_tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'model': model,
        }
        print(f"  {name}: train={train_acc:.4f}, test={test_acc:.4f}")

    return results


def select_and_save_best(results, feature_names, feature_names_kr):
    best_name = max(results, key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_name]['model']

    # 변수 중요도
    if hasattr(best_model, 'feature_importances_'):
        importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, 'coef_'):
        importance = dict(zip(feature_names, np.abs(best_model.coef_[0]).tolist()))
    else:
        importance = {f: 0.0 for f in feature_names}

    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'survival_model.pkl'))

    meta = {
        'best_model': best_name,
        'train_accuracy': results[best_name]['train_accuracy'],
        'test_accuracy': results[best_name]['test_accuracy'],
        'feature_importance': importance_sorted,
        'top_risk_factor': list(importance_sorted.keys())[0],
        'feature_names': feature_names,
        'feature_names_kr': feature_names_kr,
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()},
    }
    with open(os.path.join(MODEL_DIR, 'feature_importance.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nBest: {best_name} (test={results[best_name]['test_accuracy']})")
    print(f"Top factor: {list(importance_sorted.keys())[0]} ({feature_names_kr.get(list(importance_sorted.keys())[0], '')})")
    return best_model


if __name__ == '__main__':
    X, y, feature_names, feature_names_kr, slug_encoder = build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\n모델 학습:")
    results = train_models(X_train, X_test, y_train, y_test)
    best = select_and_save_best(results, feature_names, feature_names_kr)

    # slug encoder 저장
    joblib.dump(slug_encoder, os.path.join(MODEL_DIR, 'slug_encoder.pkl'))
    print("\n모델 저장 완료: models/survival_model.pkl, feature_importance.json, slug_encoder.pkl")
