from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class DiagnosisRequest(BaseModel):
    slug: str = ''
    business: str
    sido: str = '전국'
    sigungu: Optional[str] = None
    survival_rate: float
    grade: str
    national_survival_rate: Optional[float] = None
    national_n5: Optional[float] = None
    sido_survival_rate: Optional[float] = None
    sido_n1: Optional[float] = None
    sido_total: Optional[int] = None
    competitors: Optional[int] = None
    comp_count: Optional[int] = None
    density_rank: Optional[str] = None
    population: Optional[int] = None
    density: Optional[int] = None
    avg_income: Optional[int] = None
    young_ratio: Optional[float] = None
    old_ratio: Optional[float] = None
    total: Optional[int] = None


class DiagnosisResponse(BaseModel):
    diagnosis: str
    ml_prediction: dict
    rule_advice: list


@router.post('/api/v1/diagnosis', response_model=DiagnosisResponse)
async def get_diagnosis(req: DiagnosisRequest):
    from services.ml_service import predict_survival

    try:
        ml_input = req.dict()
        ml_result = predict_survival(ml_input)

        # 규칙 기반 조언
        advice = _rule_based_advice(req, ml_result)

        # Claude AI 진단 (API 키 있을 때만)
        diagnosis_text = ''
        try:
            from services.claude_service import generate_diagnosis
            analysis = {**req.dict(), 'ml_prediction': ml_result}
            diagnosis_text = await generate_diagnosis(analysis)
        except Exception:
            diagnosis_text = ' '.join(advice)

        return DiagnosisResponse(
            diagnosis=diagnosis_text,
            ml_prediction=ml_result,
            rule_advice=advice,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/api/v1/predict')
async def quick_predict(slug: str, sido: str = '', survival_rate: float = 50):
    """간단 예측 — GET 요청용"""
    from services.ml_service import predict_survival
    return predict_survival({
        'slug': slug,
        'survival_rate': survival_rate,
        'sido_survival_rate': survival_rate,
    })


def _rule_based_advice(req: DiagnosisRequest, ml: dict) -> list:
    tips = []
    grade = req.grade.upper()

    if grade in ('A', 'B'):
        tips.append(f'{req.sido}에서 {req.business}은(는) 비교적 안정적인 업종입니다.')
    elif grade == 'C':
        tips.append(f'{req.business}은(는) 평균 수준입니다. 차별화 전략이 필요합니다.')
    else:
        tips.append(f'{req.business}은(는) 위험도가 높습니다. 신중하게 검토하세요.')

    if req.competitors and req.competitors > 20:
        tips.append(f'동종 업체가 {req.competitors}개로 경쟁이 치열합니다.')

    prob = ml.get('survival_prob', 0)
    if prob < 0.4:
        tips.append(f'ML 예측 생존확률이 {prob*100:.0f}%로 낮습니다. 대안 업종을 검토하세요.')

    return tips
