from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class DiagnosisRequest(BaseModel):
    slug: str = ''
    business: str
    sido: str = '전국'
    sigungu: Optional[str] = None
    survival_rate: float = 50.0
    grade: str = 'C'
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
    # Phase 6-A 추가 필드
    saturation: Optional[float] = None
    rent: Optional[float] = None
    labor_cost: Optional[float] = None
    net_income: Optional[float] = None
    subsidies_count: Optional[int] = None
    subsidies_total: Optional[float] = None
    # 사용자 입력 (맞춤 진단 폼)
    user_age: Optional[int] = None
    user_capital: Optional[float] = None
    user_experience: Optional[str] = None
    user_note: Optional[str] = None


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
            diagnosis_text = '\n\n'.join(advice)

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


def _fmt(val, unit='', fallback='데이터 없음'):
    if val is None:
        return fallback
    if isinstance(val, float):
        if val == int(val):
            return f'{int(val)}{unit}'
        return f'{val:,.1f}{unit}'
    return f'{val:,}{unit}'


def _rule_based_advice(req: DiagnosisRequest, ml: dict) -> list:
    tips = []
    grade = (req.grade or 'C').upper()
    loc = f'{req.sido} {req.sigungu}' if req.sigungu else req.sido
    biz = req.business

    # 1. 종합 진단 — 등급 + 생존율 숫자 명시
    if grade in ('A', 'B'):
        tips.append(
            f'{loc}에서 {biz}은(는) 5년 생존율 {req.survival_rate:.1f}%로 '
            f'비교적 안정적인 업종입니다. '
            f'차별화된 서비스로 초기 고객을 확보하는 것이 핵심입니다.'
        )
    elif grade == 'C':
        tips.append(
            f'{loc}에서 {biz}은(는) 5년 생존율 {req.survival_rate:.1f}%로 '
            f'평균 수준입니다. 뚜렷한 차별화 전략 없이는 경쟁에서 밀릴 수 있습니다.'
        )
    else:
        alive = max(1, round(req.survival_rate / 10))
        tips.append(
            f'{loc}에서 {biz}은(는) 5년 생존율 {req.survival_rate:.1f}%로 위험 수준입니다. '
            f'10곳이 개업하면 약 {10 - alive}곳이 5년 내 폐업합니다. '
            f'신중한 검토가 필요합니다.'
        )

    # 2. 핵심 위험 요인
    risks = []
    if req.saturation is not None and req.saturation > 100:
        risks.append(f'포화도 {req.saturation:.1f}(인구 1만명당 업체 수)로 시장이 과포화 상태')
    if req.net_income is not None and req.net_income < 0:
        risks.append(f'예상 월 순이익 {req.net_income:,.0f}만원으로 적자 구조')
    if req.competitors is not None and req.competitors > 30:
        risks.append(f'동종 업체 {req.competitors}개로 경쟁 치열')

    if risks:
        tips.append('⚠️ 핵심 위험 요인: ' + ', '.join(risks) + '.')

    # 3. 재무 요약
    finance_parts = []
    if req.rent is not None:
        finance_parts.append(f'임대료 월 {_fmt(req.rent, "만원")}')
    if req.labor_cost is not None:
        finance_parts.append(f'인건비 월 {_fmt(req.labor_cost, "만원")}')
    if req.net_income is not None:
        finance_parts.append(f'순이익 월 {_fmt(req.net_income, "만원")}')
    if finance_parts:
        tips.append('💰 예상 재무: ' + ', '.join(finance_parts) + '.')

    # 4. ML 예측
    prob = ml.get('survival_prob', 0)
    risk_kr = ml.get('top_risk_factor_kr', '')
    if prob < 0.4:
        tips.append(
            f'🤖 ML 예측 생존확률 {prob * 100:.0f}% (위험). '
            f'최대 위험 요인: {risk_kr}. 대안 업종·지역을 검토하세요.'
        )
    elif prob >= 0.6:
        tips.append(f'🤖 ML 예측 생존확률 {prob * 100:.0f}% (양호). 최대 주의 요인: {risk_kr}.')
    else:
        tips.append(f'🤖 ML 예측 생존확률 {prob * 100:.0f}% (보통). 최대 주의 요인: {risk_kr}.')

    # 5. 지원금
    if req.subsidies_count and req.subsidies_count > 0:
        tips.append(
            f'📋 활용 가능 지원금 {req.subsidies_count}건'
            + (f' (최대 {_fmt(req.subsidies_total, "만원")})' if req.subsidies_total else '')
            + '. 보조금24 탭에서 상세 조건을 확인하세요.'
        )

    # 6. 사용자 맞춤 (입력된 경우)
    if req.user_capital is not None and req.rent is not None:
        monthly_burn = req.rent + (req.labor_cost or 0)
        if monthly_burn > 0:
            runway = int(req.user_capital / monthly_burn)
            if runway < 18:
                tips.append(
                    f'💡 자본금 {_fmt(req.user_capital, "만원")} 기준 월 고정비 '
                    f'{_fmt(monthly_burn, "만원")}으로 약 {runway}개월 운영 가능. '
                    f'최소 18개월치 운영자금 확보를 권장합니다.'
                )
            else:
                tips.append(
                    f'💡 자본금 {_fmt(req.user_capital, "만원")} 기준 약 {runway}개월 '
                    f'운영 가능으로 자금 여력은 양호합니다.'
                )

    return tips
