import anthropic
from config import CLAUDE_API_KEY

_client = None


def _get_client():
    global _client
    if _client is None:
        if not CLAUDE_API_KEY:
            raise RuntimeError('CLAUDE_API_KEY 환경변수가 설정되지 않았습니다.')
        _client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _client


def _fmt(val, unit=''):
    if val is None:
        return '데이터 없음'
    if isinstance(val, float):
        if val == int(val):
            return f'{int(val)}{unit}'
        return f'{val:,.1f}{unit}'
    return f'{val:,}{unit}'


async def generate_diagnosis(analysis_data: dict) -> str:
    ml = analysis_data.get('ml_prediction', {})

    prompt = f"""당신은 10년 경력의 창업 전문 컨설턴트입니다.
아래 공공데이터 분석 결과를 바탕으로, 이 사람이 창업해도 되는지 진단해주세요.
등급에 따라 톤을 다르게 하세요:
- A/B등급: 긍정적 톤. "조건이 양호합니다" "도전해볼 만합니다" 등. 주의사항도 함께 언급.
- C등급: 중립적 톤. 가능성과 위험을 균형있게 제시.
- D등급 이하: 부정적 톤. "추천하지 않습니다" 명확히. 반드시 대안 제시.

[분석 대상]
- 업종: {analysis_data.get('business', '미지정')}
- 지역: {analysis_data.get('sido', '전국')} {analysis_data.get('sigungu', '')}
- 사용자 연령: {_fmt(analysis_data.get('user_age'), '세')}
- 사용자 자본금: {_fmt(analysis_data.get('user_capital'), '만원')}
- 사용자 경험: {analysis_data.get('user_experience') or '미입력'}
- 사용자 메모: "{analysis_data.get('user_note') or '없음'}"

[공공데이터 기반 현황 — 9개 지표]
1. 생존율: {_fmt(analysis_data.get('survival_rate'), '%')} ({analysis_data.get('grade', '?')}등급)
   — 출처: 행정안전부 지방행정 인허가 데이터 1,125만 건 분석
2. 경쟁: 동종 업체 {_fmt(analysis_data.get('competitors'), '개')}, 포화도 {_fmt(analysis_data.get('saturation'))} (인구 1만명당 업체 수)
   — 출처: 소상공인시장진흥공단 상권정보 + KOSIS 인구통계
3. 임대료: 월 {_fmt(analysis_data.get('rent'), '만원')}
   — 출처: 한국부동산원 상업용부동산 임대동향조사
4. 인건비: 월 {_fmt(analysis_data.get('labor_cost'), '만원')} (업종 평균)
   — 출처: 고용노동부 사업체노동력조사 2025
5. 예상 순이익: 월 {_fmt(analysis_data.get('net_income'), '만원')}
   — 산출: 업종 평균 매출 − 임대료 − 인건비 − 경비율(국세청 기준경비율)
6. 지역 인구: {_fmt(analysis_data.get('population'), '명')}
7. 관련 지원금: {analysis_data.get('subsidies_count', 0)}건, 최대 {_fmt(analysis_data.get('subsidies_total'), '만원')}
   — 출처: 보조금24
8. ML 예측 생존확률: {_fmt(ml.get('survival_prob', None))}
9. 최대 위험 요인: {ml.get('top_risk_factor_kr', '데이터 없음')}
10. 인허가 절차: {analysis_data.get('license_info') or '데이터 없음'}

[요청사항 — 5개 항목을 반드시 모두 작성]

1. **종합 진단** (3~4문장)
   - 이 사람이 이해할 수 있는 일상 언어로 작성
   - 위 9개 지표 중 핵심 숫자를 인용하며 설명
   - 위험하면 솔직하게 "추천하지 않습니다"라고 말할 것

2. **핵심 위험 요인** (1~2개)
   - 위 데이터에서 가장 치명적인 지표를 구체적 숫자와 함께 명시
   - "왜" 위험한지 한 문장으로 설명

3. **구체적 대안** (2가지)
   - 대안 A: 같은 지역에서 더 안전한 업종 (생존율 차이 명시)
   - 대안 B: 같은 업종이지만 더 유리한 지역 또는 운영 방식
   - 각 대안에 구체적 이유 포함

4. **지원금 활용 팁** (1~2문장)
   - 위 지원금 건수가 있으면 우선 신청할 것 추천
   - 없으면 "소상공인시장진흥공단 정책자금을 먼저 알아보세요" 안내

5. **인허가 안내** (인허가 절차 데이터가 있으면 1~2문장)
   - 해당 업종에 필요한 인허가 절차를 간단히 안내
   - 데이터가 "데이터 없음"이면 이 항목은 생략

[규칙 — 반드시 준수]
- 위 데이터에 있는 숫자만 인용하세요. "15-20% 높음", "절반으로 줄이고" 같은 추정치를 절대 만들지 마세요.
- 대안 업종/지역을 제안할 때도 구체적 생존율 수치가 없으면 "추천업종 탭에서 확인하세요"로 안내하세요.
- "데이터 없음"인 항목은 언급하지 마세요.
- 등급이 A/B이면 긍정적 요소를 먼저 말하고, D 이하이면 위험을 먼저 말하세요.
- "~입니다" 체로 작성하세요.
- 면책 문구, 인사말, 마무리 인사는 넣지 마세요.
- 각 항목 앞에 **번호**를 붙여 구분하세요."""

    # 보고서 모드: 상세 분석 (유료)
    is_report = analysis_data.get('is_report', False)
    if is_report:
        prompt += """

[보고서 모드 — 상세 분석]
위 데이터를 기반으로 종합 창업 분석 보고서를 작성해주세요.
반드시 아래 9개 항목을 모두 포함하세요:

1. 종합 판정 (3~4문장)
2. 핵심 위험/기회 요인 (상위 3개, 각 1~2문장)
3. 재무 분석 (매출-임대료-인건비-순이익, 자본금 소진 시점 계산)
4. 경쟁 환경 (포화도, 동종업체, 시장 트렌드)
5. 대안 시나리오 A: 같은 지역 다른 업종 (구체적 업종명+이유)
6. 대안 시나리오 B: 같은 업종 다른 지역 (구체적 지역명+이유)
7. 프랜차이즈 비교 (해당 업종 브랜드 있으면 폐점률 기준 추천)
8. 활용 가능 지원금 안내
9. 인허가 절차 및 준비사항

각 항목을 2~3문장 이상으로 충분히 서술하세요."""

    max_tokens = 2500 if is_report else 1200

    client = _get_client()
    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}],
    )

    return response.content[0].text
