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


async def generate_diagnosis(analysis_data: dict) -> str:
    prompt = f"""당신은 창업 컨설턴트입니다. 아래 데이터를 바탕으로 예비 창업자에게 3~5문장의 종합 진단과 조언을 해주세요.

분석 데이터:
- 업종: {analysis_data.get('business', '')}
- 지역: {analysis_data.get('sido', '전국')} {analysis_data.get('sigungu', '')}
- 5년 생존율: {analysis_data.get('survival_rate', 'N/A')}% ({analysis_data.get('grade', 'N/A')}등급)
- 반경 동종 업체: {analysis_data.get('competitors', 'N/A')}개
- 밀집도: {analysis_data.get('density_rank', 'N/A')}
- 지역 인구: {analysis_data.get('population', 'N/A')}명
- 월 가구소득: {analysis_data.get('avg_income', 'N/A')}만원
- ML 예측 생존확률: {analysis_data.get('ml_prediction', {}).get('survival_prob', 'N/A')}
- 최대 위험 요인: {analysis_data.get('ml_prediction', {}).get('top_risk_factor_kr', 'N/A')}
- 받을 수 있는 지원금: {analysis_data.get('subsidies_count', 0)}건, 최대 {analysis_data.get('subsidies_total', 0)}만원
- 사용자 메모: "{analysis_data.get('user_note', '없음')}"

규칙:
1. 숫자를 나열하지 말고, 이 사람이 이해할 수 있는 일상 언어로 설명하세요.
2. 위험하면 솔직하게 말하되, 반드시 대안(다른 업종 또는 다른 지역)을 제시하세요.
3. 지원금 정보가 있으면 마지막에 언급하세요.
4. "~입니다" 체로 작성하세요.
5. 면책 문구는 넣지 마세요 (앱에서 별도 표시합니다)."""

    client = _get_client()
    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=500,
        messages=[{'role': 'user', 'content': prompt}],
    )

    return response.content[0].text
