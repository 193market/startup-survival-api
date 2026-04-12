import json
import os
from fastapi import APIRouter
from pydantic import BaseModel
from config import MODEL_DIR

router = APIRouter()

# 업종 목록 로드
with open(os.path.join(MODEL_DIR, 'business_list.json'), 'r', encoding='utf-8') as f:
    BUSINESS_LIST = json.load(f)

_NAMES = [b['name_ko'] for b in BUSINESS_LIST]
_SLUG_MAP = {b['name_ko']: b['slug'] for b in BUSINESS_LIST}
_NAME_LIST_STR = ', '.join(_NAMES)

# 키워드맵 로드 (slug -> keywords)
_kw_path = os.path.join(MODEL_DIR, 'keyword-map.json')
_KEYWORD_MAP: dict = {}
if os.path.exists(_kw_path):
    with open(_kw_path, 'r', encoding='utf-8') as f:
        _KEYWORD_MAP = json.load(f)
# 역방향: keyword -> (slug, name_ko)
_KW_REVERSE: dict[str, tuple[str, str]] = {}
for slug, entry in _KEYWORD_MAP.items():
    name = entry.get('matchHint', slug)
    # matchHint가 없으면 business_list에서 찾기
    for b in BUSINESS_LIST:
        if b['slug'] == slug:
            name = b['name_ko']
            break
    for kw in entry.get('keywords', []):
        _KW_REVERSE[kw.lower()] = (slug, name)


class MatchRequest(BaseModel):
    keyword: str


class MatchResponse(BaseModel):
    matched: str
    slug: str
    confidence: float
    reason: str


@router.post('/api/v1/match-business', response_model=MatchResponse)
async def match_business(req: MatchRequest):
    keyword = req.keyword.strip()
    if not keyword:
        return MatchResponse(matched='', slug='', confidence=0, reason='키워드가 비어있습니다.')

    # 1차: 정확히 일치하는 업종명
    if keyword in _SLUG_MAP:
        return MatchResponse(
            matched=keyword, slug=_SLUG_MAP[keyword],
            confidence=1.0, reason='정확히 일치하는 업종명입니다.',
        )

    # 2차: 부분 일치 (업종명)
    for name in _NAMES:
        if keyword in name or name in keyword:
            return MatchResponse(
                matched=name, slug=_SLUG_MAP[name],
                confidence=0.9, reason=f'"{keyword}"은(는) {name}에 해당합니다.',
            )

    # 2.5차: 키워드맵 매칭
    kw_lower = keyword.lower()
    if kw_lower in _KW_REVERSE:
        slug, name = _KW_REVERSE[kw_lower]
        return MatchResponse(
            matched=name, slug=slug,
            confidence=0.9, reason=f'"{keyword}"은(는) {name}에 해당합니다.',
        )
    # 키워드맵 부분 매칭
    for kw, (slug, name) in _KW_REVERSE.items():
        if kw_lower in kw or kw in kw_lower:
            return MatchResponse(
                matched=name, slug=slug,
                confidence=0.85, reason=f'"{keyword}"은(는) {name} 업종으로 분류됩니다.',
            )

    # 3차: Claude API 매핑
    try:
        from services.claude_service import _get_client
        client = _get_client()
        response = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=200,
            messages=[{'role': 'user', 'content': f"""아래 195개 업종 목록에서 "{keyword}"에 가장 적합한 업종 1개를 골라주세요.

업종 목록: {_NAME_LIST_STR}

반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만:
{{"matched": "업종명", "confidence": 0.0~1.0, "reason": "한줄 이유"}}"""}],
        )
        text = response.content[0].text.strip()
        # JSON 파싱
        if '{' in text:
            text = text[text.index('{'):text.rindex('}') + 1]
        data = json.loads(text)
        matched_name = data.get('matched', '')
        if matched_name in _SLUG_MAP:
            return MatchResponse(
                matched=matched_name,
                slug=_SLUG_MAP[matched_name],
                confidence=min(float(data.get('confidence', 0.8)), 1.0),
                reason=data.get('reason', ''),
            )
        # AI가 반환한 이름이 목록에 없으면 가장 유사한 것 찾기
        for name in _NAMES:
            if matched_name in name or name in matched_name:
                return MatchResponse(
                    matched=name, slug=_SLUG_MAP[name],
                    confidence=0.7, reason=data.get('reason', ''),
                )
    except Exception:
        pass

    # 4차: 폴백 — 일반음식점
    return MatchResponse(
        matched='', slug='',
        confidence=0, reason=f'"{keyword}"에 매칭되는 업종을 찾지 못했습니다.',
    )
