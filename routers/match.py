import json
import os
import re
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from config import MODEL_DIR

logger = logging.getLogger('match')
router = APIRouter()

# ── 데이터 로드 ──
with open(os.path.join(MODEL_DIR, 'business_list.json'), 'r', encoding='utf-8') as f:
    BUSINESS_LIST = json.load(f)

_NAMES = [b['name_ko'] for b in BUSINESS_LIST]
_SLUG_MAP = {b['name_ko']: b['slug'] for b in BUSINESS_LIST}
_NAME_LIST_STR = ', '.join(_NAMES)

_kw_path = os.path.join(MODEL_DIR, 'keyword-map.json')
_KEYWORD_MAP: dict = {}
if os.path.exists(_kw_path):
    with open(_kw_path, 'r', encoding='utf-8') as f:
        _KEYWORD_MAP = json.load(f)

_KW_REVERSE: dict[str, tuple[str, str]] = {}
for slug, entry in _KEYWORD_MAP.items():
    name = slug
    for b in BUSINESS_LIST:
        if b['slug'] == slug:
            name = b['name_ko']
            break
    for kw in entry.get('keywords', []):
        _KW_REVERSE[kw.lower()] = (slug, name)

# ── 캐시 (파일 기반) ──
_CACHE_PATH = os.path.join(MODEL_DIR, 'match_cache.json')
_cache: dict[str, dict] = {}
if os.path.exists(_CACHE_PATH):
    with open(_CACHE_PATH, 'r', encoding='utf-8') as f:
        _cache = json.load(f)


def _save_cache():
    try:
        with open(_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(_cache, f, ensure_ascii=False, indent=0)
    except Exception:
        pass


# ── 정규화 전처리 ──
_SUFFIXES = re.compile(r'(전문)?[집가게점관샵소실]+$')
_SIOT_MAP = {'깃': '기', '곳': '고', '잇': '이', '핏': '피', '닛': '니', '깃': '기'}


def normalize(keyword: str) -> str:
    s = keyword.strip()
    s = re.sub(r'\s+', '', s)           # 띄어쓰기 제거
    s = _SUFFIXES.sub('', s)            # 접미사 제거
    # 사이시옷 변형 통일
    for old, new in _SIOT_MAP.items():
        s = s.replace(old, new)
    return s.lower()


# ── 모델 ──
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

    norm = normalize(keyword)

    # 0차: 캐시 확인
    if norm in _cache:
        c = _cache[norm]
        return MatchResponse(matched=c['matched'], slug=c['slug'], confidence=c['confidence'], reason=c['reason'])

    # 1차: 정확히 일치하는 업종명
    if keyword in _SLUG_MAP:
        return _respond_and_cache(norm, keyword, _SLUG_MAP[keyword], 1.0, '정확히 일치하는 업종명입니다.')

    # 2차: 부분 일치 (업종명)
    for name in _NAMES:
        if keyword in name or name in keyword:
            return _respond_and_cache(norm, name, _SLUG_MAP[name], 0.9, f'"{keyword}"은(는) {name}에 해당합니다.')

    # 3차: 키워드맵 매칭 (정규화 적용)
    if norm in _KW_REVERSE:
        slug, name = _KW_REVERSE[norm]
        return _respond_and_cache(norm, name, slug, 0.9, f'"{keyword}"은(는) {name}에 해당합니다.')

    # 정규화된 형태로 키워드맵 부분 매칭
    kw_lower = keyword.lower()
    for kw, (slug, name) in _KW_REVERSE.items():
        nkw = normalize(kw)
        if norm == nkw or norm in nkw or nkw in norm:
            return _respond_and_cache(norm, name, slug, 0.85, f'"{keyword}"은(는) {name} 업종으로 분류됩니다.')
    # 원본 키워드로도 부분 매칭
    for kw, (slug, name) in _KW_REVERSE.items():
        if kw_lower in kw or kw in kw_lower:
            return _respond_and_cache(norm, name, slug, 0.85, f'"{keyword}"은(는) {name} 업종으로 분류됩니다.')

    # 4차: Claude API 매핑
    try:
        from services.claude_service import _get_client
        client = _get_client()
        response = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=200,
            messages=[{'role': 'user', 'content': f"""아래 195개 업종 목록에서 "{keyword}"에 가장 적합한 업종 1개를 골라주세요.

업종 목록: {_NAME_LIST_STR}

중요 규칙:
- 같은 의미의 다른 표현은 반드시 같은 업종으로 매핑할 것.
- 예: 고기집 = 고깃집 = 삼겹살집 = 갈비집 = 곱창집 → 전부 "일반음식점"
- 예: 커피숍 = 카페 = 커피전문점 → 전부 "휴게음식점"
- 예: 네일샵 = 미용실 = 헤어샵 → 전부 "미용업"
- 반드시 위 목록에 있는 정확한 업종명을 사용할 것.

반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만:
{{"matched": "업종명", "confidence": 0.0~1.0, "reason": "한줄 이유"}}"""}],
        )
        text = response.content[0].text.strip()
        if '{' in text:
            text = text[text.index('{'):text.rindex('}') + 1]
        data = json.loads(text)
        matched_name = data.get('matched', '')
        reason = data.get('reason', '')
        conf = min(float(data.get('confidence', 0.8)), 1.0)

        if matched_name in _SLUG_MAP:
            return _respond_and_cache(norm, matched_name, _SLUG_MAP[matched_name], conf, reason)
        # AI가 반환한 이름이 목록에 없으면 유사한 것 찾기
        for name in _NAMES:
            if matched_name in name or name in matched_name:
                return _respond_and_cache(norm, name, _SLUG_MAP[name], 0.7, reason)
    except Exception as e:
        logger.warning(f'Claude API match failed: {e}')

    # 5차: 실패
    return MatchResponse(matched='', slug='', confidence=0, reason=f'"{keyword}"에 매칭되는 업종을 찾지 못했습니다.')


def _respond_and_cache(norm: str, matched: str, slug: str, confidence: float, reason: str) -> MatchResponse:
    entry = {'matched': matched, 'slug': slug, 'confidence': confidence, 'reason': reason}
    _cache[norm] = entry
    _save_cache()
    return MatchResponse(**entry)
