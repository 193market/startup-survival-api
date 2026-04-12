"""Supabase DB 클라이언트. 환경변수 없으면 인메모리 폴백."""
from config import SUPABASE_URL, SUPABASE_KEY

_client = None
_fallback = False


def get_client():
    global _client, _fallback
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        _fallback = True
        return None
    from supabase import create_client
    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def is_fallback() -> bool:
    if _client is None and not _fallback:
        get_client()
    return _fallback
