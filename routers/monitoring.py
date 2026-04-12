from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter()


class WatchRequest(BaseModel):
    user_id: str
    business: str
    slug: Optional[str] = None
    sido: str
    sigungu: Optional[str] = None


# 인메모리 폴백 (Supabase 미설정 시)
_mem_watched: list[dict] = []
_mem_notifications: list[dict] = []


@router.post('/api/v1/watch')
async def watch_location(req: WatchRequest):
    from services.db import get_client, is_fallback

    entry = {**req.dict(), 'created_at': datetime.now().isoformat()}

    if is_fallback():
        _mem_watched.append(entry)
        total = len([w for w in _mem_watched if w['user_id'] == req.user_id])
    else:
        sb = get_client()
        sb.table('watched_locations').insert(entry).execute()
        res = sb.table('watched_locations').select('id', count='exact').eq('user_id', req.user_id).execute()
        total = res.count or 0

    return {'status': 'ok', 'message': f'{req.sido} {req.business} 모니터링 등록', 'total': total}


@router.delete('/api/v1/watch')
async def unwatch_location(user_id: str, sido: str, business: str):
    from services.db import get_client, is_fallback

    if is_fallback():
        before = len(_mem_watched)
        _mem_watched[:] = [
            w for w in _mem_watched
            if not (w['user_id'] == user_id and w['sido'] == sido and w['business'] == business)
        ]
        removed = before - len(_mem_watched)
    else:
        sb = get_client()
        sb.table('watched_locations').delete().eq('user_id', user_id).eq('sido', sido).eq('business', business).execute()
        removed = 1

    return {'status': 'ok', 'removed': removed}


@router.get('/api/v1/watch/{user_id}')
async def get_watched(user_id: str):
    from services.db import get_client, is_fallback

    if is_fallback():
        return [w for w in _mem_watched if w['user_id'] == user_id]

    sb = get_client()
    res = sb.table('watched_locations').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
    return res.data


@router.get('/api/v1/notifications/{user_id}')
async def get_notifications(user_id: str, unread_only: bool = False):
    from services.db import get_client, is_fallback

    if is_fallback():
        items = [n for n in _mem_notifications if n.get('user_id') == user_id]
        if unread_only:
            items = [n for n in items if not n.get('read')]
        return items

    sb = get_client()
    q = sb.table('notifications').select('*').eq('user_id', user_id)
    if unread_only:
        q = q.eq('read', False)
    res = q.order('created_at', desc=True).limit(50).execute()
    return res.data


@router.patch('/api/v1/notifications/{notification_id}/read')
async def mark_read(notification_id: str):
    from services.db import get_client, is_fallback

    if is_fallback():
        for n in _mem_notifications:
            if n.get('id') == notification_id:
                n['read'] = True
        return {'status': 'ok'}

    sb = get_client()
    sb.table('notifications').update({'read': True}).eq('id', notification_id).execute()
    return {'status': 'ok'}
