import logging
from datetime import datetime

logger = logging.getLogger('error_handler')


class ErrorHandler:
    def __init__(self):
        self.retry_count = {}

    async def auto_recover(self, error_type: str, func, *args, max_retries=3):
        """1단계: 단순 오류 → 재시도"""
        key = f'{error_type}:{func.__name__}'
        for attempt in range(max_retries):
            try:
                result = await func(*args)
                self.retry_count[key] = 0
                return result
            except Exception as e:
                logger.warning(f'Auto-recover {attempt+1}/{max_retries}: {e}')
                self.retry_count[key] = attempt + 1
        return None

    async def escalate_to_admin(self, error_type: str, error_detail: str):
        """2단계: 중요/불확실 오류 → 담당자 알림"""
        alert = {
            'type': error_type,
            'detail': error_detail,
            'timestamp': datetime.now().isoformat(),
            'action_required': True,
        }
        logger.error(f'ESCALATION: {alert}')
        return alert

    async def safe_shutdown(self, error_type: str, error_detail: str):
        """3단계: 치명적 오류 → 안전 중단"""
        logger.critical(f'SAFE SHUTDOWN: {error_type} - {error_detail}')
        await self.escalate_to_admin(f'CRITICAL_{error_type}', f'안전 중단: {error_detail}')
        return {'status': 'shutdown', 'reason': error_detail}
