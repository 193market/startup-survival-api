import logging
from datetime import datetime

logger = logging.getLogger('monitoring')


class MonitoringService:
    """자동 모니터링 5단계 — 데이터 변동 감지 + 알림"""

    def check_data_update(self, old_data: dict, new_data: dict) -> list:
        """1~3단계: 트리거→수집→분석 (전월 대비 변동 감지)"""
        changes = []
        for slug in new_data:
            if slug not in old_data:
                continue
            for sido in new_data[slug].get('시도별', []):
                sido_name = sido.get('sido', '')
                old_slug = old_data.get(slug, {})
                old_sidos = {s['sido']: s for s in old_slug.get('시도별', [])}
                old_sido = old_sidos.get(sido_name)
                if not old_sido:
                    continue
                diff = sido['survival_rate'] - old_sido['survival_rate']
                if abs(diff) >= 3:
                    changes.append({
                        'slug': slug,
                        'sido': sido_name,
                        'old_rate': old_sido['survival_rate'],
                        'new_rate': sido['survival_rate'],
                        'diff': round(diff, 1),
                        'direction': '상승' if diff > 0 else '하락',
                    })
        return changes

    def validate_changes(self, changes: list) -> tuple:
        """4단계: 이상치 검증"""
        validated, alerts = [], []
        for c in changes:
            if abs(c['diff']) >= 15:
                alerts.append({**c, 'alert_type': '급격한 변동 — 데이터 확인 필요'})
            else:
                validated.append(c)
        return validated, alerts

    def generate_notifications(self, changes: list) -> list:
        """5단계: 알림 메시지 생성"""
        msgs = []
        for c in changes:
            msgs.append({
                'slug': c['slug'],
                'sido': c['sido'],
                'message': (
                    f"{c['sido']} {c['slug']}: "
                    f"생존율 {c['old_rate']}% → {c['new_rate']}% "
                    f"({c['diff']:+.1f}%p {c['direction']})"
                ),
                'created_at': datetime.now().isoformat(),
            })
        return msgs
