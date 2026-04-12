import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import diagnosis, monitoring, match

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

app = FastAPI(
    title='창업 생존율 분석 API',
    version='1.0.0',
    description='ML 예측 + AI 자연어 진단 + 모니터링',
)

ALLOWED_ORIGINS = [
    'https://toss.im',
    'https://*.toss.im',
    'https://startup-survival.vercel.app',
    'https://*.vercel.app',
    'http://localhost:5173',
    'http://localhost:5174',
    'http://localhost:5175',
    'http://localhost:5176',
    'http://localhost:5177',
    'http://localhost:5178',
    'http://localhost:5179',
    'http://localhost:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r'(https://.*\.vercel\.app|http://localhost:\d+)',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(diagnosis.router, tags=['diagnosis'])
app.include_router(monitoring.router, tags=['monitoring'])
app.include_router(match.router, tags=['match'])


@app.get('/')
def root():
    return {
        'service': '창업 생존율 분석 API',
        'version': '1.0.0',
        'endpoints': [
            'POST /api/v1/match-business',
            'POST /api/v1/diagnosis',
            'GET  /api/v1/predict?slug=...&sido=...&survival_rate=...',
            'POST /api/v1/watch',
            'DELETE /api/v1/watch?user_id=...&sido=...&business=...',
            'GET  /api/v1/watch/{user_id}',
            'GET  /api/v1/notifications/{user_id}',
            'PATCH /api/v1/notifications/{id}/read',
        ],
    }


@app.get('/health')
def health():
    from services.db import is_fallback
    from config import CLAUDE_API_KEY
    return {
        'status': 'ok',
        'ml_model': 'loaded',
        'supabase': 'fallback(in-memory)' if is_fallback() else 'connected',
        'claude_api': 'configured' if CLAUDE_API_KEY else 'not_configured',
    }
