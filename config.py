import os

CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', '')
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
