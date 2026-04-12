-- 관심 입지 모니터링 등록
CREATE TABLE IF NOT EXISTS watched_locations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    business TEXT NOT NULL,
    slug TEXT,
    sido TEXT NOT NULL,
    sigungu TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_watched_user ON watched_locations(user_id);
CREATE INDEX IF NOT EXISTS idx_watched_region ON watched_locations(sido, business);

-- 변동 알림
CREATE TABLE IF NOT EXISTS notifications (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT,
    slug TEXT,
    sido TEXT NOT NULL,
    business TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    read BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notif_created ON notifications(created_at DESC);

-- RLS (Row Level Security)
ALTER TABLE watched_locations ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- 정책: 자기 데이터만 읽기/쓰기 (anon 키 사용 시 user_id 매칭)
CREATE POLICY watched_select ON watched_locations FOR SELECT USING (true);
CREATE POLICY watched_insert ON watched_locations FOR INSERT WITH CHECK (true);
CREATE POLICY watched_delete ON watched_locations FOR DELETE USING (true);

CREATE POLICY notif_select ON notifications FOR SELECT USING (true);
CREATE POLICY notif_insert ON notifications FOR INSERT WITH CHECK (true);
CREATE POLICY notif_update ON notifications FOR UPDATE USING (true);
