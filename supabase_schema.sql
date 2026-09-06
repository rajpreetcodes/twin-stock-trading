-- ==============================================================================
-- Supabase Schema for Twin Stock Trading Monitor
-- Run this script in your Supabase SQL Editor to initialize required tables.
-- ==============================================================================

-- 1. Pair State Table
-- Tracks daily locked Beta, alert cooldowns, zero-crossing state, and latest metrics
CREATE TABLE IF NOT EXISTS pair_state (
    pair_key TEXT PRIMARY KEY,                       -- e.g. "KO-PEP"
    locked_beta NUMERIC,                             -- Daily locked OLS hedge ratio
    beta_date TEXT,                                  -- Date beta was locked (YYYY-MM-DD)
    last_alert_date TEXT,                            -- Date of most recent alert
    last_alert_timestamp TEXT,                       -- ISO timestamp of most recent alert
    last_alert_score NUMERIC,                        -- Z-score when alert fired
    last_alert_direction TEXT,                       -- "UPPER" or "LOWER"
    alert_sent_today BOOLEAN DEFAULT FALSE,          -- True if alert sent today
    reverted_to_zero BOOLEAN DEFAULT FALSE,          -- True if reverted to zero / crossed 0
    last_z_score NUMERIC,                            -- Most recent observed Z-score
    last_seen_timestamp TEXT,                        -- ISO timestamp of latest candle observed
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Candle Cache Table
-- Stores rolling 5-minute OHLCV candles per ticker to avoid repeated downloads
CREATE TABLE IF NOT EXISTS candle_cache (
    ticker TEXT NOT NULL,                            -- e.g. "KO"
    timestamp TEXT NOT NULL,                         -- ISO 8601 string, e.g. "2026-09-04T15:55:00-04:00"
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    PRIMARY KEY (ticker, timestamp)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_candle_cache_ticker_ts ON candle_cache(ticker, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candle_cache_created ON candle_cache(created_at);

-- RLS Policies (Allow full access for service_role or authenticated, or disable RLS for direct script access)
ALTER TABLE pair_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE candle_cache ENABLE ROW LEVEL SECURITY;

-- Allow read/write access for authenticated and service roles
CREATE POLICY "Allow full access to pair_state" ON pair_state
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow full access to candle_cache" ON candle_cache
    FOR ALL USING (true) WITH CHECK (true);
