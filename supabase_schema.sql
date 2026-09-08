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

-- 3. Paper Trading Portfolio Table
-- Tracks account bankroll, cash, invested capital, equity, and aggregate realized P&L
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INT PRIMARY KEY DEFAULT 1,                    -- Singleton row (id = 1)
    starting_balance NUMERIC DEFAULT 100000.00 NOT NULL,
    cash_balance NUMERIC DEFAULT 100000.00 NOT NULL,
    invested_capital NUMERIC DEFAULT 0.00 NOT NULL,
    total_equity NUMERIC DEFAULT 100000.00 NOT NULL,
    total_realized_pnl NUMERIC DEFAULT 0.00 NOT NULL,
    total_trades_count INT DEFAULT 0 NOT NULL,
    win_trades_count INT DEFAULT 0 NOT NULL,
    loss_trades_count INT DEFAULT 0 NOT NULL,
    last_trade_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    CONSTRAINT single_portfolio_row CHECK (id = 1)
);

-- Seed initial portfolio balance if not exists
INSERT INTO paper_portfolio (id, starting_balance, cash_balance, total_equity)
VALUES (1, 100000.00, 100000.00, 100000.00)
ON CONFLICT (id) DO NOTHING;

-- 4. Paper Active Positions Table
-- Tracks currently active open pair trades and live floating unrealized P&L
CREATE TABLE IF NOT EXISTS paper_positions (
    pair_key TEXT PRIMARY KEY,                       -- e.g. "XOM-CVX"
    direction TEXT NOT NULL,                         -- "LONG_SPREAD" (Buy A, Short B) or "SHORT_SPREAD" (Short A, Buy B)
    ticker_a TEXT NOT NULL,
    side_a TEXT NOT NULL,                            -- "BUY" or "SELL_SHORT"
    shares_a INT NOT NULL,
    entry_price_a NUMERIC NOT NULL,
    current_price_a NUMERIC NOT NULL,
    ticker_b TEXT NOT NULL,
    side_b TEXT NOT NULL,                            -- "SELL_SHORT" or "BUY"
    shares_b INT NOT NULL,
    entry_price_b NUMERIC NOT NULL,
    current_price_b NUMERIC NOT NULL,
    locked_beta NUMERIC NOT NULL,
    entry_spread NUMERIC NOT NULL,
    current_spread NUMERIC NOT NULL,
    entry_z_score NUMERIC NOT NULL,
    current_z_score NUMERIC NOT NULL,
    capital_invested NUMERIC NOT NULL,
    unrealized_pnl NUMERIC DEFAULT 0.00 NOT NULL,
    unrealized_pnl_pct NUMERIC DEFAULT 0.00 NOT NULL,
    opened_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 5. Paper Trades Ledger Table
-- Immutable historical record of every paper trade executed (entry and exit)
CREATE TABLE IF NOT EXISTS paper_trades (
    id BIGSERIAL PRIMARY KEY,
    pair_key TEXT NOT NULL,                          -- e.g. "XOM-CVX"
    action TEXT NOT NULL,                            -- "ENTER_PAIR" or "EXIT_PAIR"
    direction TEXT NOT NULL,                         -- "LONG_SPREAD" or "SHORT_SPREAD"
    reason TEXT NOT NULL,                            -- "STATISTICAL_DIVERGENCE", "MEAN_REVERSION_ZERO_CROSSING", "MANUAL_TEST"
    ticker_a TEXT NOT NULL,
    side_a TEXT NOT NULL,                            -- "BUY" or "SELL_SHORT"
    shares_a INT NOT NULL,
    price_a NUMERIC NOT NULL,
    ticker_b TEXT NOT NULL,
    side_b TEXT NOT NULL,                            -- "SELL_SHORT" or "BUY"
    shares_b INT NOT NULL,
    price_b NUMERIC NOT NULL,
    locked_beta NUMERIC NOT NULL,
    spread NUMERIC NOT NULL,
    z_score NUMERIC NOT NULL,
    capital_allocated NUMERIC NOT NULL,
    realized_pnl NUMERIC DEFAULT 0.00,
    return_pct NUMERIC DEFAULT 0.00,
    executed_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    notes TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_candle_cache_ticker_ts ON candle_cache(ticker, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candle_cache_created ON candle_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_paper_trades_pair_ts ON paper_trades(pair_key, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_executed_at ON paper_trades(executed_at DESC);

-- RLS Policies (Allow full access for service_role or authenticated, or disable RLS for direct script access)
ALTER TABLE pair_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE candle_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_portfolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

-- Allow read/write access for authenticated and service roles
DROP POLICY IF EXISTS "Allow full access to pair_state" ON pair_state;
CREATE POLICY "Allow full access to pair_state" ON pair_state
    FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow full access to candle_cache" ON candle_cache;
CREATE POLICY "Allow full access to candle_cache" ON candle_cache
    FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow full access to paper_portfolio" ON paper_portfolio;
CREATE POLICY "Allow full access to paper_portfolio" ON paper_portfolio
    FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow full access to paper_positions" ON paper_positions;
CREATE POLICY "Allow full access to paper_positions" ON paper_positions
    FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow full access to paper_trades" ON paper_trades;
CREATE POLICY "Allow full access to paper_trades" ON paper_trades
    FOR ALL USING (true) WITH CHECK (true);

