"""
Unit tests for the Twin Stock Trading Monitoring System.
"""

import datetime
import numpy as np
import pandas as pd
import pytest
import pytz

from monitor import (
    DEFAULT_USER_AGENT,
    SupabaseBackend,
    calculate_trade_sizing,
    check_sustained_deviation,
    compute_ols_beta,
    compute_pair_metrics_from_cache,
    create_yf_session,
    evaluate_cooldown_and_zero_crossing,
    format_email_content,
    format_human_timestamp,
    is_market_open,
    parse_pairs,
)


# ---------------------------------------------------------------------------
# Test Market Hours Verification
# ---------------------------------------------------------------------------
def test_is_market_open_weekdays_and_weekends():
    tz = pytz.timezone("America/New_York")

    # Wednesday 10:30 AM (Open)
    wed_open = tz.localize(datetime.datetime(2026, 9, 2, 10, 30, 0))
    assert is_market_open("America/New_York", "09:30", "16:00", now_dt=wed_open) is True

    # Wednesday 09:15 AM (Pre-market - Closed)
    wed_early = tz.localize(datetime.datetime(2026, 9, 2, 9, 15, 0))
    assert is_market_open("America/New_York", "09:30", "16:00", now_dt=wed_early) is False

    # Wednesday 04:05 PM (After-hours - Closed)
    wed_late = tz.localize(datetime.datetime(2026, 9, 2, 16, 5, 0))
    assert is_market_open("America/New_York", "09:30", "16:00", now_dt=wed_late) is False

    # Saturday 11:00 AM (Weekend - Closed)
    sat_noon = tz.localize(datetime.datetime(2026, 9, 5, 11, 0, 0))
    assert is_market_open("America/New_York", "09:30", "16:00", now_dt=sat_noon) is False

    # Sunday 02:00 PM (Weekend - Closed)
    sun_afternoon = tz.localize(datetime.datetime(2026, 9, 6, 14, 0, 0))
    assert is_market_open("America/New_York", "09:30", "16:00", now_dt=sun_afternoon) is False


# ---------------------------------------------------------------------------
# Test Daily OLS Beta Calculation
# ---------------------------------------------------------------------------
def test_compute_ols_beta():
    n = 90
    dates = pd.date_range("2026-06-01", periods=n, freq="B")
    np.random.seed(42)

    price_b = 100 + np.cumsum(np.random.randn(n) * 0.5)
    # Price A cointegrated with Price B with Beta = 0.65
    noise = np.random.randn(n) * 0.1
    price_a = 40 + (0.65 * price_b) + noise

    df_a = pd.DataFrame({"Close": price_a}, index=dates)
    df_b = pd.DataFrame({"Close": price_b}, index=dates)

    beta = compute_ols_beta(df_a, df_b)
    assert 0.60 < beta < 0.70


# ---------------------------------------------------------------------------
# Test Supabase Backend State & Candle Cache (Memory Store)
# ---------------------------------------------------------------------------
def test_supabase_backend_memory_mode():
    backend = SupabaseBackend(url=None, key=None)
    pair_key = "KO-PEP"

    # 1. Initial empty state
    state = backend.get_pair_state(pair_key)
    assert state["pair_key"] == pair_key
    assert state["locked_beta"] is None

    # 2. Update state with locked beta
    state["locked_beta"] = 0.45
    state["beta_date"] = "2026-09-08"
    backend.upsert_pair_state(state)

    retrieved = backend.get_pair_state(pair_key)
    assert retrieved["locked_beta"] == 0.45
    assert retrieved["beta_date"] == "2026-09-08"

    # 3. Candle Cache upsert and load
    now = datetime.datetime.now(pytz.utc)
    dates = [now - datetime.timedelta(minutes=5 * i) for i in range(10)][::-1]
    df_candles = pd.DataFrame(
        {
            "Open": [50.0 + i for i in range(10)],
            "High": [51.0 + i for i in range(10)],
            "Low": [49.5 + i for i in range(10)],
            "Close": [50.5 + i for i in range(10)],
            "Volume": [1000 * (i + 1) for i in range(10)],
        },
        index=dates,
    )

    backend.upsert_candles("KO", df_candles)
    cached = backend.load_cached_candles("KO", days=7)
    assert len(cached) == 10
    assert "close" in cached.columns

    # 4. Pruning test
    backend.prune_old_candles("KO", days_to_keep=1)
    assert len(backend.load_cached_candles("KO", days=7)) == 10  # recent candles not pruned


# ---------------------------------------------------------------------------
# Test Sustained Deviation Rule
# ---------------------------------------------------------------------------
def test_sustained_deviation_rule():
    # Sustained upper deviation (>= 2.5 for 2 periods)
    z_upper = pd.Series([0.5, 1.2, 2.1, 2.6, 2.8])
    is_sustained, direction, recent = check_sustained_deviation(z_upper, threshold=2.5, min_consecutive_periods=2)
    assert is_sustained is True
    assert direction == "UPPER"
    assert recent == [2.6, 2.8]

    # Sustained lower deviation (<= -2.5 for 2 periods)
    z_lower = pd.Series([-0.5, -1.5, -2.2, -2.7, -3.0])
    is_sustained, direction, recent = check_sustained_deviation(z_lower, threshold=2.5, min_consecutive_periods=2)
    assert is_sustained is True
    assert direction == "LOWER"
    assert recent == [-2.7, -3.0]

    # False Positive: Single period spike
    z_spike = pd.Series([0.5, 1.2, 1.8, 1.9, 2.9])
    is_sustained, direction, recent = check_sustained_deviation(z_spike, threshold=2.5, min_consecutive_periods=2)
    assert is_sustained is False
    assert direction == "NONE"


# ---------------------------------------------------------------------------
# Test Cooldown & Sign-Inversion Zero-Crossing Logic
# ---------------------------------------------------------------------------
def test_cooldown_and_sign_inversion_rearming():
    pair_state = {
        "pair_key": "KO-PEP",
        "last_alert_date": None,
        "alert_sent_today": False,
        "reverted_to_zero": False,
    }
    today = "2026-09-08"

    # Step 1: First sustained deviation -> Should alert
    should_alert, reason = evaluate_cooldown_and_zero_crossing(
        pair_state=pair_state,
        today_date_str=today,
        current_z=2.85,
        prev_z=2.65,
        is_sustained=True,
        direction="UPPER",
        zero_threshold=0.25,
    )
    assert should_alert is True
    assert "First sustained deviation" in reason

    # Simulate alert dispatch
    pair_state["last_alert_date"] = today
    pair_state["alert_sent_today"] = True
    pair_state["reverted_to_zero"] = False

    # Step 2: Next run, still sustained at 2.90 -> Cooldown blocks alert
    should_alert, reason = evaluate_cooldown_and_zero_crossing(
        pair_state=pair_state,
        today_date_str=today,
        current_z=2.90,
        prev_z=2.85,
        is_sustained=True,
        direction="UPPER",
        zero_threshold=0.25,
    )
    assert should_alert is False
    assert "Cooldown active" in reason

    # Step 3: Sign Inversion occurs! Previous Z was +0.50, Current Z is -0.60
    # (prev_z * current_z <= 0) => Rapid cross through zero
    should_alert, reason = evaluate_cooldown_and_zero_crossing(
        pair_state=pair_state,
        today_date_str=today,
        current_z=-0.60,
        prev_z=0.50,
        is_sustained=False,
        direction="NONE",
        zero_threshold=0.25,
    )
    assert should_alert is False
    assert pair_state["reverted_to_zero"] is True  # Re-armed via sign inversion!

    # Step 4: Now lower deviation occurs and is sustained -> Allowed to alert!
    should_alert, reason = evaluate_cooldown_and_zero_crossing(
        pair_state=pair_state,
        today_date_str=today,
        current_z=-2.75,
        prev_z=-2.60,
        is_sustained=True,
        direction="LOWER",
        zero_threshold=0.25,
    )
    assert should_alert is True
    assert "returning to zero / sign inversion" in reason

    # Step 5: Day rollover resets daily state
    tomorrow = "2026-09-09"
    should_alert, reason = evaluate_cooldown_and_zero_crossing(
        pair_state=pair_state,
        today_date_str=tomorrow,
        current_z=2.70,
        prev_z=2.60,
        is_sustained=True,
        direction="UPPER",
        zero_threshold=0.25,
    )
    assert should_alert is True
    assert "First sustained deviation" in reason


# ---------------------------------------------------------------------------
# Test Spread Calculation with Locked Beta
# ---------------------------------------------------------------------------
def test_compute_pair_metrics_from_cache():
    n = 80
    dates = pd.date_range("2026-09-01 09:30", periods=n, freq="5min", tz="UTC")

    np.random.seed(42)
    price_b = 100 + np.cumsum(np.random.randn(n) * 0.1)
    locked_beta = 0.5
    price_a = 50 + (locked_beta * price_b) + (np.random.randn(n) * 0.05)

    df_a = pd.DataFrame({"close": price_a}, index=dates)
    df_b = pd.DataFrame({"close": price_b}, index=dates)

    metrics = compute_pair_metrics_from_cache(
        cached_df_a=df_a,
        cached_df_b=df_b,
        ticker_a="AAA",
        ticker_b="BBB",
        locked_beta=locked_beta,
        rolling_window=20,
    )

    assert metrics is not None
    assert metrics["locked_beta"] == 0.5
    assert "latest_z" in metrics
    assert "prev_z" in metrics
    assert "latest_spread" in metrics


# ---------------------------------------------------------------------------
# Test Email Content Formatting & Paper Trade Presentation
# ---------------------------------------------------------------------------
def test_format_email_content():
    subject, html_content, text_content = format_email_content(
        ticker_a="KO",
        ticker_b="PEP",
        current_z=2.85,
        recent_scores=[2.65, 2.85],
        price_a=62.45,
        price_b=171.20,
        spread=0.125,
        rolling_mean=0.012,
        rolling_std=0.039,
        locked_beta=0.3645,
        timestamp_str="2026-09-08 14:35:00 America/New_York",
        direction="UPPER",
    )

    assert "Paper Trade: Short KO + Long PEP" in subject
    assert "KO" in subject and "PEP" in subject
    assert "$62.45" in html_content and "$171.20" in html_content
    assert "0.3645" in html_content
    assert "PAPER TRADE EXECUTED" in html_content
    assert "Everyday Paper Portfolio Snapshot" in html_content
    assert "Quant Reference:" in html_content
    assert "WHAT WE TRADED:" in text_content


def test_custom_user_agent_session():
    session = create_yf_session()
    assert "User-Agent" in session.headers
    assert "TwinStockTrading" in session.headers["User-Agent"]


def test_parse_pairs():
    # Explicit custom pairs
    pairs = parse_pairs("KO:PEP, V:MA, HD:LOW, XOM:CVX")
    assert pairs == [("KO", "PEP"), ("V", "MA"), ("HD", "LOW"), ("XOM", "CVX")]

    # Default True Twins fallback
    default_pairs = parse_pairs(None)
    assert default_pairs == [("KO", "PEP"), ("V", "MA"), ("HD", "LOW"), ("XOM", "CVX")]


def test_format_email_content_verification_mode():
    subject, html_content, text_content = format_email_content(
        ticker_a="XOM",
        ticker_b="CVX",
        current_z=-2.07,
        recent_scores=[-2.08, -2.07],
        price_a=118.50,
        price_b=158.20,
        spread=-1.2345,
        rolling_mean=-0.1200,
        rolling_std=0.5400,
        locked_beta=0.7417,
        timestamp_str="2026-09-07 16:00:00 UTC",
        direction="LOWER",
        is_test=True,
    )

    assert "[TEST]" in subject
    assert "Paper Trade: Long XOM + Short CVX" in subject
    assert "System Verification Mode Active" in html_content
    assert "$118.50" in html_content and "$158.20" in html_content
    assert "0.7417" in html_content
    assert "PAPER TRADE EXECUTED" in html_content
    assert "TWIN STOCK TRADING // PAPER TRADING DESK" in text_content


# ---------------------------------------------------------------------------
# Test Trade Sizing & Position Calculations
# ---------------------------------------------------------------------------
def test_calculate_trade_sizing():
    shares_a, shares_b, cap_a, cap_b, total_cap = calculate_trade_sizing(
        price_a=100.0,
        price_b=200.0,
        locked_beta=0.5,
        target_leg_dollars=10000.0,
    )
    assert shares_a == 100
    assert cap_a == 10000.0
    # target_b = 10000 * 0.5 = 5000. 5000 / 200 = 25 shares
    assert shares_b == 25
    assert cap_b == 5000.0
    assert total_cap == 15000.0


# ---------------------------------------------------------------------------
# Test Human-Readable Timestamp Formatting
# ---------------------------------------------------------------------------
def test_format_human_timestamp():
    formatted = format_human_timestamp("2026-09-04T19:55:00+00:00")
    assert "Sep 04, 2026" in formatted
    assert "EDT" in formatted or "PM" in formatted or "AM" in formatted


# ---------------------------------------------------------------------------
# Test Paper Trading Engine Persistence (In-Memory Backend)
# ---------------------------------------------------------------------------
def test_paper_trading_persistence_memory_mode():
    backend = SupabaseBackend(url=None, key=None)

    # 1. Portfolio initialization
    portfolio = backend.get_portfolio()
    assert portfolio["cash_balance"] == 100000.0
    assert portfolio["total_equity"] == 100000.0

    # 2. Save Position
    pos = {
        "pair_key": "XOM-CVX",
        "direction": "LONG_SPREAD",
        "side_a": "BUY",
        "ticker_a": "XOM",
        "shares_a": 84,
        "entry_price_a": 118.50,
        "current_price_a": 118.50,
        "side_b": "SELL_SHORT",
        "ticker_b": "CVX",
        "shares_b": 46,
        "entry_price_b": 158.20,
        "current_price_b": 158.20,
        "locked_beta": 0.7417,
        "entry_z_score": -2.07,
        "capital_invested": 17231.20,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
    }
    backend.save_position(pos)
    fetched_pos = backend.get_open_position("XOM-CVX")
    assert fetched_pos is not None
    assert fetched_pos["shares_a"] == 84

    all_positions = backend.get_all_open_positions()
    assert len(all_positions) == 1

    # 3. Record Trade
    trade = {
        "trade_id": "test-uuid-1234",
        "pair_key": "XOM-CVX",
        "action": "ENTER_PAIR",
        "direction": "LONG_SPREAD",
        "side_a": "BUY",
        "ticker_a": "XOM",
        "shares_a": 84,
        "price_a": 118.50,
        "capital_a": 9954.0,
        "side_b": "SELL_SHORT",
        "ticker_b": "CVX",
        "shares_b": 46,
        "price_b": 158.20,
        "capital_b": 7277.20,
        "total_capital": 17231.20,
        "z_score_at_trade": -2.07,
        "beta_used": 0.7417,
        "realized_pnl": 0.0,
        "executed_at": "2026-09-07T16:00:00Z",
    }
    backend.record_trade(trade)
    recent = backend.get_recent_trades(5)
    assert len(recent) == 1
    assert recent[0]["trade_id"] == "test-uuid-1234"

    # 4. Delete Position
    backend.delete_position("XOM-CVX")
    assert backend.get_open_position("XOM-CVX") is None
    assert len(backend.get_all_open_positions()) == 0


