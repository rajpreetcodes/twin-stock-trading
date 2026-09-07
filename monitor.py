#!/usr/bin/env python3
"""
Intraday Stock Pair Monitoring System (Twin Stock Trading)
---------------------------------------------------------
Monitors intraday stock pairs using:
1. Supabase state backend (pair_state & candle_cache) replacing local git storage.
2. Daily locked Beta via 90-day daily OLS regression once per calendar day.
3. Incremental 5m candle ingestion (period="1d") with custom User-Agent and 1s delay.
4. Sustained deviation validation (>= 2.5 or <= -2.5 for at least 2 consecutive periods).
5. Cooldown with sign-inversion zero-crossing (|Z| <= 0.25 or Z_prev * Z_curr <= 0).
6. Resend email alerting.
7. Clean market hours check (America/New_York 09:30 - 16:00).
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("twin_stock_monitor")

# Custom User-Agent for yfinance requests
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 (TwinStockTrading/1.0)"
)


# ---------------------------------------------------------------------------
# Market Hours Verification
# ---------------------------------------------------------------------------
def is_market_open(
    tz_name: str = "America/New_York",
    open_time_str: str = "09:30",
    close_time_str: str = "16:00",
    now_dt: Optional[datetime.datetime] = None,
) -> bool:
    """
    Check if the exchange market is currently open.
    - Closed on weekends (Saturday=5, Sunday=6).
    - Checks local exchange time between open_time and close_time.
    """
    market_tz = pytz.timezone(tz_name)
    if now_dt is None:
        now_dt = datetime.datetime.now(pytz.utc)

    local_dt = now_dt.astimezone(market_tz)

    if local_dt.weekday() >= 5:
        return False

    current_time = local_dt.time()
    open_parts = [int(p) for p in open_time_str.split(":")]
    close_parts = [int(p) for p in close_time_str.split(":")]

    open_time = datetime.time(open_parts[0], open_parts[1])
    close_time = datetime.time(close_parts[0], close_parts[1])

    return open_time <= current_time <= close_time


# ---------------------------------------------------------------------------
# Supabase Data Access Layer (with local offline fallback)
# ---------------------------------------------------------------------------
class SupabaseBackend:
    """Wrapper around Supabase client with local memory fallback for dry-runs/tests."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        self.client = None
        self._memory_pair_state: Dict[str, Dict[str, Any]] = {}
        self._memory_candle_cache: Dict[str, List[Dict[str, Any]]] = {}

        if self.url and self.key:
            try:
                from supabase import create_client

                self.client = create_client(self.url, self.key)
                logger.info("Connected to Supabase state backend.")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}. Falling back to in-memory store.")
        else:
            logger.info("No SUPABASE_URL / SUPABASE_KEY set. Running with in-memory state store.")

    def get_pair_state(self, pair_key: str) -> Dict[str, Any]:
        """Fetch current pair state from Supabase or memory store."""
        if self.client:
            try:
                res = self.client.table("pair_state").select("*").eq("pair_key", pair_key).execute()
                if res.data and len(res.data) > 0:
                    return res.data[0]
            except Exception as e:
                logger.error(f"Error fetching pair_state for {pair_key} from Supabase: {e}")

        return self._memory_pair_state.get(
            pair_key,
            {
                "pair_key": pair_key,
                "locked_beta": None,
                "beta_date": None,
                "last_alert_date": None,
                "last_alert_timestamp": None,
                "last_alert_score": None,
                "last_alert_direction": None,
                "alert_sent_today": False,
                "reverted_to_zero": False,
                "last_z_score": None,
                "last_seen_timestamp": None,
            },
        )

    def upsert_pair_state(self, state_dict: Dict[str, Any]) -> None:
        """Save or update pair state."""
        pair_key = state_dict.get("pair_key")
        if not pair_key:
            return

        self._memory_pair_state[pair_key] = state_dict

        if self.client:
            try:
                self.client.table("pair_state").upsert(state_dict).execute()
            except Exception as e:
                logger.error(f"Error upserting pair_state for {pair_key} to Supabase: {e}")

    def upsert_candles(self, ticker: str, df: pd.DataFrame) -> None:
        """Upsert 5-minute candles into candle_cache."""
        if df.empty:
            return

        records = []
        for idx, row in df.iterrows():
            # Normalize to UTC ISO format for consistent SQL string sorting and comparisons
            try:
                dt_utc = idx.tz_convert(pytz.utc) if hasattr(idx, "tz_convert") else pd.to_datetime(idx, utc=True)
                ts_str = dt_utc.isoformat()
            except Exception:
                ts_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)

            record = {
                "ticker": ticker,
                "timestamp": ts_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row.get("Volume", 0)),
            }
            records.append(record)

        # Update in-memory cache
        existing = {r["timestamp"]: r for r in self._memory_candle_cache.get(ticker, [])}
        for r in records:
            existing[r["timestamp"]] = r
        self._memory_candle_cache[ticker] = sorted(existing.values(), key=lambda x: x["timestamp"])

        # Update Supabase
        if self.client and records:
            try:
                # Upsert in chunks to avoid payload limits
                chunk_size = 100
                for i in range(0, len(records), chunk_size):
                    chunk = records[i : i + chunk_size]
                    self.client.table("candle_cache").upsert(
                        chunk, on_conflict="ticker,timestamp"
                    ).execute()
            except Exception as e:
                logger.error(f"Error upserting candle_cache for {ticker} in Supabase: {e}")

    def load_cached_candles(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Load cached 5m candles for ticker from Supabase or memory."""
        records = []
        if self.client:
            try:
                cutoff = (datetime.datetime.now(pytz.utc) - datetime.timedelta(days=days + 4)).isoformat()
                res = (
                    self.client.table("candle_cache")
                    .select("*")
                    .eq("ticker", ticker)
                    .gte("timestamp", cutoff)
                    .order("timestamp", desc=False)
                    .limit(5000)
                    .execute()
                )
                if res.data:
                    records = res.data
            except Exception as e:
                logger.error(f"Error reading candle_cache for {ticker} from Supabase: {e}")

        if not records:
            records = self._memory_candle_cache.get(ticker, [])

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("datetime").sort_index()
        # Deduplicate index to prevent reindexing / concat errors
        df = df[~df.index.duplicated(keep="last")]
        return df

    def prune_old_candles(self, ticker: str, days_to_keep: int = 10) -> None:
        """Prune candles older than days_to_keep trading/calendar days."""
        cutoff_dt = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=days_to_keep)
        cutoff_str = cutoff_dt.isoformat()

        # Prune memory
        if ticker in self._memory_candle_cache:
            self._memory_candle_cache[ticker] = [
                r for r in self._memory_candle_cache[ticker] if r["timestamp"] >= cutoff_str
            ]

        # Prune Supabase
        if self.client:
            try:
                self.client.table("candle_cache").delete().eq("ticker", ticker).lt(
                    "timestamp", cutoff_str
                ).execute()
            except Exception as e:
                logger.error(f"Error pruning candle_cache for {ticker} in Supabase: {e}")


# ---------------------------------------------------------------------------
# Data Fetching & Daily Beta Lock Engine
# ---------------------------------------------------------------------------
def create_yf_session() -> requests.Session:
    """Create requests session configured with custom browser User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def fetch_daily_data(
    ticker: str,
    period: str = "90d",
    interval: str = "1d",
    delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch 90-day daily closing data with rate-limiting delay."""
    logger.info(f"Fetching {period} daily candles for beta calculation: {ticker}")
    session = create_yf_session()
    ticker_obj = yf.Ticker(ticker, session=session)
    df = ticker_obj.history(period=period, interval=interval)

    if delay > 0:
        time.sleep(delay)

    if df.empty or "Close" not in df.columns:
        logger.warning(f"No daily data returned for {ticker}.")
        return pd.DataFrame()

    return df


def compute_ols_beta(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Compute OLS hedge ratio (Beta) = Cov(A, B) / Var(B) using daily close prices."""
    close_a = df_a["Close"]
    close_b = df_b["Close"]

    # Align dates
    aligned = pd.concat([close_a, close_b], axis=1).dropna()
    if len(aligned) < 20:
        logger.warning("Fewer than 20 daily closing points aligned. Defaulting Beta to 1.0.")
        return 1.0

    series_a = aligned.iloc[:, 0].values
    series_b = aligned.iloc[:, 1].values

    var_b = np.var(series_b)
    if var_b <= 0:
        return 1.0

    cov_ab = np.cov(series_a, series_b)[0, 1]
    beta = float(cov_ab / var_b)
    return beta


def get_or_lock_daily_beta(
    backend: SupabaseBackend,
    ticker_a: str,
    ticker_b: str,
    pair_key: str,
    today_str: str,
    api_delay: float = 1.0,
) -> float:
    """
    Retrieves the locked daily beta from Supabase if already computed today.
    Otherwise queries 90-day daily closing data, computes OLS Beta, and locks it.
    """
    pair_state = backend.get_pair_state(pair_key)
    locked_beta = pair_state.get("locked_beta")
    beta_date = pair_state.get("beta_date")

    # Check if beta is already locked for today
    if locked_beta is not None and beta_date == today_str:
        logger.info(f"[{pair_key}] Using locked Beta from Supabase: {float(locked_beta):.4f} (locked on {beta_date})")
        return float(locked_beta)

    # Compute new Beta from 90 days of daily data
    logger.info(f"[{pair_key}] No locked Beta for {today_str}. Querying 90d daily candles...")
    df_a = fetch_daily_data(ticker_a, period="90d", interval="1d", delay=api_delay)
    df_b = fetch_daily_data(ticker_b, period="90d", interval="1d", delay=api_delay)

    if df_a.empty or df_b.empty:
        fallback_beta = float(locked_beta) if locked_beta is not None else 1.0
        logger.warning(f"[{pair_key}] Failed to fetch 90d daily data. Using fallback beta: {fallback_beta:.4f}")
        return fallback_beta

    new_beta = compute_ols_beta(df_a, df_b)
    logger.info(f"[{pair_key}] Computed new locked Beta for {today_str}: {new_beta:.4f}")

    # Lock beta in Supabase
    pair_state["locked_beta"] = new_beta
    pair_state["beta_date"] = today_str
    backend.upsert_pair_state(pair_state)

    return new_beta


def fetch_intraday_candles(
    ticker: str,
    period: str = "1d",
    interval: str = "5m",
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch intraday 5m candles (current day) with custom User-Agent and delay.
    """
    logger.info(f"Fetching {period} of {interval} candles for ticker: {ticker}")
    session = create_yf_session()
    ticker_obj = yf.Ticker(ticker, session=session)
    df = ticker_obj.history(period=period, interval=interval)

    if delay > 0:
        time.sleep(delay)

    if df.empty or "Close" not in df.columns:
        logger.warning(f"No intraday candle data returned for {ticker}.")
        return pd.DataFrame()

    return df


# ---------------------------------------------------------------------------
# Statistical Calculations on Cached 7-Day History
# ---------------------------------------------------------------------------
def compute_pair_metrics_from_cache(
    cached_df_a: pd.DataFrame,
    cached_df_b: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    locked_beta: float,
    rolling_window: int = 78,
) -> Optional[Dict[str, Any]]:
    """
    Aligns 5m cached candles, applies locked Beta:
    Spread = Price_A - (Locked_Beta * Price_B)
    Computes rolling mean, rolling std, and Z-score series.
    """
    col_a = "close" if "close" in cached_df_a.columns else "Close"
    col_b = "close" if "close" in cached_df_b.columns else "Close"
    close_a = cached_df_a[col_a].rename(ticker_a)
    close_b = cached_df_b[col_b].rename(ticker_b)

    aligned = pd.concat([close_a, close_b], axis=1).dropna()
    aligned = aligned[~aligned.index.duplicated(keep="last")]
    if len(aligned) < max(20, rolling_window // 4):
        logger.warning(
            f"Insufficient aligned cached candles ({len(aligned)}) for pair {ticker_a}-{ticker_b}."
        )
        return None

    # Spread with locked Beta
    spread = aligned[ticker_a] - (locked_beta * aligned[ticker_b])

    # Rolling mean and std dev
    min_periods = max(10, rolling_window // 4)
    rolling_mean = spread.rolling(window=rolling_window, min_periods=min_periods).mean()
    rolling_std = spread.rolling(window=rolling_window, min_periods=min_periods).std()

    safe_std = rolling_std.replace(0, np.nan)
    z_scores = (spread - rolling_mean) / safe_std
    z_scores = z_scores.dropna()

    if len(z_scores) < 2:
        logger.warning(f"Not enough valid Z-score points for {ticker_a}-{ticker_b}.")
        return None

    latest_idx = z_scores.index[-1]
    latest_ts = latest_idx.isoformat() if hasattr(latest_idx, "isoformat") else str(latest_idx)

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "locked_beta": locked_beta,
        "aligned_count": len(aligned),
        "spread_series": spread,
        "z_scores": z_scores,
        "latest_z": float(z_scores.iloc[-1]),
        "prev_z": float(z_scores.iloc[-2]),
        "latest_price_a": float(aligned[ticker_a].iloc[-1]),
        "latest_price_b": float(aligned[ticker_b].iloc[-1]),
        "latest_spread": float(spread.iloc[-1]),
        "rolling_mean": float(rolling_mean.dropna().iloc[-1]),
        "rolling_std": float(rolling_std.dropna().iloc[-1]),
        "timestamp": latest_ts,
    }


# ---------------------------------------------------------------------------
# Sustained Deviation Verification
# ---------------------------------------------------------------------------
def check_sustained_deviation(
    z_scores: pd.Series,
    threshold: float = 2.5,
    min_consecutive_periods: int = 2,
) -> Tuple[bool, str, List[float]]:
    """
    Sustained Deviation Rule:
    A deviation is only valid if the score stays above +2.5 (or below -2.5)
    for at least min_consecutive_periods (default: 2) consecutive 5m periods.
    """
    if len(z_scores) < min_consecutive_periods:
        return False, "NONE", []

    recent = [float(z) for z in z_scores.iloc[-min_consecutive_periods:]]

    if all(score >= threshold for score in recent):
        return True, "UPPER", recent

    if all(score <= -threshold for score in recent):
        return True, "LOWER", recent

    return False, "NONE", recent


# ---------------------------------------------------------------------------
# Cooldown & Zero-Crossing Logic
# ---------------------------------------------------------------------------
def evaluate_cooldown_and_zero_crossing(
    pair_state: Dict[str, Any],
    today_date_str: str,
    current_z: float,
    prev_z: float,
    is_sustained: bool,
    direction: str,
    zero_threshold: float = 0.25,
) -> Tuple[bool, str]:
    """
    Evaluates cooldown and mean-reversion re-arming:
    - If alert_sent_today is True:
      Re-arm trigger if |current_z| <= zero_threshold OR (prev_z * current_z <= 0).
      This prevents rapid price swings from skipping the cooldown reset.
    - Day rollover resets alert_sent_today and re-arms.
    """
    last_alert_date = pair_state.get("last_alert_date")

    # Day rollover check
    if last_alert_date != today_date_str:
        pair_state["alert_sent_today"] = False
        pair_state["reverted_to_zero"] = False

    alert_active = pair_state.get("alert_sent_today", False)

    # Check zero-crossing or reversion to zero
    if alert_active:
        reversion_detected = False
        if abs(current_z) <= zero_threshold:
            reversion_detected = True
        elif prev_z * current_z <= 0:
            reversion_detected = True

        if reversion_detected and not pair_state.get("reverted_to_zero", False):
            logger.info(
                f"[{pair_state.get('pair_key')}] Reversion to zero detected "
                f"(current_z={current_z:+.2f}, prev_z={prev_z:+.2f}). Re-arming trigger."
            )
            pair_state["reverted_to_zero"] = True

    # Evaluate alert decision
    should_alert = False
    reason = "No sustained deviation"

    if is_sustained:
        if not alert_active:
            should_alert = True
            reason = "First sustained deviation of the day"
        elif pair_state.get("reverted_to_zero", False):
            should_alert = True
            reason = "Sustained deviation after returning to zero / sign inversion"
        else:
            reason = "Cooldown active: alert already sent today without returning to zero"

    return should_alert, reason


# ---------------------------------------------------------------------------
# Resend Email Alerting
# ---------------------------------------------------------------------------
def format_email_content(
    ticker_a: str,
    ticker_b: str,
    current_z: float,
    recent_scores: List[float],
    price_a: float,
    price_b: float,
    spread: float,
    rolling_mean: float,
    rolling_std: float,
    locked_beta: float,
    timestamp_str: str,
    direction: str,
    is_test: bool = False,
) -> Tuple[str, str, str]:
    """
    Generate subject, HTML body, and plain-text body for Resend email.
    Crafted under high-agency UI/UX principles: strict typography hierarchy,
    calibrated desaturated accents, dark neutral surfaces, and bulletproof
    cross-client table architecture.
    """
    if direction == "UPPER":
        tag = "OVERBOUGHT (UPPER)"
        status_label = "Statistical Expansion: Upper Deviation Threshold"
        action_summary = f"Spread (P_{ticker_a} - &beta;P_{ticker_b}) is significantly elevated above its 78-period mean. Mean-reversion posture favors SHORT {ticker_a} / LONG {ticker_b}."
        accent_color = "#f43f5e"  # Rose-500
        accent_bg = "rgba(244, 63, 94, 0.12)"
        accent_border = "rgba(244, 63, 94, 0.32)"
        tag_text = "#fb7185"
    elif direction == "LOWER":
        tag = "OVERSOLD (LOWER)"
        status_label = "Statistical Compression: Lower Deviation Threshold"
        action_summary = f"Spread (P_{ticker_a} - &beta;P_{ticker_b}) is depressed below its 78-period mean. Mean-reversion posture favors LONG {ticker_a} / SHORT {ticker_b}."
        accent_color = "#10b981"  # Emerald-500
        accent_bg = "rgba(16, 185, 129, 0.12)"
        accent_border = "rgba(16, 185, 129, 0.32)"
        tag_text = "#34d399"
    else:
        tag = "NEUTRAL / VERIFICATION"
        status_label = "Statistical Baseline Verification"
        action_summary = f"Spread is within expected deviation bands. Live test verifying Resend API connectivity and calculation pipeline."
        accent_color = "#38bdf8"  # Sky-400
        accent_bg = "rgba(56, 189, 248, 0.12)"
        accent_border = "rgba(56, 189, 248, 0.32)"
        tag_text = "#7dd3fc"

    prefix = "[TEST VERIFICATION] " if is_test else ""
    subject = f"{prefix}Pair Alert: {ticker_a}/{ticker_b} Z-Score {current_z:+.2f} ({tag})"

    # Format discrete consecutive confirmation pills
    scores_pills_html = ""
    for idx, s in enumerate(recent_scores):
        period_label = f"t-{len(recent_scores) - 1 - idx}" if idx < len(recent_scores) - 1 else "latest"
        scores_pills_html += (
            f'<span style="display: inline-block; padding: 4px 10px; margin: 2px 4px; '
            f'background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); '
            f'border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; '
            f'font-size: 12px; color: #e2e8f0;">{period_label}: <strong style="color: {accent_color};">{s:+.2f}&sigma;</strong></span>'
        )

    scores_formatted_text = " -> ".join([f"{s:+.2f}σ" for s in recent_scores])

    test_banner_html = ""
    if is_test:
        test_banner_html = """
        <div style="background-color: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 14px 18px; margin-bottom: 24px;">
          <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38bdf8; margin-bottom: 4px;">System Verification Mode Active</div>
          <div style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">This transmission confirms that your end-to-end monitoring pipeline, Resend credentials, Supabase state store, and HTML email engine are fully operational.</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subject}</title>
</head>
<body style="margin: 0; padding: 32px 16px; background-color: #090d16; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; color: #f8fafc; -webkit-font-smoothing: antialiased;">
  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; margin: 0 auto; background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid {accent_color}; border-radius: 12px; overflow: hidden; box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);">
    <!-- Header -->
    <tr>
      <td style="padding: 24px 28px 16px 28px; border-bottom: 1px solid rgba(255, 255, 255, 0.06);">
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td>
              <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: #94a3b8;">Twin Stock Trading // Intraday Statistical Monitor</div>
              <div style="font-size: 24px; font-weight: 700; letter-spacing: -0.02em; color: #ffffff; margin-top: 4px;">{ticker_a} <span style="color: #64748b; font-weight: 400;">/</span> {ticker_b}</div>
            </td>
            <td align="right" valign="top">
              <span style="display: inline-block; padding: 4px 12px; background: {accent_bg}; border: 1px solid {accent_border}; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; color: {tag_text}; text-transform: uppercase;">{tag}</span>
            </td>
          </tr>
        </table>
      </td>
    </tr>

    <!-- Main Content Area -->
    <tr>
      <td style="padding: 24px 28px;">
        {test_banner_html}

        <!-- Deviation Hero Metric Box -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #0b0f19; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; margin-bottom: 22px;">
          <tr>
            <td style="padding: 22px 24px; text-align: center;">
              <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 6px;">Statistical Deviation Score (Z-Score)</div>
              <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 44px; font-weight: 800; letter-spacing: -0.03em; color: {accent_color}; margin: 2px 0 10px 0;">{current_z:+.2f}&sigma;</div>
              <div style="font-size: 12px; color: #94a3b8;">
                Consecutive Confirmation:
                <div style="margin-top: 8px;">{scores_pills_html}</div>
              </div>
            </td>
          </tr>
        </table>

        <!-- Quantitative Posture & Direction -->
        <div style="background-color: rgba(255, 255, 255, 0.02); border-left: 2px solid {accent_color}; padding: 12px 16px; margin-bottom: 24px; border-radius: 0 6px 6px 0;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #cbd5e1; margin-bottom: 3px;">Quantitative Signal Posture</div>
          <div style="font-size: 13px; line-height: 1.5; color: #94a3b8;">{action_summary}</div>
        </div>

        <!-- Component Price Cards -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="margin-bottom: 24px;">
          <tr>
            <td width="48%" style="background-color: #0d1322; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 14px 16px;">
              <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">Stock A (Base)</div>
              <div style="font-size: 15px; font-weight: 700; color: #ffffff; margin-top: 2px;">{ticker_a}</div>
              <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 20px; font-weight: 700; color: #f8fafc; margin-top: 6px;">${price_a:,.2f}</div>
            </td>
            <td width="4%"></td>
            <td width="48%" style="background-color: #0d1322; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 14px 16px;">
              <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">Stock B (Hedge &beta; {locked_beta:.4f})</div>
              <div style="font-size: 15px; font-weight: 700; color: #ffffff; margin-top: 2px;">{ticker_b}</div>
              <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 20px; font-weight: 700; color: #f8fafc; margin-top: 6px;">${price_b:,.2f}</div>
            </td>
          </tr>
        </table>

        <!-- High-Density Spread Metric Rows -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="border-collapse: collapse;">
          <tr>
            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 13px; color: #94a3b8;">Daily Locked Beta (&beta;)</td>
            <td align="right" style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 600; color: #f8fafc;">{locked_beta:.4f}</td>
          </tr>
          <tr>
            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 13px; color: #94a3b8;">Current Spread (P<sub>A</sub> &minus; &beta;P<sub>B</sub>)</td>
            <td align="right" style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 600; color: #f8fafc;">${spread:,.4f}</td>
          </tr>
          <tr>
            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 13px; color: #94a3b8;">Rolling Mean Spread (78-Period)</td>
            <td align="right" style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 600; color: #f8fafc;">${rolling_mean:,.4f}</td>
          </tr>
          <tr>
            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 13px; color: #94a3b8;">Rolling Spread Volatility (&sigma;)</td>
            <td align="right" style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 600; color: #f8fafc;">${rolling_std:,.4f}</td>
          </tr>
          <tr>
            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 13px; color: #94a3b8;">Sampling Granularity</td>
            <td align="right" style="padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 600; color: #f8fafc;">5-Minute Candles (7-Day Rolling Cache)</td>
          </tr>
          <tr>
            <td style="padding: 10px 0; font-size: 13px; color: #94a3b8;">Observation Timestamp</td>
            <td align="right" style="padding: 10px 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 600; color: #94a3b8;">{timestamp_str}</td>
          </tr>
        </table>
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td style="padding: 20px 28px; background-color: #0b0f19; border-top: 1px solid rgba(255, 255, 255, 0.06); text-align: center;">
        <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; color: #64748b;">
          Twin Stock Trading &bull; Automated via GitHub Actions, Supabase &amp; Resend
        </div>
        <div style="font-size: 11px; color: #475569; margin-top: 4px;">
          Statistically disciplined pair monitoring. Cooldown reset enforced at |Z| &le; 0.25 or sign-inversion zero-crossing.
        </div>
      </td>
    </tr>
  </table>
</body>
</html>"""

    text_content = f"""TWIN STOCK TRADING // STATISTICAL PAIR ALERT
=====================================================
Pair: {ticker_a} / {ticker_b}
Condition: {tag}
Status: {status_label}
Deviation Score: {current_z:+.2f} sigma
Consecutive Confirmation: {scores_formatted_text}

ACTIONABLE SIGNAL POSTURE:
{action_summary}

PRICE SUMMARY:
- {ticker_a} (Base Asset):   ${price_a:,.2f}
- {ticker_b} (Hedge Asset):  ${price_b:,.2f}

STATISTICAL PARAMETERS:
- Daily Locked Beta:         {locked_beta:.4f}
- Current Spread:            ${spread:,.4f}
- Rolling Mean Spread:       ${rolling_mean:,.4f}
- Rolling Spread Volatility: ${rolling_std:,.4f}
- Observation Window:        5m Intraday (7-day cache)
- Timestamp:                 {timestamp_str}

=====================================================
Automated notification dispatched via Resend SDK.
"""

    return subject, html_content, text_content


def send_alert_email(
    subject: str,
    html_content: str,
    text_content: str,
    dry_run: bool = False,
) -> bool:
    """Send email using the Resend Python SDK."""
    if dry_run:
        logger.info(f"[DRY RUN] Email suppressed. Subject: {subject}")
        return True

    api_key = os.environ.get("RESEND_API_KEY")
    to_addr = os.environ.get("ALERT_EMAIL_TO")
    from_addr = os.environ.get("ALERT_EMAIL_FROM", "onboarding@resend.dev")

    if not api_key:
        logger.error("RESEND_API_KEY is not configured. Email suppressed.")
        return False

    if not to_addr:
        logger.error("ALERT_EMAIL_TO is not configured. Email suppressed.")
        return False

    try:
        import resend

        resend.api_key = api_key
        params = {
            "from": from_addr,
            "to": [addr.strip() for addr in to_addr.split(",") if addr.strip()],
            "subject": subject,
            "html": html_content,
            "text": text_content,
        }
        response = resend.Emails.send(params)
        logger.info(f"Resend email dispatched successfully: {response}")
        return True
    except Exception as e:
        logger.error(f"Failed to dispatch Resend email: {e}")
        return False


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def parse_pairs(pairs_input: Optional[str]) -> List[Tuple[str, str]]:
    """Parse comma-separated pairs like 'KO:PEP,V:MA'."""
    raw = pairs_input or os.environ.get("PAIRS_CONFIG", "KO:PEP,V:MA,HD:LOW,XOM:CVX")
    parsed: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":")
        elif "-" in item:
            parts = item.split("-")
        else:
            continue
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            parsed.append((parts[0].strip().upper(), parts[1].strip().upper()))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intraday stock pair statistical deviation monitor (Twin Stock Trading)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass market hours check (runs even if market is closed or weekend).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending real emails.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated stock pairs, e.g. 'KO:PEP,MSFT:AAPL'.",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Dispatch a test alert email using current pair metrics to verify Resend delivery and email formatting.",
    )
    args = parser.parse_args()

    # Environment overrides
    tz_name = os.environ.get("MARKET_TIMEZONE", "America/New_York")
    open_str = os.environ.get("MARKET_OPEN", "09:30")
    close_str = os.environ.get("MARKET_CLOSE", "16:00")
    threshold = float(os.environ.get("Z_SCORE_THRESHOLD", "2.5"))
    sustained_periods = int(os.environ.get("SUSTAINED_PERIODS", "2"))
    zero_threshold = float(os.environ.get("ZERO_THRESHOLD", "0.25"))
    rolling_window = int(os.environ.get("ROLLING_WINDOW", "78"))
    api_delay = float(os.environ.get("YFINANCE_DELAY_SECONDS", "1.0"))
    ignore_hours_env = os.environ.get("IGNORE_MARKET_HOURS", "false").lower() in ("true", "1", "yes")
    test_alert_mode = args.test_alert or os.environ.get("TEST_ALERT", "false").lower() in ("true", "1", "yes")

    # 1. Market Hours Check
    if not (args.force or ignore_hours_env or test_alert_mode):
        if not is_market_open(tz_name=tz_name, open_time_str=open_str, close_time_str=close_str):
            logger.info("Market is currently closed. Exiting silently.")
            return 0

    if test_alert_mode and not is_market_open(tz_name=tz_name, open_time_str=open_str, close_time_str=close_str):
        logger.info("Market is closed, but --test-alert is active. Bypassing market hours check for verification.")
    else:
        logger.info("Market is open (or bypass active). Starting pair evaluation.")

    market_tz = pytz.timezone(tz_name)
    now_market = datetime.datetime.now(market_tz)
    today_date_str = now_market.strftime("%Y-%m-%d")

    # 2. Initialize Supabase Backend
    backend = SupabaseBackend()
    pairs = parse_pairs(args.pairs)
    logger.info(f"Monitoring {len(pairs)} pairs: {pairs}")

    evaluated_candidates: List[Dict[str, Any]] = []
    alerts_dispatched = 0

    # 3. Process Each Pair
    for ticker_a, ticker_b in pairs:
        pair_key = f"{ticker_a}-{ticker_b}"
        logger.info(f"=== Evaluating Pair: {pair_key} ===")

        try:
            # A. Daily Beta Lock (from 90-day daily closing data)
            locked_beta = get_or_lock_daily_beta(
                backend=backend,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                pair_key=pair_key,
                today_str=today_date_str,
                api_delay=api_delay,
            )

            # B. Incremental 5m Data Ingestion
            # Check if cache needs initial backfill
            cached_a = backend.load_cached_candles(ticker_a, days=7)
            cached_b = backend.load_cached_candles(ticker_b, days=7)

            period_a = "7d" if len(cached_a) < 50 else "1d"
            period_b = "7d" if len(cached_b) < 50 else "1d"

            new_df_a = fetch_intraday_candles(ticker_a, period=period_a, interval="5m", delay=api_delay)
            new_df_b = fetch_intraday_candles(ticker_b, period=period_b, interval="5m", delay=api_delay)

            if new_df_a.empty or new_df_b.empty:
                logger.warning(f"[{pair_key}] Fresh candle data unavailable for {ticker_a} or {ticker_b}. Skipping evaluation.")
                continue

            backend.upsert_candles(ticker_a, new_df_a)
            backend.prune_old_candles(ticker_a, days_to_keep=10)

            backend.upsert_candles(ticker_b, new_df_b)
            backend.prune_old_candles(ticker_b, days_to_keep=10)

            # Load full 7-day 5m dataset from cache
            full_cached_a = backend.load_cached_candles(ticker_a, days=7)
            full_cached_b = backend.load_cached_candles(ticker_b, days=7)

            if full_cached_a.empty or full_cached_b.empty:
                logger.warning(f"[{pair_key}] Missing candle data from cache. Skipping.")
                continue

            # C. Compute Spread & Rolling Z-Score
            metrics = compute_pair_metrics_from_cache(
                cached_df_a=full_cached_a,
                cached_df_b=full_cached_b,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                locked_beta=locked_beta,
                rolling_window=rolling_window,
            )

            if not metrics:
                continue

            current_z = metrics["latest_z"]
            prev_z = metrics["prev_z"]
            z_series = metrics["z_scores"]

            # D. Sustained Deviation Verification
            is_sustained, direction, recent_scores = check_sustained_deviation(
                z_scores=z_series,
                threshold=threshold,
                min_consecutive_periods=sustained_periods,
            )

            logger.info(
                f"[{pair_key}] Latest Z: {current_z:+.2f} | Prev Z: {prev_z:+.2f} | "
                f"Sustained: {is_sustained} ({direction})"
            )

            # Record for potential test verification dispatch
            evaluated_candidates.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "pair_key": pair_key,
                "current_z": current_z,
                "prev_z": prev_z,
                "recent_scores": recent_scores if recent_scores else [prev_z, current_z],
                "direction": direction if direction != "NONE" else ("UPPER" if current_z >= 0 else "LOWER"),
                "metrics": metrics,
                "locked_beta": locked_beta,
            })

            # E. Cooldown & Zero-Crossing Logic
            pair_state = backend.get_pair_state(pair_key)
            should_alert, reason = evaluate_cooldown_and_zero_crossing(
                pair_state=pair_state,
                today_date_str=today_date_str,
                current_z=current_z,
                prev_z=prev_z,
                is_sustained=is_sustained,
                direction=direction,
                zero_threshold=zero_threshold,
            )

            logger.info(f"[{pair_key}] Alert Decision: {should_alert} (Reason: {reason})")

            # F. Dispatch Alert or Update Observation
            if should_alert:
                subject, html_body, text_body = format_email_content(
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    current_z=current_z,
                    recent_scores=recent_scores,
                    price_a=metrics["latest_price_a"],
                    price_b=metrics["latest_price_b"],
                    spread=metrics["latest_spread"],
                    rolling_mean=metrics["rolling_mean"],
                    rolling_std=metrics["rolling_std"],
                    locked_beta=locked_beta,
                    timestamp_str=metrics["timestamp"],
                    direction=direction,
                    is_test=False,
                )

                email_sent = send_alert_email(
                    subject=subject,
                    html_content=html_body,
                    text_content=text_body,
                    dry_run=args.dry_run,
                )

                if email_sent:
                    alerts_dispatched += 1
                    pair_state["last_alert_date"] = today_date_str
                    pair_state["last_alert_timestamp"] = metrics["timestamp"]
                    pair_state["last_alert_score"] = float(current_z)
                    pair_state["last_alert_direction"] = direction
                    pair_state["alert_sent_today"] = True
                    pair_state["reverted_to_zero"] = False

            pair_state["last_z_score"] = float(current_z)
            pair_state["last_seen_timestamp"] = metrics["timestamp"]
            backend.upsert_pair_state(pair_state)

        except Exception as e:
            logger.error(f"Unexpected error processing pair {pair_key}: {e}", exc_info=True)

    # G. Verification Test Dispatch (if requested and no real alert triggered)
    if test_alert_mode and alerts_dispatched == 0 and evaluated_candidates:
        # Choose the pair with the most pronounced statistical divergence
        best = max(evaluated_candidates, key=lambda p: abs(p["current_z"]))
        logger.info(
            f"[TEST ALERT] Initiating system verification alert dispatch for pair {best['pair_key']} "
            f"(Z = {best['current_z']:+.2f}σ)..."
        )
        b_metrics = best["metrics"]
        subject, html_body, text_body = format_email_content(
            ticker_a=best["ticker_a"],
            ticker_b=best["ticker_b"],
            current_z=best["current_z"],
            recent_scores=best["recent_scores"],
            price_a=b_metrics["latest_price_a"],
            price_b=b_metrics["latest_price_b"],
            spread=b_metrics["latest_spread"],
            rolling_mean=b_metrics["rolling_mean"],
            rolling_std=b_metrics["rolling_std"],
            locked_beta=best["locked_beta"],
            timestamp_str=b_metrics["timestamp"],
            direction=best["direction"],
            is_test=True,
        )
        test_sent = send_alert_email(
            subject=subject,
            html_content=html_body,
            text_content=text_body,
            dry_run=args.dry_run,
        )
        if test_sent:
            logger.info(f"[TEST ALERT] Verification alert successfully dispatched for {best['pair_key']}.")
        else:
            logger.error(f"[TEST ALERT] Failed to send verification alert email.")

    logger.info("Twin Stock Trading monitoring run completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
