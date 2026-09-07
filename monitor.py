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
import uuid
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

    # -----------------------------------------------------------------------
    # Paper Trading Persistence (Portfolio, Active Positions, Trades Ledger)
    # -----------------------------------------------------------------------
    def get_portfolio(self) -> Dict[str, Any]:
        """Fetch singleton paper portfolio state."""
        default_portfolio = {
            "id": 1,
            "starting_balance": 100000.00,
            "cash_balance": 100000.00,
            "invested_capital": 0.00,
            "total_equity": 100000.00,
            "total_realized_pnl": 0.00,
            "total_trades_count": 0,
            "win_trades_count": 0,
            "loss_trades_count": 0,
            "last_trade_at": None,
        }
        if self.client:
            try:
                res = self.client.table("paper_portfolio").select("*").eq("id", 1).execute()
                if res.data and len(res.data) > 0:
                    row = res.data[0]
                    for k in ["starting_balance", "cash_balance", "invested_capital", "total_equity", "total_realized_pnl"]:
                        if k in row and row[k] is not None:
                            row[k] = float(row[k])
                    return row
            except Exception as e:
                logger.debug(f"Could not load paper_portfolio from Supabase: {e}. Using local store.")

        if not hasattr(self, "_memory_portfolio") or not self._memory_portfolio:
            self._memory_portfolio = default_portfolio
        return self._memory_portfolio

    def update_portfolio(self, portfolio_dict: Dict[str, Any]) -> None:
        """Upsert singleton paper portfolio state."""
        portfolio_dict["id"] = 1
        portfolio_dict["updated_at"] = datetime.datetime.now(pytz.utc).isoformat()
        self._memory_portfolio = portfolio_dict

        if self.client:
            try:
                self.client.table("paper_portfolio").upsert(portfolio_dict).execute()
            except Exception as e:
                logger.error(f"Error updating paper_portfolio in Supabase: {e}")

    def get_open_position(self, pair_key: str) -> Optional[Dict[str, Any]]:
        """Fetch open position for a given pair."""
        if self.client:
            try:
                res = self.client.table("paper_positions").select("*").eq("pair_key", pair_key).execute()
                if res.data and len(res.data) > 0:
                    pos = res.data[0]
                    for k in ["entry_price_a", "current_price_a", "entry_price_b", "current_price_b", "locked_beta", "entry_spread", "current_spread", "entry_z_score", "current_z_score", "capital_invested", "unrealized_pnl", "unrealized_pnl_pct"]:
                        if k in pos and pos[k] is not None:
                            pos[k] = float(pos[k])
                    return pos
            except Exception as e:
                logger.debug(f"Could not fetch paper_position for {pair_key} from Supabase: {e}")

        if not hasattr(self, "_memory_positions"):
            self._memory_positions = {}
        return self._memory_positions.get(pair_key)

    def get_all_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch all currently open paper positions."""
        if self.client:
            try:
                res = self.client.table("paper_positions").select("*").execute()
                if res.data:
                    positions = []
                    for pos in res.data:
                        for k in ["entry_price_a", "current_price_a", "entry_price_b", "current_price_b", "locked_beta", "entry_spread", "current_spread", "entry_z_score", "current_z_score", "capital_invested", "unrealized_pnl", "unrealized_pnl_pct"]:
                            if k in pos and pos[k] is not None:
                                pos[k] = float(pos[k])
                        positions.append(pos)
                    return positions
            except Exception as e:
                logger.debug(f"Could not fetch all paper_positions from Supabase: {e}")

        if not hasattr(self, "_memory_positions"):
            self._memory_positions = {}
        return list(self._memory_positions.values())

    def save_position(self, position_dict: Dict[str, Any]) -> None:
        """Save or update an active open position."""
        pair_key = position_dict.get("pair_key")
        if not pair_key:
            return
        if not hasattr(self, "_memory_positions"):
            self._memory_positions = {}
        self._memory_positions[pair_key] = position_dict

        if self.client:
            try:
                self.client.table("paper_positions").upsert(position_dict).execute()
            except Exception as e:
                logger.error(f"Error saving paper_position for {pair_key} in Supabase: {e}")

    def delete_position(self, pair_key: str) -> None:
        """Remove a closed position from active positions table."""
        if hasattr(self, "_memory_positions") and pair_key in self._memory_positions:
            del self._memory_positions[pair_key]

        if self.client:
            try:
                self.client.table("paper_positions").delete().eq("pair_key", pair_key).execute()
            except Exception as e:
                logger.error(f"Error deleting paper_position for {pair_key} in Supabase: {e}")

    def record_trade(self, trade_dict: Dict[str, Any]) -> None:
        """Log a trade execution event to the historical ledger."""
        if not hasattr(self, "_memory_trades"):
            self._memory_trades = []
        self._memory_trades.append(trade_dict)

        if self.client:
            try:
                self.client.table("paper_trades").insert(trade_dict).execute()
            except Exception as e:
                logger.error(f"Error recording paper_trade in Supabase: {e}")

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent trades from historical ledger."""
        if self.client:
            try:
                res = self.client.table("paper_trades").select("*").order("executed_at", desc=True).limit(limit).execute()
                if res.data:
                    return res.data
            except Exception as e:
                logger.debug(f"Could not fetch recent paper_trades from Supabase: {e}")

        if not hasattr(self, "_memory_trades"):
            self._memory_trades = []
        return sorted(self._memory_trades, key=lambda x: x.get("executed_at", ""), reverse=True)[:limit]


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
# Paper Trading Helpers & Email Formatting
# ---------------------------------------------------------------------------
def format_human_timestamp(ts_str: str) -> str:
    """Format raw ISO timestamp into a clean readable string like 'Sep 04, 2026 • 3:55 PM EDT'."""
    try:
        dt = pd.to_datetime(ts_str)
        if dt.tz is None:
            dt = dt.tz_localize(pytz.utc)
        dt_est = dt.tz_convert("America/New_York")
        return dt_est.strftime("%b %d, %Y • %I:%M %p %Z")
    except Exception:
        clean = str(ts_str).replace("T", " ")
        return clean[:19]


def calculate_trade_sizing(
    price_a: float,
    price_b: float,
    locked_beta: float,
    target_leg_dollars: float = 10000.0,
) -> Tuple[int, int, float, float, float]:
    """
    Computes position sizes:
    shares_a = floor(target_leg_dollars / price_a)
    shares_b = floor((shares_a * price_a * abs(locked_beta)) / price_b)
    Returns: (shares_a, shares_b, capital_a, capital_b, total_capital)
    """
    shares_a = max(1, int(target_leg_dollars / max(0.01, price_a)))
    capital_a = round(shares_a * price_a, 2)

    target_b_dollars = capital_a * abs(locked_beta)
    shares_b = max(1, int(target_b_dollars / max(0.01, price_b)))
    capital_b = round(shares_b * price_b, 2)

    total_capital = round(capital_a + capital_b, 2)
    return shares_a, shares_b, capital_a, capital_b, total_capital


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
    event_type: str = "NEW_TRADE",
    trade_info: Optional[Dict[str, Any]] = None,
    portfolio_info: Optional[Dict[str, Any]] = None,
    open_positions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str, str]:
    """
    Executive Paper Trading Confirmation & Daily Portfolio Statement.
    Answers directly: 'What did you trade on paper? What trades took place?'
    Displays exact shares, entry prices, dollars allocated, and live portfolio equity.
    Relegates technical metrics (Z-score, Beta) to a clean, unobtrusive audit footnote.
    """
    human_ts = format_human_timestamp(timestamp_str)

    # Derive trade info if not provided directly
    if not trade_info:
        sh_a, sh_b, cap_a, cap_b, tot_cap = calculate_trade_sizing(price_a, price_b, locked_beta)
        if direction == "UPPER":
            s_a, s_b, t_dir = "SELL_SHORT", "BUY", "SHORT_SPREAD"
        else:
            s_a, s_b, t_dir = "BUY", "SELL_SHORT", "LONG_SPREAD"

        trade_info = {
            "pair_key": f"{ticker_a}-{ticker_b}",
            "action": "ENTER_PAIR" if event_type != "EXIT_TRADE" else "EXIT_PAIR",
            "direction": t_dir,
            "side_a": s_a,
            "shares_a": sh_a,
            "price_a": price_a,
            "capital_a": cap_a,
            "side_b": s_b,
            "shares_b": sh_b,
            "price_b": price_b,
            "capital_b": cap_b,
            "total_capital": tot_cap,
            "realized_pnl": 0.0,
            "return_pct": 0.0,
        }

    # Derive portfolio info if not provided
    if not portfolio_info:
        tot_cap = trade_info.get("total_capital", 17000.0)
        portfolio_info = {
            "starting_balance": 100000.00,
            "cash_balance": round(100000.00 - tot_cap, 2),
            "invested_capital": tot_cap,
            "total_equity": 100000.00,
            "total_realized_pnl": 0.00,
        }

    if open_positions is None:
        open_positions = [{
            "pair_key": f"{ticker_a}-{ticker_b}",
            "direction": trade_info.get("direction", "LONG_SPREAD"),
            "side_a": trade_info.get("side_a", "BUY"),
            "shares_a": trade_info.get("shares_a", 50),
            "entry_price_a": price_a,
            "current_price_a": price_a,
            "side_b": trade_info.get("side_b", "SELL_SHORT"),
            "shares_b": trade_info.get("shares_b", 35),
            "entry_price_b": price_b,
            "current_price_b": price_b,
            "capital_invested": trade_info.get("total_capital", 17000.0),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
        }]

    side_a = trade_info.get("side_a", "BUY")
    shares_a = trade_info.get("shares_a", 0)
    pr_a = trade_info.get("price_a", price_a)
    cap_a = trade_info.get("capital_a", round(shares_a * pr_a, 2))

    side_b = trade_info.get("side_b", "SELL_SHORT")
    shares_b = trade_info.get("shares_b", 0)
    pr_b = trade_info.get("price_b", price_b)
    cap_b = trade_info.get("capital_b", round(shares_b * pr_b, 2))

    total_cap = trade_info.get("total_capital", round(cap_a + cap_b, 2))
    realized_pnl = trade_info.get("realized_pnl", 0.0)
    return_pct = trade_info.get("return_pct", 0.0)

    # Event titles & summary
    is_exit = event_type == "EXIT_TRADE"
    if is_exit:
        badge_text = "POSITION CLOSED"
        badge_bg = "#ecfdf5" if realized_pnl >= 0 else "#fef2f2"
        badge_color = "#059669" if realized_pnl >= 0 else "#dc2626"
        pnl_sign = "+" if realized_pnl >= 0 else ""
        headline = f"Closed {ticker_a}/{ticker_b} &bull; Net P&L: {pnl_sign}${realized_pnl:,.2f} ({return_pct:+.2f}%)"
        trade_summary = (
            f"Mean reversion achieved. Closed position in {ticker_a} and {ticker_b}. "
            f"Net return of {pnl_sign}${realized_pnl:,.2f} returned to portfolio cash balance."
        )
        subject = f"{'[TEST] ' if is_test else ''}Position Closed: {ticker_a}/{ticker_b} P&L {pnl_sign}${realized_pnl:,.2f}"
    else:
        badge_text = "PAPER TRADE EXECUTED"
        badge_bg = "#ecfdf5"
        badge_color = "#059669"
        if side_a == "BUY":
            headline = f"Long {ticker_a} + Short {ticker_b}"
            trade_summary = (
                f"Bought {shares_a} shares of {ticker_a} at ${pr_a:,.2f} (${cap_a:,.2f}) "
                f"and shorted {shares_b} shares of {ticker_b} at ${pr_b:,.2f} (${cap_b:,.2f}). "
                f"Total capital allocated: ${total_cap:,.2f}."
            )
            subject = f"{'[TEST] ' if is_test else ''}Paper Trade: Long {ticker_a} + Short {ticker_b} (${total_cap:,.2f} Allocated)"
        else:
            headline = f"Short {ticker_a} + Long {ticker_b}"
            trade_summary = (
                f"Shorted {shares_a} shares of {ticker_a} at ${pr_a:,.2f} (${cap_a:,.2f}) "
                f"and bought {shares_b} shares of {ticker_b} at ${pr_b:,.2f} (${cap_b:,.2f}). "
                f"Total capital allocated: ${total_cap:,.2f}."
            )
            subject = f"{'[TEST] ' if is_test else ''}Paper Trade: Short {ticker_a} + Long {ticker_b} (${total_cap:,.2f} Allocated)"

    # Format positions table rows
    positions_rows_html = ""
    for pos in open_positions:
        p_key = pos.get("pair_key", "")
        p_dir = "LONG" if "LONG" in pos.get("direction", "") else "SHORT"
        p_cap = pos.get("capital_invested", 0.0)
        p_pnl = pos.get("unrealized_pnl", 0.0)
        p_pct = pos.get("unrealized_pnl_pct", 0.0)
        pnl_col = "#059669" if p_pnl >= 0 else "#dc2626"
        pnl_str = f"{'+' if p_pnl >= 0 else ''}${p_pnl:,.2f} ({p_pct:+.2f}%)"
        positions_rows_html += f"""
        <tr>
          <td style="padding: 10px 12px; border-bottom: 1px solid #f1f5f9; font-size: 13px; font-weight: 700; color: #0f172a;">{p_key}</td>
          <td style="padding: 10px 12px; border-bottom: 1px solid #f1f5f9; font-size: 12px; color: #475569;">{p_dir} SPREAD</td>
          <td align="right" style="padding: 10px 12px; border-bottom: 1px solid #f1f5f9; font-size: 13px; font-weight: 600; color: #334155; white-space: nowrap;">${p_cap:,.2f}</td>
          <td align="right" style="padding: 10px 12px; border-bottom: 1px solid #f1f5f9; font-size: 13px; font-weight: 700; color: {pnl_col}; white-space: nowrap;">{pnl_str}</td>
        </tr>
        """

    if not open_positions:
        positions_rows_html = """
        <tr>
          <td colspan="4" style="padding: 14px; text-align: center; font-size: 13px; color: #94a3b8;">No active open positions. 100% in cash.</td>
        </tr>
        """

    side_a_title = "BOUGHT (LONG)" if side_a == "BUY" else "SHORTED (SELL SHORT)"
    side_a_col = "#059669" if side_a == "BUY" else "#dc2626"
    side_b_title = "SHORTED (SELL SHORT)" if side_b == "SELL_SHORT" else "BOUGHT (LONG)"
    side_b_col = "#dc2626" if side_b == "SELL_SHORT" else "#059669"

    test_banner_html = ""
    if is_test:
        test_banner_html = f"""
        <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #2563eb; letter-spacing: 0.5px;">System Verification Mode Active</div>
          <div style="font-size: 13px; color: #1e40af; margin-top: 2px;">This transmission confirms that your simulated paper trading pipeline, Supabase trade persistence, and email formatting are operating correctly.</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <meta name="supported-color-schemes" content="light dark">
  <title>{subject}</title>
</head>
<body style="margin: 0; padding: 24px 12px; background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; color: #0f172a; -webkit-font-smoothing: antialiased;">
  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 580px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);">
    <!-- Header -->
    <tr>
      <td style="padding: 22px 26px; background-color: #0f172a; color: #ffffff;">
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td>
              <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;">Twin Stock Trading &bull; Paper Trading Desk</div>
              <div style="font-size: 22px; font-weight: 800; color: #ffffff; margin-top: 4px; letter-spacing: -0.5px;">Trade Execution Confirmation</div>
              <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; white-space: nowrap;">Executed: {human_ts}</div>
            </td>
            <td align="right" valign="top">
              <span style="display: inline-block; padding: 4px 10px; background-color: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 9999px; font-size: 10px; font-weight: 700; color: #f8fafc; text-transform: uppercase; white-space: nowrap;">Live Paper Sim</span>
            </td>
          </tr>
        </table>
      </td>
    </tr>

    <!-- Main Content Area -->
    <tr>
      <td style="padding: 24px 26px;">
        {test_banner_html}

        <!-- Trade Action Headline Box -->
        <div style="background-color: {badge_bg}; border: 1px solid {badge_color}33; border-radius: 10px; padding: 18px; margin-bottom: 22px;">
          <span style="display: inline-block; background-color: {badge_color}; color: #ffffff; font-size: 10px; font-weight: 800; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;">{badge_text}</span>
          <div style="font-size: 20px; font-weight: 800; color: #0f172a; margin-top: 8px; letter-spacing: -0.3px;">{headline}</div>
          <div style="font-size: 13px; color: #334155; line-height: 1.5; margin-top: 6px;">{trade_summary}</div>
        </div>

        <!-- 2-Leg Trade Breakdown -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="margin-bottom: 22px;">
          <tr>
            <td width="48%" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid {side_a_col}; border-radius: 8px; padding: 14px 16px;">
              <div style="font-size: 11px; font-weight: 800; color: {side_a_col}; text-transform: uppercase;">{side_a_title}</div>
              <div style="font-size: 22px; font-weight: 800; color: #0f172a; margin-top: 2px;">{ticker_a}</div>
              <div style="font-size: 13px; font-weight: 600; color: #334155; margin-top: 6px; white-space: nowrap;">{shares_a} shares @ ${pr_a:,.2f}</div>
              <div style="font-size: 12px; color: #64748b; margin-top: 2px; white-space: nowrap;">Total: <strong>${cap_a:,.2f}</strong></div>
            </td>
            <td width="4%"></td>
            <td width="48%" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid {side_b_col}; border-radius: 8px; padding: 14px 16px;">
              <div style="font-size: 11px; font-weight: 800; color: {side_b_col}; text-transform: uppercase;">{side_b_title}</div>
              <div style="font-size: 22px; font-weight: 800; color: #0f172a; margin-top: 2px;">{ticker_b}</div>
              <div style="font-size: 13px; font-weight: 600; color: #334155; margin-top: 6px; white-space: nowrap;">{shares_b} shares @ ${pr_b:,.2f}</div>
              <div style="font-size: 12px; color: #64748b; margin-top: 2px; white-space: nowrap;">Total: <strong>${cap_b:,.2f}</strong></div>
            </td>
          </tr>
        </table>

        <!-- Everyday Portfolio Ledger Section -->
        <div style="border-top: 1px solid #e2e8f0; padding-top: 20px; margin-top: 20px;">
          <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; margin-bottom: 12px;">Everyday Paper Portfolio Snapshot</div>
          
          <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; margin-bottom: 18px;">
            <tr>
              <td width="33%" style="padding: 14px 10px; text-align: center; border-right: 1px solid #e2e8f0;">
                <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase;">Total Portfolio</div>
                <div style="font-size: 17px; font-weight: 800; color: #0f172a; margin-top: 3px; white-space: nowrap;">${portfolio_info.get('total_equity', 100000.00):,.2f}</div>
              </td>
              <td width="33%" style="padding: 14px 10px; text-align: center; border-right: 1px solid #e2e8f0;">
                <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase;">Available Cash</div>
                <div style="font-size: 17px; font-weight: 800; color: #0f172a; margin-top: 3px; white-space: nowrap;">${portfolio_info.get('cash_balance', 100000.00):,.2f}</div>
              </td>
              <td width="34%" style="padding: 14px 10px; text-align: center;">
                <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase;">Active Positions</div>
                <div style="font-size: 17px; font-weight: 800; color: #2563eb; margin-top: 3px; white-space: nowrap;">{len(open_positions)} Open</div>
              </td>
            </tr>
          </table>

          <!-- Active Positions Table -->
          <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; margin-bottom: 8px;">Active Open Positions</div>
          <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="border-collapse: collapse; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;">
            <tr style="background-color: #f1f5f9;">
              <th align="left" style="padding: 8px 12px; font-size: 11px; color: #475569; font-weight: 700;">PAIR</th>
              <th align="left" style="padding: 8px 12px; font-size: 11px; color: #475569; font-weight: 700;">DIRECTION</th>
              <th align="right" style="padding: 8px 12px; font-size: 11px; color: #475569; font-weight: 700;">CAPITAL</th>
              <th align="right" style="padding: 8px 12px; font-size: 11px; color: #475569; font-weight: 700;">FLOATING P&amp;L</th>
            </tr>
            {positions_rows_html}
          </table>
        </div>
      </td>
    </tr>

    <!-- Discreet Quant Footnote for Auditing -->
    <tr>
      <td style="padding: 14px 26px; background-color: #f8fafc; border-top: 1px solid #e2e8f0; font-size: 11px; color: #64748b; line-height: 1.5;">
        <strong>Quant Reference:</strong> Daily Locked Beta: <strong>{locked_beta:.4f}</strong> &bull; Spread: <strong>${spread:,.4f}</strong> (Mean: ${rolling_mean:,.4f}, &sigma;: ${rolling_std:,.4f}) &bull; Divergence Z: <strong>{current_z:+.2f}&sigma;</strong> &bull; Raw Timestamp: {timestamp_str}<br>
        <strong>Database Sync:</strong> Logged to Supabase tables <code>paper_trades</code>, <code>paper_positions</code>, and <code>paper_portfolio</code>.
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td style="padding: 16px 26px; background-color: #0f172a; text-align: center; color: #94a3b8; font-size: 11px;">
        Twin Stock Trading Desk &bull; Automated via GitHub Actions, Supabase &amp; Resend
      </td>
    </tr>
  </table>
</body>
</html>"""

    text_content = f"""TWIN STOCK TRADING // PAPER TRADING DESK
=====================================================
Status: {badge_text}
Trade: {headline}
Executed: {human_ts}

WHAT WE TRADED:
- {side_a_title}: {shares_a} shares of {ticker_a} @ ${pr_a:,.2f} (${cap_a:,.2f})
- {side_b_title}: {shares_b} shares of {ticker_b} @ ${pr_b:,.2f} (${cap_b:,.2f})
- Total Capital Allocated: ${total_cap:,.2f}

PORTFOLIO SNAPSHOT:
- Total Account Value: ${portfolio_info.get('total_equity', 100000.00):,.2f}
- Cash Available:      ${portfolio_info.get('cash_balance', 100000.00):,.2f}
- Active Open Pairs:   {len(open_positions)}

DATABASE LOGGING:
Trade and position details saved to Supabase (paper_trades, paper_positions, paper_portfolio).

QUANT REFERENCE:
- Daily Locked Beta: {locked_beta:.4f}
- Current Spread: ${spread:,.4f} (Mean: ${rolling_mean:,.4f})
- Divergence Score: {current_z:+.2f} sigma
- Raw Timestamp: {timestamp_str}
=====================================================
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
    from_addr = (os.environ.get("ALERT_EMAIL_FROM") or "").strip() or "onboarding@resend.dev"

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

    # 2. Initialize Supabase Backend & Paper Portfolio
    backend = SupabaseBackend()
    portfolio = backend.get_portfolio()
    open_positions_map = {p["pair_key"]: p for p in backend.get_all_open_positions()}
    logger.info(
        f"Paper Portfolio Initialized: Balance=${portfolio.get('cash_balance', 100000.0):,.2f} | "
        f"Total Equity=${portfolio.get('total_equity', 100000.0):,.2f} | "
        f"Open Positions={len(open_positions_map)}"
    )

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

            # E. Position Tracking & Mean-Reversion Exit Evaluation
            open_pos = backend.get_open_position(pair_key)
            if open_pos:
                cur_price_a = metrics["latest_price_a"]
                cur_price_b = metrics["latest_price_b"]
                side_a = open_pos.get("side_a", "BUY")
                side_b = open_pos.get("side_b", "SELL_SHORT")
                sh_a = open_pos.get("shares_a", 0)
                sh_b = open_pos.get("shares_b", 0)
                ent_a = open_pos.get("entry_price_a", cur_price_a)
                ent_b = open_pos.get("entry_price_b", cur_price_b)

                pnl_a = (cur_price_a - ent_a) * sh_a if side_a == "BUY" else (ent_a - cur_price_a) * sh_a
                pnl_b = (cur_price_b - ent_b) * sh_b if side_b == "BUY" else (ent_b - cur_price_b) * sh_b
                unrealized_pnl = round(pnl_a + pnl_b, 2)
                cap_inv = open_pos.get("capital_invested", 1.0)
                unrealized_pct = round((unrealized_pnl / cap_inv) * 100.0, 2) if cap_inv > 0 else 0.0

                open_pos["current_price_a"] = cur_price_a
                open_pos["current_price_b"] = cur_price_b
                open_pos["unrealized_pnl"] = unrealized_pnl
                open_pos["unrealized_pnl_pct"] = unrealized_pct
                open_pos["last_updated"] = metrics["timestamp"]
                backend.save_position(open_pos)

                # Check for Mean-Reversion Exit Signal
                # Exit occurs if spread reverted to zero (|Z| <= zero_threshold) or crossed zero (prev_z * current_z <= 0)
                if abs(current_z) <= zero_threshold or (prev_z * current_z <= 0 and abs(prev_z) > zero_threshold):
                    logger.info(f"[{pair_key}] Mean-reversion exit condition met (Z={current_z:+.2f}). Closing paper position...")
                    exit_trade = {
                        "trade_id": str(uuid.uuid4()),
                        "pair_key": pair_key,
                        "action": "EXIT_PAIR",
                        "direction": open_pos.get("direction", "SPREAD"),
                        "side_a": "SELL" if side_a == "BUY" else "BUY_TO_COVER",
                        "ticker_a": ticker_a,
                        "shares_a": sh_a,
                        "price_a": cur_price_a,
                        "capital_a": round(sh_a * cur_price_a, 2),
                        "side_b": "BUY_TO_COVER" if side_b == "SELL_SHORT" else "SELL",
                        "ticker_b": ticker_b,
                        "shares_b": sh_b,
                        "price_b": cur_price_b,
                        "capital_b": round(sh_b * cur_price_b, 2),
                        "total_capital": cap_inv,
                        "z_score_at_trade": float(current_z),
                        "beta_used": locked_beta,
                        "realized_pnl": unrealized_pnl,
                        "executed_at": metrics["timestamp"],
                    }
                    backend.record_trade(exit_trade)
                    backend.delete_position(pair_key)

                    # Update portfolio balances
                    portfolio["cash_balance"] = round(portfolio.get("cash_balance", 100000.0) + cap_inv + unrealized_pnl, 2)
                    portfolio["realized_pnl"] = round(portfolio.get("realized_pnl", 0.0) + unrealized_pnl, 2)
                    current_open_positions = backend.get_all_open_positions()
                    portfolio["open_positions_count"] = len(current_open_positions)
                    total_floating = sum(p.get("unrealized_pnl", 0.0) for p in current_open_positions)
                    invested_capital = sum(p.get("capital_invested", 0.0) for p in current_open_positions)
                    portfolio["total_equity"] = round(portfolio["cash_balance"] + invested_capital + total_floating, 2)
                    portfolio["last_updated"] = metrics["timestamp"]
                    backend.update_portfolio(portfolio)

                    # Dispatch Position Closed Email
                    subject, html_body, text_body = format_email_content(
                        ticker_a=ticker_a,
                        ticker_b=ticker_b,
                        current_z=current_z,
                        recent_scores=recent_scores,
                        price_a=cur_price_a,
                        price_b=cur_price_b,
                        spread=metrics["latest_spread"],
                        rolling_mean=metrics["rolling_mean"],
                        rolling_std=metrics["rolling_std"],
                        locked_beta=locked_beta,
                        timestamp_str=metrics["timestamp"],
                        direction=direction,
                        is_test=False,
                        event_type="EXIT_TRADE",
                        trade_info=exit_trade,
                        portfolio_info=portfolio,
                        open_positions=current_open_positions,
                    )
                    send_alert_email(subject=subject, html_content=html_body, text_content=text_body, dry_run=args.dry_run)
                    alerts_dispatched += 1

            # F. Cooldown & Zero-Crossing Logic (for entry signals)
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

            # G. Execute Paper Trade Entry & Dispatch Report
            if should_alert and not backend.get_open_position(pair_key):
                shares_a, shares_b, cap_a, cap_b, total_cap = calculate_trade_sizing(
                    metrics["latest_price_a"], metrics["latest_price_b"], locked_beta
                )
                if direction == "UPPER":
                    side_a, side_b, trade_dir = "SELL_SHORT", "BUY", "SHORT_SPREAD"
                else:
                    side_a, side_b, trade_dir = "BUY", "SELL_SHORT", "LONG_SPREAD"

                new_pos = {
                    "pair_key": pair_key,
                    "direction": trade_dir,
                    "side_a": side_a,
                    "ticker_a": ticker_a,
                    "shares_a": shares_a,
                    "entry_price_a": metrics["latest_price_a"],
                    "current_price_a": metrics["latest_price_a"],
                    "side_b": side_b,
                    "ticker_b": ticker_b,
                    "shares_b": shares_b,
                    "entry_price_b": metrics["latest_price_b"],
                    "current_price_b": metrics["latest_price_b"],
                    "locked_beta": locked_beta,
                    "entry_z_score": float(current_z),
                    "capital_invested": total_cap,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "opened_at": metrics["timestamp"],
                    "last_updated": metrics["timestamp"],
                }
                backend.save_position(new_pos)

                trade_record = {
                    "trade_id": str(uuid.uuid4()),
                    "pair_key": pair_key,
                    "action": "ENTER_PAIR",
                    "direction": trade_dir,
                    "side_a": side_a,
                    "ticker_a": ticker_a,
                    "shares_a": shares_a,
                    "price_a": metrics["latest_price_a"],
                    "capital_a": cap_a,
                    "side_b": side_b,
                    "ticker_b": ticker_b,
                    "shares_b": shares_b,
                    "price_b": metrics["latest_price_b"],
                    "capital_b": cap_b,
                    "total_capital": total_cap,
                    "z_score_at_trade": float(current_z),
                    "beta_used": locked_beta,
                    "realized_pnl": 0.0,
                    "executed_at": metrics["timestamp"],
                }
                backend.record_trade(trade_record)

                portfolio["cash_balance"] = round(portfolio.get("cash_balance", 100000.0) - total_cap, 2)
                current_open_positions = backend.get_all_open_positions()
                portfolio["open_positions_count"] = len(current_open_positions)
                total_floating = sum(p.get("unrealized_pnl", 0.0) for p in current_open_positions)
                invested_capital = sum(p.get("capital_invested", 0.0) for p in current_open_positions)
                portfolio["total_equity"] = round(portfolio["cash_balance"] + invested_capital + total_floating, 2)
                portfolio["last_updated"] = metrics["timestamp"]
                backend.update_portfolio(portfolio)

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
                    event_type="NEW_TRADE",
                    trade_info=trade_record,
                    portfolio_info=portfolio,
                    open_positions=current_open_positions,
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

    # H. Verification Test Dispatch (if requested and no real trade triggered)
    if test_alert_mode and alerts_dispatched == 0 and evaluated_candidates:
        best = max(evaluated_candidates, key=lambda p: abs(p["current_z"]))
        logger.info(
            f"[TEST ALERT] Initiating simulated paper trade execution verification for pair {best['pair_key']} "
            f"(Z = {best['current_z']:+.2f}σ)..."
        )
        b_metrics = best["metrics"]
        shares_a, shares_b, cap_a, cap_b, total_cap = calculate_trade_sizing(
            b_metrics["latest_price_a"], b_metrics["latest_price_b"], best["locked_beta"]
        )
        t_dir = "SHORT_SPREAD" if best["direction"] == "UPPER" else "LONG_SPREAD"
        s_a = "SELL_SHORT" if best["direction"] == "UPPER" else "BUY"
        s_b = "BUY" if best["direction"] == "UPPER" else "SELL_SHORT"

        test_trade = {
            "trade_id": str(uuid.uuid4()),
            "pair_key": best["pair_key"],
            "action": "ENTER_PAIR",
            "direction": t_dir,
            "side_a": s_a,
            "ticker_a": best["ticker_a"],
            "shares_a": shares_a,
            "price_a": b_metrics["latest_price_a"],
            "capital_a": cap_a,
            "side_b": s_b,
            "ticker_b": best["ticker_b"],
            "shares_b": shares_b,
            "price_b": b_metrics["latest_price_b"],
            "capital_b": cap_b,
            "total_capital": total_cap,
            "z_score_at_trade": float(best["current_z"]),
            "beta_used": best["locked_beta"],
            "realized_pnl": 0.0,
            "executed_at": b_metrics["timestamp"],
        }
        # Persist test trade in Supabase paper_trades ledger
        backend.record_trade(test_trade)

        # Ensure active position is saved if not present
        if not backend.get_open_position(best["pair_key"]):
            test_pos = {
                "pair_key": best["pair_key"],
                "direction": t_dir,
                "side_a": s_a,
                "ticker_a": best["ticker_a"],
                "shares_a": shares_a,
                "entry_price_a": b_metrics["latest_price_a"],
                "current_price_a": b_metrics["latest_price_a"],
                "side_b": s_b,
                "ticker_b": best["ticker_b"],
                "shares_b": shares_b,
                "entry_price_b": b_metrics["latest_price_b"],
                "current_price_b": b_metrics["latest_price_b"],
                "locked_beta": best["locked_beta"],
                "entry_z_score": float(best["current_z"]),
                "capital_invested": total_cap,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "opened_at": b_metrics["timestamp"],
                "last_updated": b_metrics["timestamp"],
            }
            backend.save_position(test_pos)
            portfolio["cash_balance"] = round(portfolio.get("cash_balance", 100000.0) - total_cap, 2)

        current_open_positions = backend.get_all_open_positions()
        portfolio["open_positions_count"] = len(current_open_positions)
        total_floating = sum(p.get("unrealized_pnl", 0.0) for p in current_open_positions)
        invested_capital = sum(p.get("capital_invested", 0.0) for p in current_open_positions)
        portfolio["total_equity"] = round(portfolio["cash_balance"] + invested_capital + total_floating, 2)
        portfolio["last_updated"] = b_metrics["timestamp"]
        backend.update_portfolio(portfolio)

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
            event_type="NEW_TRADE",
            trade_info=test_trade,
            portfolio_info=portfolio,
            open_positions=current_open_positions,
        )
        test_sent = send_alert_email(
            subject=subject,
            html_content=html_body,
            text_content=text_body,
            dry_run=args.dry_run,
        )
        if test_sent:
            logger.info(f"[TEST ALERT] Verification paper trade alert successfully dispatched for {best['pair_key']}.")
        else:
            logger.error(f"[TEST ALERT] Failed to send verification alert email.")

    logger.info("Twin Stock Trading monitoring run completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

