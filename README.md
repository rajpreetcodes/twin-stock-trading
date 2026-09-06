# Twin Stock Trading: Intraday Pair Monitoring System

An automated Python-based statistical deviation monitoring system for stock pairs. It caches 5-minute intraday candle data in **Supabase**, locks an OLS Beta daily from 90-day daily closing prices, calculates rolling spread Z-scores, checks for sustained deviations, enforces cooldown and sign-inversion zero-crossing rules, and sends rich alert emails via the **Resend API**. Runs every 5 minutes during market hours via **GitHub Actions**.

---

## Key Refinements & Architecture

1. **Cloud State & Candle Cache via Supabase**:
   - Replaces fragile git-commit-push loops with a robust Supabase backend.
   - `pair_state`: Persists ticker pair, locked daily beta, `alert_sent_today` flag, cooldown status, and last Z-score.
   - `candle_cache`: Stores rolling 5-minute OHLCV candles per ticker to avoid repeated downloads. Automatically prunes candles older than 7 trading days.
   - Offline/Memory fallback included: dry-runs and unit tests run seamlessly even without Supabase credentials configured.

2. **Daily Beta Lock (90-Day Daily OLS)**:
   - Queries daily closing prices for the past 90 days once per calendar day to compute the OLS hedge ratio (Beta):
     $$\beta = \frac{\text{Cov}(P_{A, \text{daily}}, P_{B, \text{daily}})}{\text{Var}(P_{B, \text{daily}})}$$
   - Locks this Beta in Supabase for all intraday 5-minute spread calculations for that trading day:
     $$\text{Spread} = P_A - (\text{Locked\_Beta} \cdot P_B)$$

3. **Incremental 5-Minute Candle Ingestion**:
   - On each run, fetches only the current day (`period="1d", interval="5m"`) via `yfinance` using a custom browser User-Agent and a 1-second delay between ticker calls to prevent IP rate limits.
   - Appends new candles to `candle_cache` in Supabase and runs rolling metrics on the accumulated 7-day dataset.

4. **Sign-Inversion & Mean-Reversion Zero-Crossing**:
   - Re-arms the alert trigger if $|Z| \le 0.25$ OR if a sign inversion occurs ($Z_{\text{prev}} \cdot Z_{\text{curr}} \le 0$).
   - This prevents rapid price swings that jump across zero from skipping the cooldown reset.

5. **Updated Scheduling & Timezone Resilience**:
   - Workflow cron schedule: `*/5 13-22 * * 1-5` (UTC). Handles Eastern Daylight Time (EDT) and Eastern Standard Time (EST) shifts seamlessly.
   - `is_market_open()` in `monitor.py` verifies the exact 09:30 to 16:00 `America/New_York` window and exits cleanly if closed.

---

## File Structure

```
Trading_Strategy/
├── .github/
│   └── workflows/
│       └── alert.yml         # GitHub Actions 5m cron workflow (13-22 UTC)
├── monitor.py                # Main monitoring orchestrator & statistical engine
├── supabase_schema.sql       # SQL script to initialize Supabase tables
├── requirements.txt          # Python dependencies (supabase, resend, yfinance, etc.)
├── test_monitor.py           # Pytest test suite covering all logic
├── .env.example              # Example environment variables
└── README.md                 # System documentation
```

---

## Database Setup (Supabase)

Run the script in [supabase_schema.sql](supabase_schema.sql) in your Supabase project's **SQL Editor**. This creates:
- `pair_state` (tracks daily locked Beta and alert status per pair)
- `candle_cache` (stores 5m OHLCV candles with composite primary key `(ticker, timestamp)`)

---

## Monitored Pairs: The "True Twins"

The system focuses exclusively on genuine corporate "twins" with high cointegration, shared macroeconomic sensitivities, and identical input costs, preventing structural drift:

| Pair | Companies | Fundamental Parity & Cointegration Logic |
|------|-----------|------------------------------------------|
| **`KO : PEP`** | **Coca-Cola vs. PepsiCo** | The gold standard of pairs trading. Identical input costs (aluminum, packaging, corn syrup) and consumer defensive sector demand. |
| **`V : MA`** | **Visa vs. Mastercard** | Pure tollbooth transaction models with zero balance-sheet credit risk, ~55% operating margins, and identical global payment rail exposure. |
| **`HD : LOW`** | **Home Depot vs. Lowe's** | Home improvement retail duopoly tied directly to US housing turnover, mortgage rates, and lumber prices. |
| **`XOM : CVX`** | **ExxonMobil vs. Chevron** | Global energy titans driven tick-for-tick by crude oil benchmarks (WTI/Brent), crack spreads, and OPEC+ policy. |

---

## Local Installation & Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Add your `SUPABASE_URL`, `SUPABASE_KEY`, `RESEND_API_KEY`, and `ALERT_EMAIL_TO`.

3. **Run unit tests:**
   ```bash
   pytest -v test_monitor.py
   ```

4. **Execute dry run:**
   ```bash
   # Bypasses market hours check, uses in-memory/Supabase cache, and suppresses live emails
   python monitor.py --force --dry-run --pairs KO:PEP
   ```

---

## GitHub Actions Deployment

1. Add the following repository secrets (**Settings > Secrets and variables > Actions > Secrets**):
   - `SUPABASE_URL`: Your Supabase Project URL
   - `SUPABASE_KEY`: Your Supabase Anon or Service Role key
   - `RESEND_API_KEY`: Your Resend API Key
   - `ALERT_EMAIL_TO`: Recipient email address
   - `ALERT_EMAIL_FROM`: (Optional) Sender address
2. No write permissions required: the workflow runs strictly as a reader/notifier without git write permissions.
