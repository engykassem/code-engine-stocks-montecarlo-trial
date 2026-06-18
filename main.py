"""
IBM Cloud Code Engine Demo – Monte Carlo Stock Simulator
=========================================================
Each parallel job worker picks one stock from the STOCKS env var,
runs a Monte Carlo simulation using 10 years of historical data,
and sends the results via SMS (Twilio).

Environment variables:
 STOCKS            – comma-separated tickers, e.g. "AAPL,MSFT,GOOGL,AMZN"
 INVEST_AMOUNT     – dollar amount to simulate (capped at $10,000)
 TWILIO_TOKEN
 PHONE_NUMBER
 JOB_INDEX         – set automatically by Code Engine (0-based)
"""

import os
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from twilio.rest import Client


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000   # number of Monte Carlo paths
FORECAST_DAYS = 252        # 1 trading year (~252 days)
HISTORY_YEARS = 10         # how far back to pull historical data
MAX_INVEST = 10_000        # hard cap on investment amount


def fetch_historical_returns(ticker: str) -> np.ndarray:
"""Download 10 years of adjusted-close prices and return daily log returns."""
end = datetime.now()
start = end - timedelta(days=HISTORY_YEARS * 365)

df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

if df.empty or len(df) < 252:
raise ValueError(f"Not enough historical data for {ticker}")

# Daily log returns
closes = df["Close"].values.flatten()
log_returns = np.diff(np.log(closes))
return log_returns


def run_monte_carlo(log_returns: np.ndarray, invest_amount: float) -> dict:
"""
   Run Monte Carlo simulation using Geometric Brownian Motion (GBM).

   Each simulated day:
       price *= exp(drift + volatility * random_normal)

   where:
       drift     = mean(daily log returns) - 0.5 * variance   (risk-adjusted)
       volatility = std(daily log returns)
   """
mu = np.mean(log_returns)
sigma = np.std(log_returns)
drift = mu - 0.5 * sigma ** 2

# Simulate: each row is one path of FORECAST_DAYS daily returns
random_shocks = np.random.normal(size=(NUM_SIMULATIONS, FORECAST_DAYS))
daily_returns = drift + sigma * random_shocks          # (sims × days)
cumulative = np.cumsum(daily_returns, axis=1)          # cumulative log return
price_paths = invest_amount * np.exp(cumulative)       # simulated portfolio values

final_values = price_paths[:, -1]

# Worst case: most probable negative return (mode of the loss bucket)
# If all outcomes are positive, use the lowest return instead
losing_sims = final_values[final_values < invest_amount]
if len(losing_sims) > 0:
# Bin negative outcomes and find the peak (most likely loss)
hist_counts, bin_edges = np.histogram(losing_sims, bins=50)
peak_bin = np.argmax(hist_counts)
worst_likely = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
prob_worst = float(hist_counts[peak_bin] / NUM_SIMULATIONS * 100)
else:
# All positive — just show the lowest return
worst_likely = float(np.min(final_values))
prob_worst = float(1 / NUM_SIMULATIONS * 100)

prob_loss = float(np.mean(final_values < invest_amount) * 100)

return {
"mean_final":    float(np.mean(final_values)),
"prob_profit":   float(100 - prob_loss),
"worst_likely":  float(worst_likely),
"prob_worst":    prob_worst,
"prob_loss":     prob_loss,
}


def format_sms(ticker: str, amount: float, results: dict) -> str:
"""Build a concise SMS body with the simulation results."""
mean_val = results["mean_final"]
gain_pct = ((mean_val - amount) / amount) * 100

worst_val = results["worst_likely"]
worst_pct = ((worst_val - amount) / amount) * 100

prob = results["prob_profit"]

if prob >= 75:
buy_line = f"Buy or not: Yes - {prob:.0f}% of simulations were profitable with an avg return of {gain_pct:+.1f}%. The odds are strongly in your favour."
elif prob >= 50:
buy_line = f"Buy or not: Yes - more likely to profit ({prob:.0f}%) than not, but {100 - prob:.0f}% of simulations lost money. Don't bet the house."
elif prob >= 35:
buy_line = f"Buy or not: No - only {prob:.0f}% of simulations were profitable. You're basically flipping a weighted coin against yourself."
else:
buy_line = f"Buy or not: No - only {prob:.0f}% of simulations made money. This stock historically moves like a rollercoaster built by an intern."

return (
f"{ticker} - Monte Carlo Results\n"
        f"----\n"
f"Expected value: ${mean_val:,.0f} ({gain_pct:+.1f}%)\n"
f"Prob of profit: {prob:.1f}%\n"
        f"----\n"
f"Worst case scenario: ${worst_val:,.0f} ({worst_pct:+.1f}%)\n"
f"Prob of worst: {results['prob_worst']:.1f}%\n"
        f"----\n"
f"{buy_line}\n"
f"\n"
f"Invested: ${amount:,.0f} | Sims: {NUM_SIMULATIONS:,}\n"
f"\n"
        f"This simulation cannot be held accountable for "
        f"Disclaimer: This simulation cannot be held accountable for "
f"any losses from its suggestions, but any profits "
f"must be shared 50/50."
)


def main():
# ── Read env vars ────────────────────────────────────────────────
job_index = int(os.getenv("JOB_INDEX", "0"))

stocks_raw = os.environ.get("STOCKS", "AAPL,MSFT,GOOGL")
tickers = [s.strip().upper() for s in stocks_raw.split(",") if s.strip()]

if len(tickers) > 7:
tickers = tickers[:7]
print("Warning: capped to 7 stocks max")

if job_index >= len(tickers):
print(f"Worker {job_index} has no stock assigned (only {len(tickers)} tickers). Exiting.")
return

ticker = tickers[job_index]

invest_amount = min(float(os.environ.get("INVEST_AMOUNT", "1000")), MAX_INVEST)

# ── Twilio setup ─────────────────────────────────────────────────
account_sid  = "AC79b7a71528d09a249f521d2de052d30"
auth_token   = os.environ["TWILIO_TOKEN"]
from_number  = "+19843638872"
to_number    = os.environ["PHONE_NUMBER"]
twilio_client = Client(account_sid, auth_token)

# ── Run simulation ───────────────────────────────────────────────
print(f"Worker {job_index}: running Monte Carlo for {ticker} "
f"(${invest_amount:,.0f}, {NUM_SIMULATIONS:,} sims)...")

log_returns = fetch_historical_returns(ticker)
results = run_monte_carlo(log_returns, invest_amount)

# ── Send SMS ─────────────────────────────────────────────────────
body = format_sms(ticker, invest_amount, results)
message = twilio_client.messages.create(
body=body,
from_=from_number,
to=to_number,
)

print(f"Worker {job_index} ({ticker}): SMS sent, SID={message.sid}")
print(body)


if __name__ == "__main__":
main()
