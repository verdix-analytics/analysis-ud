import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path.cwd().resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("REPO_ROOT =", REPO_ROOT)

from kafka.producer2 import fetch_harmonic_data

import pkgutil
import importlib
import inspect

from stock_analysis.base import BasePattern
import stock_analysis.harmonic_patterns as harmonic_pkg


def load_harmonic_detectors():
    detectors = []

    for _, module_name, _ in pkgutil.iter_modules(harmonic_pkg.__path__):
        full_module_name = f"{harmonic_pkg.__name__}.{module_name}"
        module = importlib.import_module(full_module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePattern) and obj is not BasePattern:
                if getattr(obj, "enabled", False):
                    try:
                        detectors.append(obj())
                    except Exception as e:
                        print(f"Skip detector {obj.__name__}: {e}")

    return detectors


harmonic_detectors = load_harmonic_detectors()
print(f"Loaded {len(harmonic_detectors)} chart detectors:")
for d in harmonic_detectors:
    print("-", d.__class__.__name__)

def run_detector(detector, df_window: pd.DataFrame) -> dict | None:
    """
    Run one chart detector safely.
    Returns the detector result dict or None if failed.
    """
    try:
        if hasattr(detector, "_run"):
            result = detector._run(df_window)
        elif hasattr(detector, "run"):
            result = detector.run(df_window)
        else:
            return None

        if not isinstance(result, dict):
            return None

        return result

    except Exception as e:
        # first version: just skip failures
        return None
    
from stock_analysis.trendanalysis import TrendAnalysis

trend_analyzer = TrendAnalysis()

def project_pretrend_analysis(df_window: pd.DataFrame) -> dict:
    trend = trend_analyzer.pre_trend_analysis(df_window)
    if trend is None:
        return {
            "label": "unknown",
            "strength": 0.0,
            "higher_high_ratio": 0.0,
            "higher_low_ratio": 0.0,
            "lower_high_ratio": 0.0,
            "lower_low_ratio": 0.0,
            "candles": len(df_window),
        }
    return trend

from stock_analysis.config import SLD_WINDOW_CONFIG

def is_valid_harmonic_signal(result: dict, min_confidence: float = 50) -> bool:
    if not result:
        return False

    if result.get("confirmed", False):
        return True

    if result.get("structure_detected", False) and result.get("confidence", 0) >= min_confidence:
        return True

    return False


HARMONIC_PATTERN_WINDOW_SIZE = {
    "BatHarmonicPattern": 60,
    "ButterflyHarmonicPattern": 70,
    "CrabHarmonicPattern": 80,
    "CypherHarmonicPattern": 55,
    "GartleyHarmonicPattern": 65,
    "SharkHarmonicPattern": 50,
}

def collect_harmonic_formations_at_t(
    df_hist: pd.DataFrame,
    detectors: list,
    recent_windows_per_pattern: int = 3,
    verbose: bool = False,
) -> list[dict]:
    formations = []

    chart_cfg = SLD_WINDOW_CONFIG["harmonic_patterns"]
    default_window_size = chart_cfg["window_size"]
    min_candles = chart_cfg["min_candles"]

    for detector in detectors:
        pattern_name = detector.__class__.__name__

        window_size = HARMONIC_PATTERN_WINDOW_SIZE.get(
            pattern_name,
            default_window_size
        )

        if len(df_hist) < max(window_size, min_candles):
            continue

        for window_count in range(1, recent_windows_per_pattern + 1):
            end_idx = len(df_hist) - (window_count - 1)
            start_idx = end_idx - window_size

            if start_idx < 0:
                continue

            df_window = df_hist.iloc[start_idx:end_idx].reset_index(drop=True)

            if len(df_window) < min_candles:
                continue

            result = run_detector(detector, df_window)

            if not is_valid_harmonic_signal(result, min_confidence=50):
                continue

            formation = {
                "pattern": pattern_name,
                "confidence": result.get("confidence", 0),
                "window_count": window_count,
                "window_size": window_size,
                "pre_trend": project_pretrend_analysis(df_window),
                "signal": {
                    "direction": result.get("direction"),
                    "pivots": result.get("pivots", []),
                },
                "raw_result": result,
            }

            formations.append(formation)

            if verbose:
                print(
                    f"[FOUND] {pattern_name} | "
                    f"window_size={window_size} | "
                    f"window_count={window_count} | "
                    f"confidence={formation['confidence']} | "
                    f"direction={formation['signal']['direction']}"
                )

    return formations


from stock_analysis.context_scores.harmonic_score import harmonic_score

def backtest_harmonic_context_score(
    df: pd.DataFrame,
    detectors: list,
    min_history: int = 120,
    future_horizon: int = 30,
    recent_windows_per_pattern: int = 3,
    return_lookback: int = 30,
) -> pd.DataFrame:
    """
    Prototype backtest for chart context score.

    At each time t:
    - use df[:t+1] as historical data
    - collect chart formations
    - compute context score
    - compare with:
        1. recent_return: past return from t-return_lookback to t
        2. future_return: future return from t to t+future_horizon
    """
    records = []

    start_t = max(min_history, return_lookback)

    for t in range(start_t, len(df) - future_horizon):
        df_hist = df.iloc[:t+1].copy()

        past_row = df.iloc[t - return_lookback]
        current_row = df.iloc[t]
        future_row = df.iloc[t + future_horizon]

        formations = collect_harmonic_formations_at_t(
            df_hist=df_hist,
            detectors=detectors,
            recent_windows_per_pattern=recent_windows_per_pattern,
        )

        ctx = harmonic_score(formations)

        past_close = past_row["close"]
        current_close = current_row["close"]
        future_close = future_row["close"]

        recent_return = (current_close - past_close) / max(abs(past_close), 1e-9)
        future_return = (future_close - current_close) / max(abs(current_close), 1e-9)

        bias = ctx["bias"]
        score = ctx["score"]
        agreement = ctx["agreement"]

        if bias == "bullish":
            recent_correct = int(recent_return > 0)
            future_correct = int(future_return > 0)
        elif bias == "bearish":
            recent_correct = int(recent_return < 0)
            future_correct = int(future_return < 0)
        else:
            recent_correct = np.nan
            future_correct = np.nan

        records.append({
            "t": t,
            "Datetime": current_row["Datetime"],

            "score": score,
            "bias": bias,
            "agreement": agreement,
            "bull_score": ctx["bull_score"],
            "bear_score": ctx["bear_score"],
            "pattern_count": ctx["pattern_count"],

            "recent_return": recent_return,
            "future_return": future_return,

            "past_close": past_close,
            "current_close": current_close,
            "future_close": future_close,

            "recent_correct": recent_correct,
            "future_correct": future_correct,

            "formations": formations,
            "context_detail": ctx,
        })

    return pd.DataFrame(records)


import os

# define your 50 tickers
TICKERS = [
    "AAPL", "NVDA", "TSLA", "MMM", "AOS", "AFL",
    "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC",
    "GS", "MS", "WFC", "JNJ", "PFE", "MRK", "UNH"
]

# --- fetch and save all tickers ---
os.makedirs("test_data", exist_ok=True)

for ticker in TICKERS:
    path = f"test_data/{ticker.lower()}_1y.csv"
    if os.path.exists(path):
        print(f"[SKIP FETCH] {ticker} already exists")
        continue
    try:
        df = fetch_harmonic_data(ticker)
        df.to_csv(path)
        print(f"[SAVED] {ticker}")
    except Exception as e:
        print(f"[FAIL] {ticker}: {e}")

# --- build CSV_FILES dynamically ---
CSV_FILES = {
    ticker: f"test_data/{ticker.lower()}_1y.csv"
    for ticker in TICKERS
    if os.path.exists(f"test_data/{ticker.lower()}_1y.csv")
}

print(f"\nRunning backtest on {len(CSV_FILES)} tickers...")

all_results = []

for ticker, path in CSV_FILES.items():
    result_path = f"test_data/results_{ticker.lower()}.csv"

    # skip if already computed
    if os.path.exists(result_path):
        print(f"[LOAD] {ticker} results from cache")
        results = pd.read_csv(result_path)
        results["ticker"] = ticker
        all_results.append(results)
        continue

    print(f"\n{'='*40}")
    print(f"Running backtest for {ticker}...")

    df = pd.read_csv(path)

    results = backtest_harmonic_context_score(
        df=df,
        detectors=harmonic_detectors,
        min_history=60,
        future_horizon=30,
        recent_windows_per_pattern=4,
        return_lookback=60,
    )

    results["ticker"] = ticker

    # save only serialisable columns
    save_cols = [
        "t", "Datetime", "ticker", "score", "bias", "agreement",
        "bull_score", "bear_score", "pattern_count",
        "recent_return", "future_return",
        "past_close", "current_close", "future_close",
        "recent_correct", "future_correct",
    ]
    results[save_cols].to_csv(result_path, index=False)
    all_results.append(results)

    print(f"[{ticker}] correlation:")
    print(results[["score", "recent_return", "future_return"]].corr())
    
if not all_results:
    print("No results to combine.")
    sys.exit(0)

combined = pd.concat(all_results, ignore_index=True)

print(f"\n{'='*40}")
print("COMBINED (all tickers) correlation:")
print(f"{'='*40}")
print(combined[["score", "recent_return", "future_return"]].corr())

# neutral band filter (score near 0.5 = no conviction)
filtered_combined = combined[abs(combined["score"] - 0.5) > 0.05].copy()
print(f"\nCOMBINED filtered (|score-0.5| > 0.05) correlation:")
print(filtered_combined[["score", "recent_return", "future_return"]].corr())

# ── Per-ticker summary ────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print("PER-TICKER SUMMARY")
print(f"{'='*40}")

summary = (
    combined.groupby("ticker")
    .agg(
        rows=("score", "count"),
        score_mean=("score", "mean"),
        bias_bullish=("bias", lambda x: (x == "bullish").mean()),
        bias_bearish=("bias", lambda x: (x == "bearish").mean()),
        recent_return_mean=("recent_return", "mean"),
        recent_correct_mean=("recent_correct", "mean"),
        future_return_mean=("future_return", "mean"),
        future_correct_mean=("future_correct", "mean"),
    )
    .round(4)
)
print(summary)
summary.to_csv("cs_stock_summary.csv")

# ── Per-ticker correlation chart ──────────────────────────────────────────────
corr_list = []
sum_corres,l = 0,0
for ticker in combined["ticker"].unique():
    subset = combined[combined["ticker"] == ticker]
    corrs  = subset[["score", "recent_return", "future_return"]].corr()
    sum_corres += corrs.loc["recent_return"]
    l += 1
    corr_list.append({
        "Ticker":           ticker,
        "Recent_Return_Corr": corrs.loc["score", "recent_return"],
        "Future_Return_Corr": corrs.loc["score", "future_return"],
    })

corr_df   = pd.DataFrame(corr_list)
plot_data = corr_df.melt(id_vars="Ticker", var_name="Metric", value_name="Correlation")
avg_recent_corr = corr_df["Recent_Return_Corr"].mean()

print(f"Average Correlation (Score vs Recent Return): {avg_recent_corr:.4f}")

plt.figure(figsize=(14, 6))
sns.barplot(data=plot_data, x="Ticker", y="Correlation", hue="Metric", palette="magma")
plt.title("Harmonic Score — Correlation vs Returns (per ticker)", fontsize=14)
plt.ylabel("Correlation Coefficient (r)")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("hs_ticker_correlation_chart.png", dpi=150)
print("\n[SAVED] hs_ticker_correlation_chart.png")

corr_df.to_csv("hs_stock_correlation_summary.csv", index=False)
print("[SAVED] hs_stock_correlation_summary.csv")
print("[SAVED] hs_stock_summary.csv")