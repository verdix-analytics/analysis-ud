from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from stock_analysis.candlestick_patterns.Doji_Candlestick import (
    DojiCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Hammer_Candlestick import (
    HammerCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Inverted_Hammer_Candlestick import (
    InvertedHammerCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Shooting_Star_Candlestick import (
    ShootingStarCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Spinning_Top_Candlestick import (
    SpinningTopCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Bullish_Engulfing_Candlestick import (
    BullishEngulfingCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Bearish_Engulfing_Candlestick import (
    BearishEngulfingCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Bullish_Harami_Candlestick import (
    BullishHaramiCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Bearish_Harami_Candlestick import (
    BearishHaramiCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Morning_Star_Candlestick import (
    MorningStarCandlestickPattern,
)
from stock_analysis.candlestick_patterns.Evening_Star_Candlestick import (
    EveningStarCandlestickPattern,
)

from stock_analysis.config import SLD_WINDOW_CONFIG
from Utilities.window_registry import registry


# ==========================================
# 1. Load test data
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "TestData" / "stock_1min_OHLC_long.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.sort_values(["Ticker", "Datetime"]).reset_index(drop=True)


# ==========================================
# 2. Instantiate all candlestick patterns
# ==========================================
PATTERNS = [
    DojiCandlestickPattern(),
    HammerCandlestickPattern(),
    InvertedHammerCandlestickPattern(),
    ShootingStarCandlestickPattern(),
    SpinningTopCandlestickPattern(),
    BullishEngulfingCandlestickPattern(),
    BearishEngulfingCandlestickPattern(),
    BullishHaramiCandlestickPattern(),
    BearishHaramiCandlestickPattern(),
    MorningStarCandlestickPattern(),
    EveningStarCandlestickPattern(),
]


# ==========================================
# 3. Sliding-window scan helper
# ==========================================
def scan_candlestick_patterns_for_ticker(
    df_ticker: pd.DataFrame,
    patterns: list,
    window_size: int = 3,
) -> pd.DataFrame:
    """
    Scan one ticker's dataframe with sliding windows and run all candlestick patterns.
    Each detector checks the LAST candle(s) inside the window.
    """
    results = []

    df_ticker = df_ticker.sort_values("Datetime").reset_index(drop=True).copy()

    if len(df_ticker) < window_size:
        return pd.DataFrame()

    for end_idx in range(window_size, len(df_ticker) + 1):
        start_idx = end_idx - window_size
        window_df = df_ticker.iloc[start_idx:end_idx].reset_index(drop=True)

        event_time = window_df.iloc[-1]["Datetime"]
        ticker = window_df.iloc[-1]["Ticker"]

        for pattern in patterns:
            result = pattern.execute(window_df)

            if result.confirmed:
                results.append(
                    {
                        "Ticker": ticker,
                        "Datetime": event_time,
                        "Pattern": pattern.__class__.__name__,
                        "Category": pattern.category,
                        "Confidence": result.confidence,
                        "WindowStartIdx": start_idx,
                        "WindowEndIdx": end_idx - 1,
                    }
                )

    if not results:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Datetime",
                "Pattern",
                "Category",
                "Confidence",
                "WindowStartIdx",
                "WindowEndIdx",
            ]
        )

    return pd.DataFrame(results)


# ==========================================
# 4. Full scan over all tickers
# ==========================================
def scan_all_candlestick_patterns(df_all: pd.DataFrame, patterns: list) -> pd.DataFrame:
    all_results = []

    window_size = SLD_WINDOW_CONFIG["candlestick_patterns"]["window_size"]

    for ticker in df_all["Ticker"].unique():
        df_ticker = df_all[df_all["Ticker"] == ticker].copy()
        results_ticker = scan_candlestick_patterns_for_ticker(
            df_ticker=df_ticker,
            patterns=patterns,
            window_size=window_size,
        )
        all_results.append(results_ticker)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# ==========================================
# 5. Smoke test: pattern interface works
# ==========================================
def test_pattern_execute_smoke():
    """
    Basic smoke test:
    - every detector can run on a 3-candle window
    - returned result has confirmed/confidence
    - confidence is within [0, 100]
    """
    sample_ticker = df["Ticker"].iloc[0]
    sample_df = (
        df[df["Ticker"] == sample_ticker]
        .sort_values("Datetime")
        .head(3)
        .reset_index(drop=True)
    )

    for pattern in PATTERNS:
        result = pattern.execute(sample_df)

        assert hasattr(result, "confirmed")
        assert hasattr(result, "confidence")
        assert isinstance(result.confirmed, bool)
        assert 0 <= result.confidence <= 100


# ==========================================
# 6. Smoke test: full scan runs
# ==========================================
def test_scan_all_candlestick_patterns_runs():
    results = scan_all_candlestick_patterns(df, PATTERNS)

    assert isinstance(results, pd.DataFrame)

    expected_cols = {
        "Ticker",
        "Datetime",
        "Pattern",
        "Category",
        "Confidence",
        "WindowStartIdx",
        "WindowEndIdx",
    }
    assert expected_cols.issubset(set(results.columns))


def test_new_pattern_examples_confirm():
    bullish_harami_df = pd.DataFrame(
        [
            {"Open": 10.0, "High": 10.2, "Low": 8.6, "Close": 8.8},
            {"Open": 9.1, "High": 9.6, "Low": 9.0, "Close": 9.5},
        ]
    )
    assert BullishHaramiCandlestickPattern().execute(bullish_harami_df).confirmed

    bearish_harami_df = pd.DataFrame(
        [
            {"Open": 8.8, "High": 10.2, "Low": 8.7, "Close": 10.0},
            {"Open": 9.7, "High": 9.8, "Low": 9.1, "Close": 9.2},
        ]
    )
    assert BearishHaramiCandlestickPattern().execute(bearish_harami_df).confirmed

    assert registry.get("InvertedHammerCandlestickPattern") == 1

    inverted_hammer_df = pd.DataFrame(
        [
            {"Open": 10.2, "High": 10.8, "Low": 10.18, "Close": 10.3},
        ]
    )
    assert InvertedHammerCandlestickPattern().execute(inverted_hammer_df).confirmed

    assert registry.get("SpinningTopCandlestickPattern") == 1

    spinning_top_df = pd.DataFrame(
        [
            {"Open": 10.0, "High": 10.7, "Low": 9.3, "Close": 10.12},
        ]
    )
    assert SpinningTopCandlestickPattern().execute(spinning_top_df).confirmed


# ==========================================
# 7. Optional: run as script for manual checking
# ==========================================
if __name__ == "__main__":
    print("Loaded data shape:", df.shape)
    print("Tickers:", df["Ticker"].unique().tolist())

    results = scan_all_candlestick_patterns(df, PATTERNS)

    print("\nTotal detected candlestick patterns:", len(results))

    if results.empty:
        print("No candlestick patterns detected.")
    else:
        print("\nPattern counts:")
        print(results["Pattern"].value_counts())

        print("\nCounts by ticker:")
        print(results.groupby(["Ticker", "Pattern"]).size())

        print("\nSample detections:")
        print(results.head(20))

        out_path = REPO_ROOT / "TestData" / "candlestick_test_results.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSaved results to: {out_path}")
