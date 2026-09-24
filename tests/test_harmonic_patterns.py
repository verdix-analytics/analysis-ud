from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from stock_analysis.harmonic_patterns.Gartley_Harmonic import (
    GartleyHarmonicPattern,
)
from stock_analysis.harmonic_patterns.Bat_Harmonic import (
    BatHarmonicPattern,
)
from stock_analysis.harmonic_patterns.Butterfly_Harmonic import (
    ButterflyHarmonicPattern,
)
from stock_analysis.harmonic_patterns.Crab_Harmonic import (
    CrabHarmonicPattern,
)

from stock_analysis.config import SLD_WINDOW_CONFIG


# 1. Load test data
DATA_PATH = REPO_ROOT / "TestData" / "stock_1min_OHLC_long.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.sort_values(["Ticker", "Datetime"]).reset_index(drop=True)


# 2. Instantiate all harmonic patterns
PATTERNS = [
    GartleyHarmonicPattern(),
    BatHarmonicPattern(),
    ButterflyHarmonicPattern(),
    CrabHarmonicPattern(),
]


# 3. Sliding-window scan helper
def scan_harmonic_patterns_for_ticker(
    df_ticker: pd.DataFrame,
    patterns: list,
    window_size: int = 25,
) -> pd.DataFrame:
    """
    Scan one ticker's dataframe with sliding windows and run all harmonic detectors.
    Each detector checks whether the WHOLE window forms the target harmonic pattern.
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

# 4. Full scan over all tickers
def scan_all_harmonic_patterns(df_all: pd.DataFrame, patterns: list) -> pd.DataFrame:
    all_results = []

    window_size = SLD_WINDOW_CONFIG["harmonic_patterns"]["window_size"]

    for ticker in df_all["Ticker"].unique():
        df_ticker = df_all[df_all["Ticker"] == ticker].copy()
        results_ticker = scan_harmonic_patterns_for_ticker(
            df_ticker=df_ticker,
            patterns=patterns,
            window_size=window_size,
        )
        all_results.append(results_ticker)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# 5. Smoke test: pattern interface works
def test_harmonic_pattern_execute_smoke():
    """
    Basic smoke test:
    - every harmonic detector can run on a harmonic window
    - returned result has confirmed/confidence
    - confidence is within [0, 100]
    """
    sample_ticker = df["Ticker"].iloc[0]
    window_size = SLD_WINDOW_CONFIG["harmonic_patterns"]["window_size"]

    sample_df = (
        df[df["Ticker"] == sample_ticker]
        .sort_values("Datetime")
        .head(window_size)
        .reset_index(drop=True)
    )

    for pattern in PATTERNS:
        result = pattern.execute(sample_df)

        assert hasattr(result, "confirmed")
        assert hasattr(result, "confidence")
        assert isinstance(result.confirmed, bool)
        assert 0 <= result.confidence <= 100


# 6. Smoke test: full scan runs
def test_scan_all_harmonic_patterns_runs():
    results = scan_all_harmonic_patterns(df, PATTERNS)

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


# 7. Optional: run as script for manual checking
if __name__ == "__main__":
    print("Loaded data shape:", df.shape)
    print("Tickers:", df["Ticker"].unique().tolist())

    print("\nHarmonic patterns being tested:")
    for p in PATTERNS:
        print("-", p.__class__.__name__)

    results = scan_all_harmonic_patterns(df, PATTERNS)

    print("\nTotal detected harmonic patterns:", len(results))

    if results.empty:
        print("No harmonic patterns detected.")
    else:
        print("\nPattern counts:")
        print(results["Pattern"].value_counts())

        print("\nCounts by ticker:")
        print(results.groupby(["Ticker", "Pattern"]).size())

        print("\nSample detections:")
        print(results.head(20))

        out_path = REPO_ROOT / "TestData" / "harmonic_test_results.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSaved results to: {out_path}")

    print("\n✅ HARMONIC PATTERN TEST DONE")