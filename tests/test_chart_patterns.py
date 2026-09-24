from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from stock_analysis.chart_patterns.Double_Top_Chart import (
    DoubleTopChartPattern,
)
from stock_analysis.chart_patterns.Double_Bottom_Chart import (
    DoubleBottomChartPattern,
)
from stock_analysis.chart_patterns.Head_and_Shoulders_Chart import (
    HeadAndShouldersChartPattern,
)
from stock_analysis.chart_patterns.Inverse_Head_and_Shoulders_Chart import (
    InverseHeadAndShouldersChartPattern,
)
from stock_analysis.chart_patterns.Triple_Top_Chart import (
    TripleTopChartPattern,
)
from stock_analysis.chart_patterns.Triple_Bottom_Chart import (
    TripleBottomChartPattern,
)
from stock_analysis.chart_patterns.Ascending_Triangle_Chart import (
    AscendingTriangleChartPattern,
)
from stock_analysis.chart_patterns.Descending_Triangle_Chart import (
    DescendingTriangleChartPattern,
)
from stock_analysis.chart_patterns.Symmetrical_Triangle_Chart import (
    SymmetricalTriangleChartPattern,
)
from stock_analysis.chart_patterns.Flag_Chart import (
    FlagChartPattern,
)
from stock_analysis.chart_patterns.Pennant_Chart import (
    PennantChartPattern,
)
from stock_analysis.chart_patterns.Cup_and_Handle_Chart import (
    CupAndHandleChartPattern,
)
from stock_analysis.chart_patterns.Ascending_Channel_Chart import (
    AscendingChannelChartPattern,
)
from stock_analysis.chart_patterns.Descending_Channel_Chart import (
    DescendingChannelChartPattern,
)
from stock_analysis.chart_patterns.Diamond_Top_Chart import (
    DiamondTopChartPattern,
)
from stock_analysis.chart_patterns.Diamond_Bottom_Chart import (
    DiamondBottomChartPattern,
)
from stock_analysis.chart_patterns.Megaphone_Chart import (
    MegaphoneChartPattern,
)
from stock_analysis.chart_patterns.Rectangle_Chart import (
    RectangleChartPattern,
)
from stock_analysis.chart_patterns.Rising_Wedge_Chart import (
    RisingWedgeChartPattern,
)
from stock_analysis.chart_patterns.Falling_Wedge_Chart import (
    FallingWedgeChartPattern,
)

from stock_analysis.config import SLD_WINDOW_CONFIG


# 1. Load test data
DATA_PATH = REPO_ROOT / "TestData" / "stock_1min_OHLC_long.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.sort_values(["Ticker", "Datetime"]).reset_index(drop=True)


# 2. Instantiate all chart patterns
PATTERNS = [
    DoubleTopChartPattern(),
    DoubleBottomChartPattern(),
    HeadAndShouldersChartPattern(),
    InverseHeadAndShouldersChartPattern(),
    TripleTopChartPattern(),
    TripleBottomChartPattern(),
    AscendingTriangleChartPattern(),
    DescendingTriangleChartPattern(),
    SymmetricalTriangleChartPattern(),
    FlagChartPattern(),
    PennantChartPattern(),
    CupAndHandleChartPattern(),
    AscendingChannelChartPattern(),
    DescendingChannelChartPattern(),
    DiamondTopChartPattern(),
    DiamondBottomChartPattern(),
    MegaphoneChartPattern(),
    RectangleChartPattern(),
    RisingWedgeChartPattern(),
    FallingWedgeChartPattern(),
]


# 3. Sliding-window scan helper
def scan_chart_patterns_for_ticker(
    df_ticker: pd.DataFrame,
    patterns: list,
    window_size: int = 30,
) -> pd.DataFrame:
    """
    Scan one ticker's dataframe with sliding windows and run all chart pattern detectors.
    Each detector checks whether the WHOLE window forms the target chart pattern.
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
def scan_all_chart_patterns(df_all: pd.DataFrame, patterns: list) -> pd.DataFrame:
    all_results = []

    window_size = SLD_WINDOW_CONFIG["chart_patterns"]["window_size"]

    for ticker in df_all["Ticker"].unique():
        df_ticker = df_all[df_all["Ticker"] == ticker].copy()
        results_ticker = scan_chart_patterns_for_ticker(
            df_ticker=df_ticker,
            patterns=patterns,
            window_size=window_size,
        )
        all_results.append(results_ticker)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# 5. Smoke test: pattern interface works
def test_chart_pattern_execute_smoke():
    """
    Basic smoke test:
    - every chart detector can run on a chart window
    - returned result has confirmed/structure_detected/confidence
    - confidence is within [0, 100]
    """
    sample_ticker = df["Ticker"].iloc[0]
    window_size = SLD_WINDOW_CONFIG["chart_patterns"]["window_size"]

    sample_df = (
        df[df["Ticker"] == sample_ticker]
        .sort_values("Datetime")
        .head(window_size)
        .reset_index(drop=True)
    )

    for pattern in PATTERNS:
        result = pattern.execute(sample_df)

        assert hasattr(result, "confirmed")
        assert hasattr(result, "structure_detected")
        assert hasattr(result, "confidence")
        assert isinstance(result.confirmed, bool)
        assert isinstance(result.structure_detected, bool)
        assert 0 <= result.confidence <= 100


# 6. Smoke test: full scan runs
def test_scan_all_chart_patterns_runs():
    window_size = SLD_WINDOW_CONFIG["chart_patterns"]["window_size"]
    sample_ticker = df["Ticker"].iloc[0]
    df_subset = (
        df[df["Ticker"] == sample_ticker]
        .sort_values("Datetime")
        .head(window_size * 4)
        .reset_index(drop=True)
    )

    representative_patterns = PATTERNS[: min(3, len(PATTERNS))]
    results = scan_all_chart_patterns(df_subset, representative_patterns)

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

    print("\nChart patterns being tested:")
    for p in PATTERNS:
        print("-", p.__class__.__name__)

    results = scan_all_chart_patterns(df, PATTERNS)

    print("\nTotal detected chart patterns:", len(results))

    if results.empty:
        print("No chart patterns detected.")
    else:
        print("\nPattern counts:")
        print(results["Pattern"].value_counts())

        print("\nCounts by ticker:")
        print(results.groupby(["Ticker", "Pattern"]).size())

        print("\nSample detections:")
        print(results.head(20))

        out_path = REPO_ROOT / "TestData" / "chart_test_results.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSaved results to: {out_path}")

    print("\n✅ CHART PATTERN TEST DONE")
