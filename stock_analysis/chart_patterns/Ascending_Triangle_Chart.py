from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    fit_line_params,
    fit_line_pct_slope,
    line_value,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.ascending_triangle_model import (
    load_model,
    learned_confidence,
    ascending_triangle_score,
)

import pandas as pd
import numpy as np


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class AscendingTriangleChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = "uptrend"

    def __init__(self):
        try:
            self._model = load_model()
        except Exception:
            self._model = None

    def _run(self, df: pd.DataFrame) -> dict:
        empty_result = {
            "confirmed": False,
            "confidence": 0,
            "pivots": [],
            "direction": "bullish",
        }

        if len(df) < 12:
            return empty_result

        pivots = detect_pivots(
            df,
            lookback=2,
            min_move_pct=0.005,
        )

        if pivots is None or len(pivots) < 4:
            return empty_result

        highs = pivots[pivots["pivot_type"] == "H"].copy()
        lows = pivots[pivots["pivot_type"] == "L"].copy()

        if len(highs) < 2 or len(lows) < 2:
            return empty_result

        high_x = highs["candle_index"].tolist()
        high_y = highs["pivot_price"].tolist()

        low_x = lows["candle_index"].tolist()
        low_y = lows["pivot_price"].tolist()

        high_slope, high_intercept = fit_line_params(high_x, high_y)
        low_slope, low_intercept = fit_line_params(low_x, low_y)

        high_pct_slope = fit_line_pct_slope(high_x, high_y)
        low_pct_slope = fit_line_pct_slope(low_x, low_y)

        high_abs_pct_slope = abs(high_pct_slope)

        high_std = float(np.std(high_y))
        high_mean = float(np.mean(high_y))
        high_deviation = safe_ratio(high_std, high_mean)

        resistance_now = line_value(high_slope, high_intercept, len(df) - 1)
        support_now = line_value(low_slope, low_intercept, len(df) - 1)

        avg_level = max((resistance_now + support_now) / 2, 1e-9)
        gap_pct_now = abs(resistance_now - support_now) / avg_level

        pivot_count = len(highs) + len(lows)
        touch_balance_ratio = safe_ratio(
            min(len(highs), len(lows)),
            max(len(highs), len(lows)),
        )

        if high_abs_pct_slope > 0.006:
            return empty_result

        if low_pct_slope <= 0.0:
            return empty_result

        if gap_pct_now > 0.12:
            return empty_result

        rule_score = ascending_triangle_score(
            high_abs_pct_slope=high_abs_pct_slope,
            low_pct_slope=low_pct_slope,
            high_deviation=high_deviation,
            gap_pct_now=gap_pct_now,
            pivot_count=pivot_count,
            touch_balance_ratio=touch_balance_ratio,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    high_abs_pct_slope=high_abs_pct_slope,
                    low_pct_slope=low_pct_slope,
                    high_deviation=high_deviation,
                    gap_pct_now=gap_pct_now,
                    pivot_count=pivot_count,
                    touch_balance_ratio=touch_balance_ratio,
                    model=self._model,
                )
                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        confirmed = (
            confidence >= 50
            and float(df.iloc[-1]["Close"]) > float(resistance_now)
        )

        best_pivots = []

        for j, (_, row) in enumerate(highs.iterrows(), start=1):
            pivot_idx = int(row["candle_index"])

            best_pivots.append({
                "type": "H",
                "price": float(row["pivot_price"]),
                "time": df.iloc[pivot_idx]["Datetime"],
                "label": f"H{j}",
            })

        for j, (_, row) in enumerate(lows.iterrows(), start=1):
            pivot_idx = int(row["candle_index"])

            best_pivots.append({
                "type": "L",
                "price": float(row["pivot_price"]),
                "time": df.iloc[pivot_idx]["Datetime"],
                "label": f"L{j}",
            })

        best_pivots = sorted(
            best_pivots,
            key=lambda x: pd.to_datetime(x["time"]),
        )

        return {
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 2),
            "pivots": best_pivots,
            "direction": "bullish",
        }
