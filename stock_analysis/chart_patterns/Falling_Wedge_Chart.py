from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    fit_line_params,
    fit_line_pct_slope,
    line_value,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.falling_wedge_model import (
    load_model,
    learned_confidence,
    falling_wedge_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class FallingWedgeChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = "downtrend"

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

        pivots = detect_pivots(df, lookback=2, min_move_pct=0.005)

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

        high_pct = fit_line_pct_slope(high_x, high_y)
        low_pct = fit_line_pct_slope(low_x, low_y)

        high_neg_pct = max(-high_pct, 0.0)
        low_neg_pct = max(-low_pct, 0.0)

        if high_pct >= -0.00005 or low_pct >= -0.00005:
            return empty_result

        if high_neg_pct <= low_neg_pct:
            return empty_result

        start_range = line_value(
            high_slope,
            high_intercept,
            0,
        ) - line_value(
            low_slope,
            low_intercept,
            0,
        )

        end_range = line_value(
            high_slope,
            high_intercept,
            len(df) - 1,
        ) - line_value(
            low_slope,
            low_intercept,
            len(df) - 1,
        )

        if start_range <= 0 or end_range <= 0:
            return empty_result

        range_ratio = safe_ratio(
            end_range,
            max(start_range, 1e-9),
        )

        if range_ratio > 1.00:
            return empty_result

        slope_gap_ratio = safe_ratio(
            high_neg_pct - low_neg_pct,
            max(abs(high_neg_pct), 1e-9),
        )

        slope_divergence_score = min(max(slope_gap_ratio / 0.6, 0.0), 1.0)

        if range_ratio <= 0.60:
            convergence_score = 1.0
        elif range_ratio <= 0.80:
            convergence_score = 0.8
        elif range_ratio <= 0.90:
            convergence_score = 0.6
        elif range_ratio <= 1.00:
            convergence_score = 0.4
        else:
            convergence_score = 0.0

        touch_richness_score = min((len(highs) + len(lows)) / 8.0, 1.0)

        touch_balance_ratio = safe_ratio(
            min(len(highs), len(lows)),
            max(len(highs), len(lows)),
        )

        rule_score = falling_wedge_score(
            high_neg_pct_slope=high_neg_pct,
            low_neg_pct_slope=low_neg_pct,
            slope_divergence_score=slope_divergence_score,
            convergence_score=convergence_score,
            touch_richness_score=touch_richness_score,
            touch_balance_ratio=touch_balance_ratio,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    high_neg_pct_slope=high_neg_pct,
                    low_neg_pct_slope=low_neg_pct,
                    slope_divergence_score=slope_divergence_score,
                    convergence_score=convergence_score,
                    touch_richness_score=touch_richness_score,
                    touch_balance_ratio=touch_balance_ratio,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        resistance_now = line_value(
            high_slope,
            high_intercept,
            len(df) - 1,
        )

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