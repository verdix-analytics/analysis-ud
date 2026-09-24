from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    fit_line_params,
    fit_line_pct_slope,
    count_line_touches,
    safe_ratio,
    line_value,
)
from stock_analysis.confidence_models.chart_patterns.ascending_channel_model import (
    load_model,
    learned_confidence,
    ascending_channel_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class AscendingChannelChartPattern(BasePattern):
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

        upper_slope, upper_intercept = fit_line_params(high_x, high_y)
        lower_slope, lower_intercept = fit_line_params(low_x, low_y)

        upper_pct = fit_line_pct_slope(high_x, high_y)
        lower_pct = fit_line_pct_slope(low_x, low_y)

        if upper_pct <= 0.00005 or lower_pct <= 0.00005:
            return empty_result

        slope_gap = safe_ratio(
            abs(upper_pct - lower_pct),
            max(abs(upper_pct), abs(lower_pct), 1e-9),
        )

        if slope_gap > 0.60:
            return empty_result

        touch_high = count_line_touches(
            high_x,
            high_y,
            upper_slope,
            upper_intercept,
            tol_pct=0.015,
        )
        touch_low = count_line_touches(
            low_x,
            low_y,
            lower_slope,
            lower_intercept,
            tol_pct=0.015,
        )

        if touch_high < 2 or touch_low < 2:
            return empty_result

        channel_width_start = (
            upper_slope * 0 + upper_intercept
        ) - (
            lower_slope * 0 + lower_intercept
        )

        channel_width_end = (
            upper_slope * (len(df) - 1) + upper_intercept
        ) - (
            lower_slope * (len(df) - 1) + lower_intercept
        )

        width_change_ratio = safe_ratio(
            abs(channel_width_end - channel_width_start),
            max(abs(channel_width_start), 1e-9),
        )

        slope_parallel_score = max(0.0, 1.0 - min(slope_gap / 0.6, 1.0))
        touch_balance_ratio = safe_ratio(
            min(touch_high, touch_low),
            max(touch_high, touch_low),
        )
        width_stability_score = max(0.0, 1.0 - min(width_change_ratio / 0.5, 1.0))
        touch_richness_score = min((touch_high + touch_low) / 8.0, 1.0)

        rule_score = ascending_channel_score(
            upper_pct_slope=upper_pct,
            lower_pct_slope=lower_pct,
            slope_parallel_score=slope_parallel_score,
            touch_balance_ratio=touch_balance_ratio,
            width_stability_score=width_stability_score,
            touch_richness_score=touch_richness_score,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    upper_pct_slope=upper_pct,
                    lower_pct_slope=lower_pct,
                    slope_parallel_score=slope_parallel_score,
                    touch_balance_ratio=touch_balance_ratio,
                    width_stability_score=width_stability_score,
                    touch_richness_score=touch_richness_score,
                    model=self._model,
                )
                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        resistance_now = line_value(
            upper_slope,
            upper_intercept,
            len(df) - 1,
        )
        support_now = line_value(
            lower_slope,
            lower_intercept,
            len(df) - 1,
        )
        last_close = float(df.iloc[-1]["Close"])

        confirmed = confidence >= 50 and (
            last_close > float(resistance_now)
            or last_close < float(support_now)
        )

        if last_close > resistance_now:
            direction = "bullish"
        elif last_close < support_now:
            direction = "bearish"
        else:
            direction = "bullish"

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
            "direction": direction,
        }
